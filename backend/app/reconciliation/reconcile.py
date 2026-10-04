"""Pure reconciliation logic: original value + ordered addendum changes -> reconciled field.

No I/O and no LLM here, so every rule is unit-testable. The caller (service) is
responsible for supplying only *validated* changes: cited chunks must have been
retrieved and new values must pass the Phase 5 grounding check.
"""

from dataclasses import dataclass, field
from typing import Any

from app.extraction.schema import ConflictingValue, Source
from app.extraction.validation import comparison_key, flatten
from app.reconciliation.schema import ReconciledField, ReconciliationChange, Timeline

# Confidence ceilings per resolution path. Heuristics, not calibrated probabilities.
EXPLICIT_CONFIDENCE = 0.95
ORDERED_CONFIDENCE = 0.90
LLM_CONFIDENCE_CEILING = 0.80
UNCHANGED_CONFIDENCE = 0.75
UNCHANGED_CEILING = 0.85
ORIGINAL_UNKNOWN_PENALTY = 0.10
PREVIOUS_MISMATCH_PENALTY = 0.10
AMBIGUITY_PENALTY = 0.20

VALUE_CHANGING = ("replacement", "removal")


@dataclass
class OriginalValue:
    value: Any
    sources: list[Source] = field(default_factory=list)
    confidence: float | None = None


@dataclass
class ChangeRecord:
    change_type: str
    explanation: str
    sources: list[Source]
    addendum_number: int | None = None
    addendum_file: str | None = None
    document_id: int | None = None
    order: int | None = None  # position in the timeline, only when reliably established
    previous_value: Any | None = None  # as stated by the addendum itself, if it said
    new_value: Any | None = None
    method: str = "rule"  # "rule" | "llm"
    llm_confidence: float | None = None


def _same(a: Any, b: Any) -> bool:
    return comparison_key(a) == comparison_key(b)


def _unique_originals(originals: list[OriginalValue]) -> list[OriginalValue]:
    merged: dict[str, OriginalValue] = {}
    for item in originals:
        key = comparison_key(item.value)
        if key in merged:
            known = {s.chunk_id for s in merged[key].sources}
            merged[key].sources += [s for s in item.sources if s.chunk_id not in known]
        else:
            merged[key] = OriginalValue(item.value, list(item.sources), item.confidence)
    return list(merged.values())


def _record(c: ChangeRecord, previous: Any) -> ReconciliationChange:
    return ReconciliationChange(
        addendum_number=c.addendum_number,
        addendum_file=c.addendum_file,
        change_type=c.change_type,  # type: ignore[arg-type]
        previous_value=previous if previous is not None else c.previous_value,
        new_value=c.new_value,
        explanation=c.explanation,
        sources=c.sources,
        method=c.method,  # type: ignore[arg-type]
    )


def _apply(state: Any, change: ChangeRecord, kind: str) -> Any:
    if change.change_type == "removal":
        return None
    if change.change_type == "replacement":
        return [change.new_value] if kind == "list" else change.new_value
    if change.change_type == "addition" and kind == "list" and change.new_value:
        items = list(state or [])
        return items if change.new_value in items else items + [change.new_value]
    return state


def _group_by_addendum(changes: list[ChangeRecord]) -> list[list[ChangeRecord]]:
    groups: dict[Any, list[ChangeRecord]] = {}
    for c in changes:
        groups.setdefault(c.document_id if c.document_id is not None else ("n", c.addendum_number), []).append(c)
    return list(groups.values())


def reconcile_field(
    key: str,
    name: str,
    originals: list[OriginalValue],
    changes: list[ChangeRecord],
    timeline: Timeline,
    kind: str = "scalar",
) -> ReconciledField:
    """Decide the authoritative value of one field.

    Rules, in order:
      * only explicit replacements/removals change a value; clarifications and
        additions are recorded but leave the original in force (additions extend list fields);
      * one addendum changing the value wins over the original;
      * several addenda changing it are applied oldest to newest, but only when
        the timeline establishes their order; otherwise, unless they agree, it is a conflict;
      * contradictory originals with no superseding addendum stay a conflict;
      * ambiguous change language that nothing could interpret is surfaced, never guessed.
    """
    originals = _unique_originals(originals)
    originals_conflict = kind == "scalar" and len(originals) > 1
    original = originals[0] if originals and not originals_conflict else None
    original_value = original.value if original else None
    original_sources = original.sources if original else [s for o in originals for s in o.sources]
    conflicting = [ConflictingValue(value=o.value, sources=o.sources) for o in originals] if originals_conflict else []

    ordered_changes = sorted(changes, key=lambda c: (c.order is None, c.order or 0))
    value_changes = [c for c in ordered_changes if c.change_type in VALUE_CHANGING]
    unknown = [c for c in ordered_changes if c.change_type == "unknown"]

    def build(**kw: Any) -> ReconciledField:
        base = dict(
            field=name, key=key, original_value=original_value, original_sources=original_sources,
            conflicting_values=conflicting,
        )  # fmt: skip
        return ReconciledField(**{**base, **kw})

    history = [_record(c, original_value) for c in ordered_changes]

    # ------------------------------------------------ no value-changing addendum
    if not value_changes:
        extra_additions = [c for c in ordered_changes if c.change_type == "addition" and kind == "list" and c.new_value]
        if originals_conflict:
            return build(
                changes=history, final_value=None, sources=[], status="conflict", resolution_method="unresolved",
                confidence=0.0,
                notes="Original documents disagree and no addendum establishes which value applies; not guessing.",
            )  # fmt: skip
        if not original and not extra_additions:
            return build(
                changes=history, final_value=None, sources=[], status="insufficient_evidence",
                resolution_method="unresolved", confidence=0.0,
                notes="No original value and no explicit addendum change were found in the evidence.",
            )  # fmt: skip
        if unknown:
            return build(
                changes=history, final_value=None, sources=[], status="insufficient_evidence",
                resolution_method="unresolved", confidence=0.0,
                notes="An addendum contains wording that may modify this field but its effect could not be determined; "
                      "the original value is preserved above.",
            )  # fmt: skip
        state = original_value
        for c in extra_additions:
            state = _apply(state, c, kind)
        sources = list(original_sources)
        confidence = min(original.confidence if original and original.confidence is not None else UNCHANGED_CONFIDENCE, UNCHANGED_CEILING)
        notes = "No addendum changes this field." if not ordered_changes else \
            "Addenda mention this field without changing it (clarification/addition only)."
        if extra_additions:
            sources += [s for c in extra_additions for s in c.sources]
            notes = "Addenda add items to this list field; nothing was removed or replaced."
        return build(
            changes=history, final_value=state, sources=sources, status="unchanged", resolution_method="no_change",
            confidence=round(confidence, 2) if state is not None else 0.0, notes=notes,
        )  # fmt: skip

    # ------------------------------------------------- value-changing addenda
    groups = _group_by_addendum(value_changes)
    for group in groups:  # one addendum contradicting itself
        if len({comparison_key(c.new_value) + c.change_type for c in group}) > 1:
            return build(
                changes=history, final_value=None, sources=[], status="conflict", resolution_method="unresolved",
                confidence=0.0,
                notes=f"Addendum {group[0].addendum_number or '?'} gives contradictory new values; not guessing.",
            )  # fmt: skip

    representatives = [g[0] for g in groups]
    ordered = len(representatives) > 1
    if ordered:
        orders = [c.order for c in representatives]
        order_known = timeline.reliable and None not in orders and len(set(orders)) == len(orders)
        agree = len({comparison_key(c.new_value) + c.change_type for c in representatives}) == 1
        if not order_known and not agree:
            return build(
                changes=history, final_value=None, sources=[], status="conflict", resolution_method="unresolved",
                confidence=0.0,
                notes="Several addenda change this field differently and their order could not be established; not guessing.",
            )  # fmt: skip
        representatives.sort(key=lambda c: (c.order is None, c.order or 0))

    # Rebuild the history applying changes in order so each previous_value is the prior state.
    state = original_value
    applied: list[ReconciliationChange] = []
    mismatch = False
    for c in sorted(ordered_changes, key=lambda c: (c.order is None, c.order or 0)):
        before = state
        if c in representatives or c.change_type not in VALUE_CHANGING:
            if c.change_type == "replacement" and c.previous_value is not None and before is not None and \
                    not _same(c.previous_value, before) and not _same(flatten(c.previous_value), flatten(before)):
                mismatch = True
            if c in representatives or c.change_type == "addition":
                state = _apply(state, c, kind)
        applied.append(_record(c, before))

    last = representatives[-1]
    method = "llm_reconciliation" if any(c.method == "llm" for c in representatives) else (
        "ordered_addendum" if ordered else "explicit_addendum_change"
    )
    if method == "llm_reconciliation":
        confidence = min(LLM_CONFIDENCE_CEILING, *(c.llm_confidence if c.llm_confidence is not None else LLM_CONFIDENCE_CEILING for c in representatives))
    else:
        confidence = ORDERED_CONFIDENCE if ordered else EXPLICIT_CONFIDENCE

    notes: list[str] = []
    if original is None:
        confidence -= ORIGINAL_UNKNOWN_PENALTY
        notes.append("The original value was not located in the evidence.")
    if mismatch:
        confidence -= PREVIOUS_MISMATCH_PENALTY
        notes.append("The addendum's stated previous value differs from the original value found.")
    if unknown:
        confidence -= AMBIGUITY_PENALTY
        notes.append("Other ambiguous addendum wording about this field could not be interpreted.")
    if last.change_type == "removal":
        notes.append("An addendum removes this requirement.")
    if originals_conflict:
        notes.append("The original documents disagreed; the addendum overrides them.")
    if original is None and not originals_conflict and last.previous_value is not None:
        notes.append(f"The addendum itself states the previous value as: {flatten(last.previous_value)}")

    return build(
        changes=applied,
        final_value=state,
        sources=list(last.sources) if state is not None or last.sources else [],
        status="resolved",
        resolution_method=method,
        confidence=round(max(0.0, min(1.0, confidence)), 2),
        notes=" ".join(notes) or None,
    )
