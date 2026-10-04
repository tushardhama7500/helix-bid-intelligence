"""Scenarios A-F from the Phase 6 spec, driven by real change-language detection."""

import pytest
from pydantic import ValidationError

from app.extraction.schema import Source
from app.reconciliation.detect import detect_changes
from app.reconciliation.reconcile import ChangeRecord, OriginalValue, reconcile_field
from app.reconciliation.schema import ReconciledField, Timeline
from app.reconciliation.addenda import build_event, build_timeline


def src(chunk_id, file="doc.pdf", doc_type="bid_document", page=1) -> Source:
    return Source(chunk_id=chunk_id, file_name=file, document_type=doc_type, page_number=page, chunk_index=0)


def timeline(*numbers, reliable=True) -> Timeline:
    events = [
        build_event(document_id=100 + n, file_name=f"Addendum {n}.pdf", document_type="addendum", bid_id="B",
                    first_page=f"ADDENDUM No. {n}", full_text="")
        for n in numbers
    ]  # fmt: skip
    t = build_timeline("B", events)
    t.reliable = reliable and t.reliable
    return t


def from_text(text, terms, n, t: Timeline, chunk_id) -> list[ChangeRecord]:
    """What the service does: detect change language, attach the addendum's order and citation."""
    event = next(e for e in t.addenda if e.addendum_number == n)
    return [
        ChangeRecord(
            change_type=d.change_type, explanation=d.sentence, sources=[src(chunk_id, event.file_name, "addendum")],
            addendum_number=n, addendum_file=event.file_name, document_id=event.document_id, order=event.order,
            previous_value=d.previous_value, new_value=d.new_value,
        )
        for d in detect_changes(text, terms)
    ]  # fmt: skip


def run(original, changes, t, kind="scalar", key="x") -> ReconciledField:
    origs = [OriginalValue(original, [src(1)])] if original is not None else []
    return reconcile_field(key, key, origs, changes, t, kind)


# ---------------------------------------------------------------- A, B: replacements
def test_a_explicit_replacement() -> None:
    t = timeline(1)
    f = run("10 days", from_text("The delivery period is changed to 15 days.", ("delivery period",), 1, t, 5), t)

    assert f.final_value == "15 days" and f.original_value == "10 days"
    assert (f.status, f.resolution_method) == ("resolved", "explicit_addendum_change")
    assert f.changes[0].previous_value == "10 days" and f.changes[0].new_value == "15 days"
    assert f.sources[0].chunk_id == 5 and f.original_sources[0].chunk_id == 1


def test_b_from_to_amount() -> None:
    t = timeline(1)
    f = run("$10,000", from_text("Change the maximum amount from $10,000 to $15,000.", ("maximum amount",), 1, t, 5), t)
    assert f.final_value == "$15,000" and f.status == "resolved" and not f.notes


def test_stated_previous_value_that_disagrees_lowers_confidence() -> None:
    t = timeline(1)
    ok = run("$10,000", from_text("Change the maximum amount from $10,000 to $15,000.", ("maximum amount",), 1, t, 5), t)
    off = run("$12,000", from_text("Change the maximum amount from $10,000 to $15,000.", ("maximum amount",), 1, t, 5), t)
    assert off.final_value == "$15,000" and off.confidence < ok.confidence and "differs" in off.notes


# ------------------------------------------------------------------- C: clarification
def test_c_clarification_keeps_the_original_requirement() -> None:
    t = timeline(1)
    changes = from_text("For clarification, the warranty document may be submitted electronically.", ("warranty",), 1, t, 5)
    f = run("Warranty required", changes, t)

    assert changes[0].change_type == "clarification"
    assert f.final_value == "Warranty required"
    assert (f.status, f.resolution_method) == ("unchanged", "no_change")
    assert [c.change_type for c in f.changes] == ["clarification"]  # recorded in the history


def test_mentioning_a_field_is_not_a_change() -> None:
    assert detect_changes("Please refer to the original due date.", ("due date",)) == []
    assert detect_changes("Dallas ISD does not anticipate extending the submission deadline.", ("submission deadline",)) == []


# ---------------------------------------------------------------- D, E: several addenda
def test_d_multiple_addenda_are_applied_in_order_with_full_history() -> None:
    t = timeline(1, 2)
    changes = (
        from_text("The new due date will be July 1.", ("due date",), 2, t, 12)  # given out of order on purpose
        + from_text("The new due date will be June 25.", ("due date",), 1, t, 11)
    )
    f = run("June 20", changes, t)

    assert f.final_value == "July 1"
    assert (f.status, f.resolution_method) == ("resolved", "ordered_addendum")
    assert [(c.addendum_number, c.previous_value, c.new_value) for c in f.changes] == [
        (1, "June 20", "June 25"),
        (2, "June 25", "July 1"),
    ]
    assert f.original_value == "June 20" and f.sources[0].chunk_id == 12


def test_e_later_clarification_does_not_undo_an_earlier_change() -> None:
    t = timeline(1, 2)
    changes = (
        from_text("The warranty period is extended from 1 year to 3 years.", ("warranty",), 1, t, 11)
        + from_text("To clarify, the warranty covers parts and labor.", ("warranty",), 2, t, 12)
    )
    f = run("1 year warranty", changes, t)

    assert f.final_value == "3 years"
    assert [c.change_type for c in f.changes] == ["replacement", "clarification"]
    assert f.original_value == "1 year warranty" and f.resolution_method == "explicit_addendum_change"


def test_latest_addendum_wins_even_if_listed_first() -> None:
    t = timeline(1, 2)
    changes = from_text("The new due date will be July 1.", ("due date",), 2, t, 12) + \
        from_text("The new due date will be June 25.", ("due date",), 1, t, 11)  # fmt: skip
    assert run("June 20", list(reversed(changes)), t).final_value == "July 1"


# --------------------------------------------------------------- F: unresolved conflicts
def test_f_contradictory_originals_without_a_superseding_addendum_stay_a_conflict() -> None:
    t = timeline()
    f = reconcile_field(
        "due_date", "Due Date",
        [OriginalValue("June 27", [src(1, "rfp.pdf")]), OriginalValue("July 5", [src(2, "notice.pdf")])],
        [], t,
    )  # fmt: skip
    assert (f.status, f.final_value, f.resolution_method, f.confidence) == ("conflict", None, "unresolved", 0.0)
    assert [c.value for c in f.conflicting_values] == ["June 27", "July 5"]
    assert f.original_value is None


def test_conflicting_addenda_with_unknown_order_are_not_guessed() -> None:
    t = timeline(1, 2, reliable=False)
    changes = from_text("The new due date will be June 25.", ("due date",), 1, t, 11) + \
        from_text("The new due date will be July 1.", ("due date",), 2, t, 12)  # fmt: skip
    for c in changes:
        c.order = None
    f = run("June 20", changes, t)
    assert (f.status, f.final_value) == ("conflict", None)
    assert f.original_value == "June 20" and len(f.changes) == 2  # history kept


def test_unordered_addenda_that_agree_still_resolve() -> None:
    t = timeline(1, 2, reliable=False)
    changes = from_text("The new due date will be July 1.", ("due date",), 1, t, 11) + \
        from_text("The new due date will be July 1.", ("due date",), 2, t, 12)  # fmt: skip
    for c in changes:
        c.order = None
    assert run("June 20", changes, t).final_value == "July 1"


def test_one_addendum_contradicting_itself_is_a_conflict() -> None:
    t = timeline(1)
    changes = from_text("The new due date will be June 25. The new due date will be June 30.", ("due date",), 1, t, 11)
    f = run("June 20", changes, t)
    assert f.status == "conflict" and f.final_value is None


def test_addendum_override_resolves_disagreeing_originals_but_keeps_them() -> None:
    t = timeline(1)
    f = reconcile_field(
        "due_date", "Due Date",
        [OriginalValue("June 27", [src(1)]), OriginalValue("July 5", [src(2)])],
        from_text("The new due date will be July 9.", ("due date",), 1, t, 11), t,
    )  # fmt: skip
    assert f.final_value == "July 9" and f.status == "resolved"
    assert {c.value for c in f.conflicting_values} == {"June 27", "July 5"}


# ------------------------------------------------------------------------ other cases
def test_removal_and_list_additions() -> None:
    t = timeline(1, 2)
    removed = run("5% bid bond", from_text("The bid bond requirement is deleted.", ("bid bond",), 1, t, 11), t)
    assert removed.status == "resolved" and removed.final_value is None and "removes" in removed.notes

    added = run(["Mercury Affidavit"], from_text("All vendors must also submit Form X.", ("form",), 2, t, 12), t, kind="list")
    assert added.status == "unchanged" and added.final_value == ["Mercury Affidavit", "Form X"]
    assert added.changes[0].change_type == "addition"


def test_scalar_addition_does_not_change_the_value() -> None:
    t = timeline(1)
    f = run("Warranty required", from_text("Vendors must also submit Form X.", ("form",), 1, t, 11), t)
    assert f.final_value == "Warranty required" and f.status == "unchanged"


def test_missing_original_is_preserved_as_unknown_with_lower_confidence() -> None:
    t = timeline(1)
    with_orig = run("June 20", from_text("The due date is changed from June 20 to June 25.", ("due date",), 1, t, 11), t)
    without = run(None, from_text("The due date is changed from June 20 to June 25.", ("due date",), 1, t, 11), t)
    assert without.final_value == "June 25" and without.original_value is None
    assert without.confidence < with_orig.confidence and "stated the previous value" in (without.notes or "").replace("itself states", "stated")


def test_ambiguous_wording_is_surfaced_not_guessed() -> None:
    t = timeline(1)
    unknown = from_text("Replace Section 3.2 with the following.", ("section 3.2",), 1, t, 11)
    assert unknown[0].change_type == "unknown"
    f = run("Original text", unknown, t)
    assert (f.status, f.final_value, f.original_value) == ("insufficient_evidence", None, "Original text")


def test_no_evidence_at_all() -> None:
    f = run(None, [], timeline())
    assert (f.status, f.resolution_method, f.final_value, f.confidence) == ("insufficient_evidence", "unresolved", None, 0.0)


def test_unchanged_without_addenda_keeps_original_and_citation() -> None:
    f = run("10-JUN-2024", [], timeline())
    assert f.status == "unchanged" and f.final_value == "10-JUN-2024" and f.sources[0].chunk_id == 1


# -------------------------------------------------------------------- schema invariants
def test_reconciled_field_invariants() -> None:
    with pytest.raises(ValidationError):  # a final value must be cited
        ReconciledField(field="f", key="f", final_value="x", sources=[], status="unchanged", resolution_method="no_change")
    with pytest.raises(ValidationError):  # a conflict has no final value
        ReconciledField(field="f", key="f", final_value="x", sources=[src(1)], status="conflict", resolution_method="unresolved")
    with pytest.raises(ValidationError):  # resolved needs a change in the history
        ReconciledField(field="f", key="f", final_value="x", sources=[src(1)], status="resolved", resolution_method="explicit_addendum_change")
    with pytest.raises(ValidationError):  # confidence range
        ReconciledField(field="f", key="f", status="unchanged", resolution_method="no_change", confidence=1.5)


@pytest.mark.parametrize(
    ("text", "terms", "kind", "prev", "new"),
    [
        ("The new due date for this RFP will be July 9, 2024 at 2:00 PM CST.", ("due date",), "replacement", None, "July 9, 2024 at 2:00 PM CST"),
        ("The due date is hereby replaced with June 25, 2024.", ("due date",), "replacement", None, "June 25, 2024"),
        ("The pre-bid meeting is amended to 11-JUN-2024 10:00:00.", ("pre-bid meeting",), "replacement", None, "11-JUN-2024 10:00:00"),
        ("The deadline is extended from June 1 to June 8.", ("deadline",), "replacement", "June 1", "June 8"),
        ("Net 45 payment terms instead of Net 30.", ("payment terms",), "replacement", "Net 30", "Net 45 payment terms"),
        ("The bid bond requirement is waived.", ("bid bond",), "removal", None, None),
        ("Replace Section 3.2 with the following.", ("section 3.2",), "unknown", None, None),
        ("The deadline was extended to allow vendors to complete a compliant response.", ("deadline",), "unknown", None, None),
    ],
)
def test_change_language_detection(text, terms, kind, prev, new) -> None:
    (d,) = detect_changes(text, terms)
    assert (d.change_type, d.previous_value, d.new_value) == (kind, prev, new)
