"""Deterministic checks applied to everything the LLM returns.

The LLM only *proposes* values and chunk ids. This module decides what is kept:

1. Citations must be chunk ids that were actually retrieved for that field.
2. Every number in a value must appear in the cited text, and most of its words
   must too (so the model cannot invent or "calculate" figures).
3. Distinct supported values for a scalar field are a conflict; none is chosen.
4. Confidence is recomputed here, capped by how well the value is grounded.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.extraction.fields import FieldSpec
from app.extraction.schema import (
    CONFLICT_NOTE,
    NOT_FOUND_NOTE,
    ConflictingValue,
    ExtractedField,
    LLMCandidate,
    LLMField,
    Source,
    Value,
)
from app.retrieval.schema import Candidate

MIN_WORD_COVERAGE = 0.6  # share of a value's words that must occur in the cited chunks
CONFIDENCE_CEILING = 0.95  # even a fully grounded value is never reported as certain
MAX_SPEC_PRODUCTS = 8  # product lines kept in a Product Specification (bounds output and prompt size)
TEXT_CONFIDENCE_CEILING = 0.8  # generated summaries are less verifiable than copied values
DROPPED_CITATION_PENALTY = 0.2

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the and for with from that this are not must will per any all each their its has have been".split()
)


# ----------------------------------------------------------------- citations
def validate_citations(
    chunk_ids: Sequence[int], retrieved: Mapping[int, Candidate]
) -> tuple[list[int], list[int]]:
    """Split cited ids into (valid, invalid): valid ids are in `retrieved`. Order kept, no duplicates."""
    valid: list[int] = []
    invalid: list[int] = []
    for chunk_id in dict.fromkeys(chunk_ids):
        (valid if chunk_id in retrieved else invalid).append(chunk_id)
    return valid, invalid


def build_source(candidate: Candidate) -> Source:
    section = candidate.metadata.get("section")
    return Source(
        chunk_id=candidate.chunk_id,
        file_name=candidate.file_name,
        document_type=candidate.document_type,
        page_number=candidate.page_number,
        chunk_index=candidate.chunk_index,
        section=section if isinstance(section, str) else None,
    )


# ----------------------------------------------------------- value handling
def flatten(value: Any) -> str:
    """All the *content* of a value as one string (dict keys are labels, so skipped)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Mapping):
        return " ".join(flatten(v) for v in value.values())
    return " ".join(flatten(v) for v in value)


def normalize_value(raw: Any, kind: str) -> Value | None:
    """Coerce the model's value to the shape its field kind expects; None if unusable/empty."""
    if kind == "list":
        if isinstance(raw, Mapping):
            items = [f"{k}: {flatten(v)}" for k, v in raw.items()]
        elif isinstance(raw, (list, tuple)):
            items = [flatten(v) for v in raw]
        else:
            items = [flatten(raw)]
        items = [" ".join(i.split()) for i in items if i and i.strip()]
        return list(dict.fromkeys(items)) or None
    if kind == "spec":
        if not isinstance(raw, Mapping):
            return None
        specs = raw.get("specifications")
        if not isinstance(specs, Mapping):
            specs = {k: v for k, v in raw.items() if k != "product"}
        specs = {str(k).strip(): " ".join(flatten(v).split()) for k, v in specs.items() if flatten(v).strip()}
        product = raw.get("product")
        product = " ".join(flatten(product).split()) or None
        if not specs and not product:
            return None
        return {"product": product, "specifications": specs}
    # scalar / text
    text = "; ".join(flatten(v) for v in raw) if isinstance(raw, (list, tuple)) else flatten(raw)
    text = " ".join(text.split())
    return text or None


def comparison_key(value: Value) -> str:
    """Form used to decide whether two values are 'the same': case/punctuation-insensitive."""
    return " ".join(_TOKEN.findall(flatten(value).lower()))


def _canonical(token: str) -> str:
    return (token.lstrip("0") or "0") if token.isdigit() else token


def grounding(value: Value, chunks: Sequence[Candidate], kind: str) -> float | None:
    """How well `chunks` support `value`: a ratio in (0, 1], or None if not supported.

    Every number-like token (anything with a digit) in the value must occur in the
    cited chunks, with leading zeros ignored. For non-generated fields, at least
    MIN_WORD_COVERAGE of the alphabetic words (3+ letters, not stop-words) must also
    occur, allowing prefix matches so `June` matches `JUN`.
    """
    source_tokens = {_canonical(t) for c in chunks for t in _TOKEN.findall(c.content.lower())}
    value_tokens = _TOKEN.findall(flatten(value).lower())

    numbers = [t for t in value_tokens if any(ch.isdigit() for ch in t)]
    if any(_canonical(t) not in source_tokens for t in numbers):
        return None
    if kind == "text":  # a generated summary: only its figures can be checked mechanically
        return 1.0

    words = [t for t in value_tokens if t.isalpha() and len(t) >= 3 and t not in _STOPWORDS]
    if not words:
        return 1.0
    source_words = [t for t in source_tokens if t.isalpha() and len(t) >= 3]
    found = sum(
        1 for w in words if w in source_tokens or any(s.startswith(w) or w.startswith(s) for s in source_words)
    )
    ratio = found / len(words)
    return ratio if ratio >= MIN_WORD_COVERAGE else None


def compute_confidence(llm_confidence: float, ratio: float, dropped_citations: int, kind: str) -> float:
    """Extraction-quality score in [0, 1]. A heuristic, NOT a calibrated probability.

    = min(model's own estimate, ceiling x grounding ratio), minus a penalty when the
    model cited chunks that had to be discarded.
    """
    ceiling = TEXT_CONFIDENCE_CEILING if kind == "text" else CONFIDENCE_CEILING
    score = min(llm_confidence, ceiling * ratio, ceiling)
    if dropped_citations:
        score -= DROPPED_CITATION_PENALTY
    return round(max(0.0, min(1.0, score)), 2)


# ------------------------------------------------------------- field result
class _Kept:
    def __init__(self, value: Value, chunk_ids: list[int], ratio: float, dropped: int) -> None:
        self.value, self.chunk_ids, self.ratio, self.dropped = value, chunk_ids, ratio, dropped


def _sources(ids: Sequence[int], retrieved: Mapping[int, Candidate]) -> list[Source]:
    return [build_source(retrieved[i]) for i in dict.fromkeys(ids)]


def resolve_field(
    spec: FieldSpec, llm_field: LLMField | None, retrieved: Mapping[int, Candidate]
) -> ExtractedField:
    """Turn the model's proposal for one field into a validated `ExtractedField`.

    `retrieved` is the evidence that was sent to the model *for this field*; any
    citation outside it is rejected.
    """
    if llm_field is None:
        return ExtractedField.missing("not_found", "The model did not return this field")

    kept: list[_Kept] = []
    problems: list[str] = []
    for candidate in llm_field.candidates:
        item = _check_candidate(spec, candidate, retrieved, problems)
        if item is not None:
            kept.append(item)

    if not kept:
        if problems:
            return ExtractedField.missing(
                "unsupported", "Model output was not supported by the retrieved evidence: " + "; ".join(problems[:3])
            )
        return ExtractedField.missing("not_found", NOT_FOUND_NOTE)

    if spec.kind == "list":
        merged = list(dict.fromkeys(v for k in kept for v in k.value))  # type: ignore[union-attr]
        kept = [_Kept(merged, [i for k in kept for i in k.chunk_ids], min(k.ratio for k in kept), sum(k.dropped for k in kept))]
    elif spec.kind == "spec":
        # One candidate per product line, each grounded in its own cited chunks. Different products are not
        # a conflict; a single product keeps the original single-object shape.
        kept = _merge_equal_values(kept)[:MAX_SPEC_PRODUCTS]
        if len(kept) > 1:
            kept = [_Kept([k.value for k in kept], [i for k in kept for i in k.chunk_ids],
                          min(k.ratio for k in kept), sum(k.dropped for k in kept))]  # fmt: skip
    elif spec.kind != "scalar":
        kept = kept[:1]  # text: a single value

    if spec.kind == "scalar":
        kept = _merge_equal_values(kept)
    if len(kept) > 1:
        return _conflict(kept, retrieved)

    only = kept[0]
    return ExtractedField(
        value=only.value,
        sources=_sources(only.chunk_ids, retrieved),
        confidence=compute_confidence(llm_field.confidence, only.ratio, only.dropped, spec.kind),
        notes=llm_field.notes if llm_field.notes else None,
        status="found",
    )


def _check_candidate(
    spec: FieldSpec, candidate: LLMCandidate, retrieved: Mapping[int, Candidate], problems: list[str]
) -> _Kept | None:
    value = normalize_value(candidate.value, spec.kind)
    if value is None:
        if spec.kind == "spec":
            problems.append("value has the wrong shape")
        return None
    valid, invalid = validate_citations(candidate.chunk_ids, retrieved)
    if not valid:
        problems.append("no valid citation" + (f" (unknown chunk ids {invalid})" if invalid else ""))
        return None
    ratio = grounding(value, [retrieved[i] for i in valid], spec.kind)
    if ratio is None:
        problems.append("value is not contained in the cited evidence")
        return None
    return _Kept(value, valid, ratio, len(invalid))


def _merge_equal_values(kept: list[_Kept]) -> list[_Kept]:
    """The same value repeated (even with different wording/casing) counts once."""
    merged: dict[str, _Kept] = {}
    for item in kept:
        key = comparison_key(item.value)
        if key in merged:
            first = merged[key]
            first.chunk_ids = list(dict.fromkeys(first.chunk_ids + item.chunk_ids))
            first.ratio = min(first.ratio, item.ratio)
            first.dropped += item.dropped
        else:
            merged[key] = item
    return list(merged.values())


def _conflict(kept: list[_Kept], retrieved: Mapping[int, Candidate]) -> ExtractedField:
    """Several distinct supported values: report all of them; do not pick one."""
    all_ids = [i for k in kept for i in k.chunk_ids]
    return ExtractedField(
        value=None,
        sources=_sources(all_ids, retrieved),
        confidence=0.0,
        notes=CONFLICT_NOTE,
        status="conflict",
        conflicting_values=[ConflictingValue(value=k.value, sources=_sources(k.chunk_ids, retrieved)) for k in kept],
    )
