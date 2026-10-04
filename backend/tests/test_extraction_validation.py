import pytest

from app.extraction.fields import FIELDS_BY_KEY
from app.extraction.schema import CONFLICT_NOTE, LLMCandidate, LLMField
from app.extraction.validation import compute_confidence, grounding, resolve_field, validate_citations
from tests.extraction_fakes import BID1, BID2

RETRIEVED = {c.chunk_id: c for c in BID1}


def field(key, *cands, confidence=0.9) -> LLMField:
    return LLMField(key=key, candidates=[LLMCandidate(value=v, chunk_ids=ids) for v, ids in cands], confidence=confidence)


# ---------------------------------------------------------------- citations
def test_validate_citations_splits_valid_and_unknown_ids() -> None:
    assert validate_citations([2, 999, 2, 1], RETRIEVED) == ([2, 1], [999])
    assert validate_citations([], RETRIEVED) == ([], [])


def test_unknown_chunk_id_is_removed_and_field_unsupported_without_valid_ones() -> None:
    result = resolve_field(FIELDS_BY_KEY["due_date"], field("due_date", ("July 9, 2024", [999])), RETRIEVED)

    assert result.status == "unsupported" and result.value is None
    assert result.sources == [] and result.confidence == 0.0
    assert "999" in result.notes


def test_chunk_from_another_bid_or_field_is_rejected() -> None:
    other_bid_chunk = BID2[0].chunk_id  # exists in the database, but was not retrieved for this field
    result = resolve_field(FIELDS_BY_KEY["due_date"], field("due_date", ("45 days", [other_bid_chunk])), RETRIEVED)
    assert result.status == "unsupported"


def test_missing_citation_is_unsupported() -> None:
    result = resolve_field(FIELDS_BY_KEY["due_date"], field("due_date", ("July 9, 2024", [])), RETRIEVED)
    assert result.status == "unsupported" and result.sources == []


def test_partially_invalid_citations_keep_valid_ones_and_lower_confidence() -> None:
    clean = resolve_field(FIELDS_BY_KEY["due_date"], field("due_date", ("July 9, 2024 at 2:00 PM CST", [2])), RETRIEVED)
    mixed = resolve_field(FIELDS_BY_KEY["due_date"], field("due_date", ("July 9, 2024 at 2:00 PM CST", [2, 999])), RETRIEVED)

    assert [s.chunk_id for s in mixed.sources] == [2]
    assert mixed.status == "found" and mixed.confidence < clean.confidence


def test_sources_are_built_from_the_stored_chunk_not_from_the_model() -> None:
    result = resolve_field(FIELDS_BY_KEY["due_date"], field("due_date", ("July 9, 2024", [2])), RETRIEVED)
    s = result.sources[0]
    assert (s.chunk_id, s.file_name, s.document_type, s.page_number, s.chunk_index) == (
        2, "Addendum 2 RFP JA-207652.pdf", "addendum", 1, 0,
    )  # fmt: skip


# ----------------------------------------------------------------- grounding
def test_numbers_must_appear_in_the_cited_evidence() -> None:
    spec = FIELDS_BY_KEY["due_date"]
    assert resolve_field(spec, field("due_date", ("July 9, 2024", [2])), RETRIEVED).status == "found"
    invented = resolve_field(spec, field("due_date", ("July 10, 2024", [2])), RETRIEVED)
    assert invented.status == "unsupported" and invented.value is None


def test_date_format_and_leading_zeros_are_tolerated() -> None:
    assert grounding("June 27, 2024 14:00", [RETRIEVED[1]], "scalar") is not None  # 27-JUN-2024 14:00:00
    assert grounding("June 28, 2024", [RETRIEVED[1]], "scalar") is None


def test_words_unrelated_to_the_evidence_are_rejected() -> None:
    result = resolve_field(FIELDS_BY_KEY["title"], field("title", ("Quantum Spaceship Procurement", [1])), RETRIEVED)
    assert result.status == "unsupported"


# ------------------------------------------------------------------ conflicts
def test_two_different_values_are_a_conflict_and_none_is_chosen() -> None:
    result = resolve_field(
        FIELDS_BY_KEY["due_date"],
        field("due_date", ("27-JUN-2024 14:00:00", [1]), ("July 9, 2024 at 2:00 PM CST", [2])),
        RETRIEVED,
    )

    assert result.status == "conflict"
    assert result.value is None and result.confidence == 0.0 and result.notes == CONFLICT_NOTE
    assert {s.chunk_id for s in result.sources} == {1, 2}
    values = [c.value for c in result.conflicting_values]
    assert values == ["27-JUN-2024 14:00:00", "July 9, 2024 at 2:00 PM CST"]
    assert [[s.file_name for s in c.sources] for c in result.conflicting_values] == [
        ["JA-207652 FINAL.pdf"], ["Addendum 2 RFP JA-207652.pdf"],
    ]  # fmt: skip


def test_same_value_repeated_is_not_a_conflict() -> None:
    result = resolve_field(
        FIELDS_BY_KEY["due_date"],
        field("due_date", ("July 9, 2024", [2]), ("july 9, 2024", [2, 3])),
        RETRIEVED,
    )
    assert result.status == "found" and result.value == "July 9, 2024"
    assert {s.chunk_id for s in result.sources} == {2, 3}


def test_unsupported_candidate_does_not_create_a_conflict() -> None:
    result = resolve_field(
        FIELDS_BY_KEY["due_date"],
        field("due_date", ("July 9, 2024", [2]), ("August 1, 2024", [999])),
        RETRIEVED,
    )
    assert result.status == "found" and result.value == "July 9, 2024"


def test_no_candidates_is_not_found() -> None:
    result = resolve_field(FIELDS_BY_KEY["bid_bond"], field("bid_bond", confidence=0.0), RETRIEVED)
    assert (result.status, result.value, result.sources, result.confidence) == ("not_found", None, [], 0.0)
    assert result.notes == "Not found in the retrieved evidence"
    assert resolve_field(FIELDS_BY_KEY["bid_bond"], None, RETRIEVED).status == "not_found"


# ----------------------------------------------------------- shapes by kind
def test_list_fields_merge_candidates_without_conflict() -> None:
    result = resolve_field(
        FIELDS_BY_KEY["additional_documentation"],
        field("additional_documentation", (["Mercury Affidavit"], [11]), (["Mercury Affidavit", "Delivery within 45 days"], [11])),
        {c.chunk_id: c for c in BID2},
    )
    assert result.status == "found"
    assert result.value == ["Mercury Affidavit", "Delivery within 45 days"]


def test_spec_field_is_structured_and_checked() -> None:
    chunks = {c.chunk_id: c for c in BID2}
    good = {"product": "Dell Latitude 5550", "specifications": {"processor": "Intel Core Ultra 5 125U", "memory": "16 GB DDR5"}}
    result = resolve_field(FIELDS_BY_KEY["product_specification"], field("product_specification", (good, [12])), chunks)
    assert result.status == "found" and result.value["specifications"]["memory"] == "16 GB DDR5"

    bad = {"product": "Dell Latitude 5550", "specifications": {"memory": "64 GB DDR5"}}
    assert resolve_field(FIELDS_BY_KEY["product_specification"], field("product_specification", (bad, [12])), chunks).status == "unsupported"
    wrong_shape = resolve_field(FIELDS_BY_KEY["product_specification"], field("product_specification", ("just text", [12])), chunks)
    assert wrong_shape.status == "unsupported"


# ----------------------------------------------------------------- confidence
def test_confidence_is_capped_penalised_and_clamped() -> None:
    assert compute_confidence(1.0, 1.0, 0, "scalar") == 0.95  # never certain
    assert compute_confidence(0.5, 1.0, 0, "scalar") == 0.5  # never above the model's own estimate
    assert compute_confidence(0.9, 0.7, 0, "scalar") == 0.66  # weaker grounding lowers it
    assert compute_confidence(0.9, 1.0, 1, "scalar") == 0.7  # dropped citation
    assert compute_confidence(0.1, 1.0, 2, "scalar") == 0.0  # floored
    assert compute_confidence(1.0, 1.0, 0, "text") == 0.8  # generated text is capped lower


@pytest.mark.parametrize("llm_conf", [0.0, 0.3, 1.0])
def test_final_confidence_never_exceeds_one_or_model_estimate(llm_conf) -> None:
    result = resolve_field(FIELDS_BY_KEY["title"], field("title", ("Student and Staff Computing Devices", [1]), confidence=llm_conf), RETRIEVED)
    assert 0.0 <= result.confidence <= min(llm_conf, 0.95)
