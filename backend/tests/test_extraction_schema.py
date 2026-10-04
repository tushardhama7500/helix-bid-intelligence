import pytest
from pydantic import ValidationError

from app.extraction.fields import FIELDS
from app.extraction.schema import (
    CONFLICT_NOTE,
    ConflictingValue,
    ExtractedField,
    LLMField,
    LLMGroupOutput,
    Source,
)
from app.extraction.service import parse_group_output

SRC = Source(chunk_id=1, file_name="a.pdf", document_type="addendum", page_number=1, chunk_index=0)


def test_all_assessment_fields_are_defined() -> None:
    assert [f.name for f in FIELDS] == [
        "Bid Number", "Title", "company_name", "contact_info", "Contract or Cooperative to use",
        "Due Date", "Pre Bid Meeting", "Delivery Date", "Term of Bid",
        "Bid Submission Type", "Bid Bond Requirement", "Payment Terms", "Installation",
        "Any Additional Documentation Required",
        "MFG for Registration", "Model_no", "Part_no", "Product", "Product Specification", "Bid Summary",
    ]  # fmt: skip


def test_valid_found_field() -> None:
    f = ExtractedField(value="July 9, 2024", sources=[SRC], confidence=0.9, status="found")
    assert f.model_dump()["sources"][0]["page_number"] == 1


def test_missing_value_defaults() -> None:
    f = ExtractedField.missing()
    assert (f.value, f.sources, f.confidence, f.status) == (None, [], 0.0, "not_found")
    assert f.notes == "Not found in the retrieved evidence"


@pytest.mark.parametrize("confidence", [-0.1, 1.01])
def test_confidence_must_be_between_0_and_1(confidence) -> None:
    with pytest.raises(ValidationError):
        ExtractedField(value="x", sources=[SRC], confidence=confidence, status="found")


def test_value_without_sources_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one source"):
        ExtractedField(value="x", sources=[], confidence=0.0, status="found")


def test_unsupported_field_cannot_have_confidence_or_sources() -> None:
    with pytest.raises(ValidationError):
        ExtractedField(value=None, sources=[], confidence=0.5, status="not_found")
    with pytest.raises(ValidationError):
        ExtractedField(value=None, sources=[SRC], confidence=0.0, status="not_found")


def test_conflict_needs_two_values_and_no_chosen_value() -> None:
    two = [ConflictingValue(value="A", sources=[SRC]), ConflictingValue(value="B", sources=[SRC])]
    ok = ExtractedField(value=None, sources=[SRC], confidence=0.0, status="conflict", conflicting_values=two, notes=CONFLICT_NOTE)
    assert ok.value is None
    with pytest.raises(ValidationError):
        ExtractedField(value=None, sources=[SRC], status="conflict", conflicting_values=two[:1])
    with pytest.raises(ValidationError):
        ExtractedField(value="A", sources=[SRC], confidence=0.5, status="conflict", conflicting_values=two)


def test_llm_output_schema_rejects_malformed_payloads() -> None:
    ok = parse_group_output('{"fields": [{"key": "title", "candidates": [{"value": "T", "chunk_ids": [1]}], "confidence": 0.8}]}')
    assert isinstance(ok, LLMGroupOutput) and ok.fields[0].candidates[0].chunk_ids == [1]

    for bad in (
        "not json at all",
        "[]",
        '{"fields": "nope"}',
        '{"fields": [{"key": "t", "confidence": 1.7}]}',
        '{"fields": [{"key": "t", "candidates": [{"value": "x", "chunk_ids": ["abc"]}]}]}',
        '{"fields": [{"candidates": []}]}',
    ):
        with pytest.raises(ValueError):
            parse_group_output(bad)


def test_llm_field_defaults() -> None:
    f = LLMField(key="x")
    assert f.candidates == [] and 0 <= f.confidence <= 1
