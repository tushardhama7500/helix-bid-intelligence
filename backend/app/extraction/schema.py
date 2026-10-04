"""Pydantic models: the LLM's raw output, and the validated API result."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Status = Literal["found", "not_found", "conflict", "unsupported", "error"]
Value = str | list[str] | dict[str, Any] | list[dict[str, Any]]  # list of dicts: several product specifications

CONFLICT_NOTE = (
    "Conflicting values found in the retrieved documents; addendum reconciliation is required."
)
NOT_FOUND_NOTE = "Not found in the retrieved evidence"


# ----------------------------------------------------------- what the LLM returns
class LLMCandidate(BaseModel):
    """One distinct value the evidence supports, with the chunks that support it."""

    value: str | int | float | list[str] | dict[str, Any]
    chunk_ids: list[int] = Field(default_factory=list)


class LLMField(BaseModel):
    key: str
    candidates: list[LLMCandidate] = Field(default_factory=list)
    # Preliminary self-assessment only; the application re-computes the final confidence.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str | None = None


class LLMGroupOutput(BaseModel):
    fields: list[LLMField]


# --------------------------------------------------------- what the API returns
class Source(BaseModel):
    """Where a value came from. Built by the backend from the stored chunk, never by the LLM."""

    chunk_id: int
    file_name: str
    document_type: str
    page_number: int | None  # None for HTML documents
    chunk_index: int
    section: str | None = None  # HTML section heading, when there is no page


class ConflictingValue(BaseModel):
    value: Value
    sources: list[Source]


class ExtractedField(BaseModel):
    """An extracted value with its evidence.

    `confidence` is an extraction-quality heuristic in [0, 1]; it is NOT a
    calibrated probability (see README, Phase 5).
    """

    model_config = ConfigDict(extra="forbid")

    value: Value | None = None
    sources: list[Source] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str | None = None
    status: Status = "not_found"
    conflicting_values: list[ConflictingValue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_invariants(self) -> "ExtractedField":
        if self.value is not None:
            if self.status != "found":
                raise ValueError("a non-null value must have status 'found'")
            if not self.sources:
                raise ValueError("a non-null value must have at least one source")
        else:
            if self.confidence != 0.0:
                raise ValueError("a field without a value must have confidence 0")
            if self.status == "found":
                raise ValueError("status 'found' requires a value")
            if self.status != "conflict" and self.sources:
                raise ValueError("a field without a value must not cite sources (except conflicts)")
        if self.status == "conflict" and len(self.conflicting_values) < 2:
            raise ValueError("a conflict needs at least two conflicting values")
        if self.status != "conflict" and self.conflicting_values:
            raise ValueError("conflicting_values is only allowed for status 'conflict'")
        if self.confidence > 0 and not self.sources:
            raise ValueError("positive confidence requires sources")
        return self

    @classmethod
    def missing(cls, status: Status = "not_found", notes: str | None = NOT_FOUND_NOTE) -> "ExtractedField":
        return cls(value=None, sources=[], confidence=0.0, notes=notes, status=status)


class BidExtraction(BaseModel):
    bid_id: str
    fields: dict[str, ExtractedField]
    # Fields whose documents disagree; resolving them is addendum reconciliation (Phase 6).
    reconciliation_required: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    complete: bool = True  # False when some fields could not be extracted because of errors
    model: str | None = None
    # Robustness counters for evaluation: extra model calls caused by malformed output / bad citations.
    schema_retries: int = 0
    citation_retries: int = 0
