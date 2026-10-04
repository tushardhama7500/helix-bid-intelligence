"""Pydantic models for addendum timelines and reconciled fields."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.extraction.schema import ConflictingValue, Source

ChangeType = Literal["replacement", "clarification", "addition", "removal", "unknown"]
Status = Literal["resolved", "unchanged", "conflict", "insufficient_evidence"]
Method = Literal["explicit_addendum_change", "ordered_addendum", "llm_reconciliation", "no_change", "unresolved"]
Ordering = Literal["explicit_number", "date", "filename", "uncertain", "none"]


# ------------------------------------------------------------------ timeline
class AddendumEvent(BaseModel):
    """One addendum document, with what could (and could not) be established about it."""

    document_id: int
    file_name: str
    document_type: str
    bid_id: str
    addendum_number: int | None = None  # never guessed: None when the evidence does not say
    number_source: Literal["content", "filename"] | None = None
    date: str | None = None  # as written in the document; None when absent
    referenced_bid_number: str | None = None
    identified_by: Literal["document_type", "content"] = "document_type"
    changes_language: bool = False  # contains wording that modifies requirements
    clarifies_language: bool = False  # contains clarification / Q&A wording
    order: int | None = None  # 1-based position when the order is reliably established


class Timeline(BaseModel):
    bid_id: str
    original_documents: list[str] = Field(default_factory=list)  # non-addendum file names
    addenda: list[AddendumEvent] = Field(default_factory=list)  # in established order
    ordering: Ordering = "none"
    reliable: bool = True  # False when some addenda cannot be ordered with confidence
    notes: list[str] = Field(default_factory=list)


# ------------------------------------------------------------ field history
class ReconciliationChange(BaseModel):
    addendum_number: int | None
    addendum_file: str | None = None
    change_type: ChangeType
    previous_value: Any | None = None
    new_value: Any | None = None
    explanation: str
    sources: list[Source] = Field(default_factory=list)
    method: Literal["rule", "llm"] = "rule"


class ReconciledField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str  # assessment field name, e.g. "Due Date"
    key: str  # stable key, e.g. "due_date"
    original_value: Any | None = None
    original_sources: list[Source] = Field(default_factory=list)
    changes: list[ReconciliationChange] = Field(default_factory=list)
    final_value: Any | None = None
    sources: list[Source] = Field(default_factory=list)  # evidence for final_value
    conflicting_values: list[ConflictingValue] = Field(default_factory=list)
    status: Status
    resolution_method: Method
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str | None = None

    @model_validator(mode="after")
    def _invariants(self) -> "ReconciledField":
        if self.status in ("conflict", "insufficient_evidence"):
            if self.final_value is not None:
                raise ValueError(f"status '{self.status}' must not have a final value")
            if self.confidence != 0.0:
                raise ValueError(f"status '{self.status}' must have confidence 0")
        if self.final_value is not None and not self.sources:
            raise ValueError("a final value must be cited")
        if self.status == "resolved" and not self.changes:
            raise ValueError("a resolved field needs at least one change")
        return self


class BidReconciliation(BaseModel):
    bid_id: str
    timeline: Timeline
    fields: list[ReconciledField]
    # Fields where no authoritative value could be determined (conflict / insufficient evidence).
    reconciliation_required: list[str] = Field(default_factory=list)
    skipped_fields: dict[str, str] = Field(default_factory=dict)  # field name -> why it is not reconciled
    errors: list[str] = Field(default_factory=list)
    complete: bool = True
    used_llm: bool = False
    model: str | None = None


# ---------------------------------------------------- Groq fallback (raw output)
class LLMChange(BaseModel):
    change_type: Literal["replacement", "clarification", "addition", "removal", "none", "unknown"]
    previous_value: str | None = None
    new_value: str | None = None
    explanation: str = ""
    chunk_ids: list[int] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class LLMReconciliationOutput(BaseModel):
    changes: list[LLMChange]
