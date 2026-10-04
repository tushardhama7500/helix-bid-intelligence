"""Pipeline state and result models. Plain Pydantic: no state-management framework."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.extraction.schema import BidExtraction, Source
from app.reconciliation.schema import BidReconciliation

StageState = Literal["complete", "partial", "failed", "skipped"]
PipelineStatus = Literal["complete", "partial", "failed"]
FailureKind = Literal["infrastructure", "no_documents", "no_evidence", "validation", "internal"]
STAGES = ("ingestion", "retrieval", "extraction", "reconciliation", "validation")

REFUSAL = "I could not find sufficient evidence to answer this question."


class StageResult(BaseModel):
    name: str
    state: StageState = "skipped"
    duration_ms: int = 0
    detail: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ ingestion
class DocumentInfo(BaseModel):
    document_id: int
    file_name: str
    document_type: str
    status: str


class IngestionResult(BaseModel):
    bid_id: str
    files_found: int = 0
    ingested: int = 0  # (re)processed in this run
    skipped: int = 0  # unchanged, left alone
    failed_files: list[str] = Field(default_factory=list)
    documents: list[DocumentInfo] = Field(default_factory=list)
    chunks_embedded: int = 0
    embedding_failures: int = 0
    notes: list[str] = Field(default_factory=list)

    @property
    def completed_documents(self) -> list[DocumentInfo]:
        return [d for d in self.documents if d.status == "completed"]


class RetrievalStageResult(BaseModel):
    bid_id: str
    fields: int = 0
    total_chunks: int = 0
    evidence_counts: dict[str, int] = Field(default_factory=dict)
    empty_fields: list[str] = Field(default_factory=list)
    dropped_foreign_chunks: int = 0  # chunks from another bid that were discarded (should be 0)


# ----------------------------------------------------------------- validation
class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    stage: Literal["extraction", "reconciliation", "pipeline"]
    field: str | None = None
    code: str
    message: str


class ValidationReport(BaseModel):
    valid: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)
    checked: dict[str, int] = Field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


# --------------------------------------------------------------------- report
class ReportField(BaseModel):
    field: str
    value: Any | None = None
    status: str  # found | not_found | unchanged | resolved | conflict | unresolved | unsupported | error | not_extracted
    origin: Literal["reconciliation", "extraction", "none"] = "none"
    sources: list[Source] = Field(default_factory=list)
    confidence: float = 0.0
    original_value: Any | None = None
    original_sources: list[Source] = Field(default_factory=list)
    changed_by_addendum: bool = False
    notes: str | None = None


class PipelineReport(BaseModel):
    bid_id: str
    fields: list[ReportField]
    unresolved: list[str] = Field(default_factory=list)


# -------------------------------------------------------------- state / result
class BidPipelineState(BaseModel):
    bid_id: str
    run_id: str = "-"  # correlation id shared by every log line of this run
    ingestion: IngestionResult | None = None
    retrieval: RetrievalStageResult | None = None
    extraction: BidExtraction | None = None
    reconciliation: BidReconciliation | None = None
    validation: ValidationReport | None = None
    report: PipelineReport | None = None

    stages: dict[str, StageResult] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fatal: bool = False
    failure: FailureKind | None = None  # why the pipeline failed (drives the HTTP status)
    status: PipelineStatus = "failed"
    complete: bool = False
    timings_ms: dict[str, int] = Field(default_factory=dict)  # per stage + total; for diagnostics, not shown in the UI


class PipelineResponse(BaseModel):
    """What the API exposes: stage outcomes and the validated report, not internals."""

    bid_id: str
    run_id: str = "-"
    status: PipelineStatus
    stages: dict[str, StageState]
    failure: FailureKind | None = None
    errors: list[str]
    warnings: list[str]
    validation: ValidationReport | None
    result: PipelineReport | None  # withheld (null) when validation found errors
    timings_ms: dict[str, int] = Field(default_factory=dict)


# ------------------------------------------------------------------------ Q&A
class QASource(BaseModel):
    file: str
    page: int | None
    chunk_id: int
    document_type: str
    chunk_index: int
    section: str | None = None


class QAAnswer(BaseModel):
    bid_id: str
    question: str
    answer: str
    answerable: bool
    sources: list[QASource] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    used_reconciliation: bool = False
    notes: str | None = None


class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
