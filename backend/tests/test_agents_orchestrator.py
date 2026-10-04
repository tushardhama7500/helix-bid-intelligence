"""Orchestrator behaviour with stub agents: ordering, failure classification, status."""

import pytest
from sqlalchemy.exc import OperationalError

from app.agents.orchestrator import Orchestrator
from app.agents.schema import IngestionResult, DocumentInfo, RetrievalStageResult, ValidationIssue, ValidationReport
from app.embeddings.model import EmbeddingModelError
from app.extraction.schema import BidExtraction
from app.extraction.service import LLMProviderError, UnknownBidError
from app.reconciliation.schema import BidReconciliation, Timeline


def doc(status="completed") -> DocumentInfo:
    return DocumentInfo(document_id=1, file_name="a.pdf", document_type="rfp", status=status)


class Recorder:
    def __init__(self) -> None:
        self.order: list[str] = []


class StubIngestion:
    def __init__(self, rec, result=None, error=None, exists=True):
        self.rec, self.result, self.error, self.exists = rec, result, error, exists

    async def bid_exists(self, bid_id):
        return self.exists

    async def run(self, bid_id, ingest=True):
        self.rec.order.append("ingestion")
        if self.error:
            raise self.error
        return self.result or IngestionResult(bid_id=bid_id, files_found=1, skipped=1, documents=[doc()])


class StubRetrieval:
    def __init__(self, rec, chunks=5, error=None):
        self.rec, self.chunks, self.error = rec, chunks, error

    async def gather(self, bid_id):
        self.rec.order.append("retrieval")
        if self.error:
            raise self.error
        return {"due_date": []}, RetrievalStageResult(bid_id=bid_id, fields=1, total_chunks=self.chunks)


class StubExtraction:
    def __init__(self, rec, available=True, result=None, error=None):
        self.rec, self.available, self.result, self.error = rec, available, result, error
        self.received = None

    async def run(self, bid_id, evidence=None):
        self.rec.order.append("extraction")
        self.received = evidence
        if self.error:
            raise self.error
        return self.result or BidExtraction(bid_id=bid_id, fields={})


class StubReconciliation:
    def __init__(self, rec, result=None, warnings=None, error=None):
        self.rec, self.result, self.warnings, self.error = rec, result, warnings or [], error
        self.received = "unset"

    async def run(self, bid_id, extraction):
        self.rec.order.append("reconciliation")
        self.received = extraction
        if self.error:
            raise self.error
        return self.result or BidReconciliation(bid_id=bid_id, timeline=Timeline(bid_id=bid_id), fields=[]), self.warnings


class StubValidation:
    def __init__(self, rec, report=None):
        self.rec, self.report = rec, report or ValidationReport(valid=True)

    async def validate(self, bid_id, extraction, reconciliation):
        self.rec.order.append("validation")
        return self.report


def make(rec=None, **overrides) -> tuple[Orchestrator, Recorder, dict]:
    rec = rec or Recorder()
    parts = {
        "ingestion": StubIngestion(rec), "retrieval": StubRetrieval(rec), "extraction": StubExtraction(rec),
        "reconciliation": StubReconciliation(rec), "validation": StubValidation(rec),
    }  # fmt: skip
    parts.update(overrides)
    return Orchestrator(**parts), rec, parts


async def test_successful_pipeline_runs_stages_in_order() -> None:
    orchestrator, rec, parts = make()
    state = await orchestrator.run("Bid1")

    assert rec.order == ["ingestion", "retrieval", "extraction", "reconciliation", "validation"]
    assert state.status == "complete" and state.complete and state.errors == [] and state.warnings == []
    assert {n: s.state for n, s in state.stages.items()} == dict.fromkeys(rec.order, "complete")
    assert state.report is not None and state.validation.valid
    # stage outputs are handed on: retrieval evidence -> extraction, extraction -> reconciliation
    assert parts["extraction"].received == {"due_date": []}
    assert parts["reconciliation"].received is state.extraction


async def test_response_exposes_stage_states_and_report() -> None:
    orchestrator, _, _ = make()
    response = Orchestrator.response(await orchestrator.run("Bid1"))
    assert response.status == "complete" and list(response.stages) == [
        "ingestion", "retrieval", "extraction", "reconciliation", "validation"
    ]  # fmt: skip
    assert response.result is not None and response.errors == []


async def test_missing_groq_gives_partial_not_failure() -> None:
    orchestrator, rec, parts = make(extraction=StubExtraction(Recorder(), available=False))
    state = await orchestrator.run("Bid1")

    assert state.status == "partial" and not state.complete and not state.fatal
    assert state.stages["extraction"].state == "skipped"
    assert any("Groq is not configured" in w for w in state.warnings)
    assert "extraction" not in rec.order and "reconciliation" in rec.order and state.report is not None
    assert parts["reconciliation"].received is None


async def test_groq_failure_during_extraction_is_recoverable() -> None:
    orchestrator, rec, _ = make(extraction=StubExtraction(Recorder(), error=LLMProviderError("Groq request failed (APIConnectionError)")))
    state = await orchestrator.run("Bid1")

    assert state.status == "partial" and state.stages["extraction"].state == "failed" and not state.fatal
    assert any("APIConnectionError" in e for e in state.errors)
    assert "reconciliation" in rec.order and "validation" in rec.order


async def test_partial_extraction_errors_and_unresolved_reconciliation_are_reported() -> None:
    extraction = BidExtraction(bid_id="Bid1", fields={}, errors=["group 'schedule': model output invalid after 2 attempts"], complete=False)
    recon = BidReconciliation(bid_id="Bid1", timeline=Timeline(bid_id="Bid1"), fields=[], reconciliation_required=["Due Date"])
    orchestrator, _, _ = make(extraction=StubExtraction(Recorder(), result=extraction), reconciliation=StubReconciliation(Recorder(), result=recon))
    state = await orchestrator.run("Bid1")

    assert state.status == "partial"
    assert state.stages["extraction"].state == "partial" and state.stages["reconciliation"].state == "partial"
    assert any("Unresolved after reconciliation: Due Date" in w for w in state.warnings)
    assert any("model output invalid" in e for e in state.errors)


async def test_ingestion_without_usable_documents_is_fatal_and_stops() -> None:
    failed = IngestionResult(bid_id="Bid1", files_found=1, failed_files=["a.pdf"], documents=[doc("failed")])
    orchestrator, rec, _ = make(ingestion=StubIngestion(Recorder(), result=failed))
    state = await orchestrator.run("Bid1")

    assert state.status == "failed" and state.fatal
    assert state.stages["ingestion"].state == "failed"
    assert [s.state for n, s in state.stages.items() if n != "ingestion"] == ["skipped"] * 4  # nothing after a fatal stage
    assert Orchestrator.response(state).result is None


async def test_some_failed_documents_are_a_warning_not_fatal() -> None:
    mixed = IngestionResult(bid_id="Bid1", files_found=2, failed_files=["bad.pdf"], documents=[doc(), doc("failed")])
    orchestrator, _, _ = make(ingestion=StubIngestion(Recorder(), result=mixed))
    state = await orchestrator.run("Bid1")
    assert state.status == "partial" and any("bad.pdf" in w for w in state.warnings)


@pytest.mark.parametrize(
    ("stage", "error", "text"),
    [
        ("ingestion", OperationalError("SELECT 1", {}, Exception("password=hunter2")), "database error"),
        ("retrieval", EmbeddingModelError("/secret/model/path"), "embedding model unavailable"),
    ],
)
async def test_database_and_embedding_failures_are_fatal_without_leaking_details(stage, error, text) -> None:
    rec = Recorder()
    stub = {"ingestion": StubIngestion(rec, error=error)} if stage == "ingestion" else {"retrieval": StubRetrieval(rec, error=error)}
    state = await make(rec, **stub)[0].run("Bid1")

    assert state.status == "failed" and state.fatal
    assert text in state.errors[0] and "hunter2" not in str(state.errors) and "/secret" not in str(state.errors)
    assert "extraction" not in rec.order


async def test_no_retrieved_evidence_is_fatal() -> None:
    state = await make(retrieval=StubRetrieval(Recorder(), chunks=0))[0].run("Bid1")
    assert state.status == "failed" and "no evidence" in state.errors[0]


async def test_unexpected_stage_exception_never_looks_like_success() -> None:
    state = await make(reconciliation=StubReconciliation(Recorder(), error=RuntimeError("boom with details")))[0].run("Bid1")
    assert state.status == "failed" and "unexpected RuntimeError" in state.errors[0] and "details" not in state.errors[0]


async def test_validation_errors_fail_the_pipeline_and_withhold_the_report() -> None:
    bad = ValidationReport(valid=False, issues=[ValidationIssue(severity="error", stage="reconciliation", field="Due Date", code="CITATION_NOT_FOUND", message="x")])
    state = await make(validation=StubValidation(Recorder(), bad))[0].run("Bid1")

    response = Orchestrator.response(state)
    assert state.status == "failed" and state.report is None
    assert response.result is None and response.validation.errors[0].code == "CITATION_NOT_FOUND"
    assert any("CITATION_NOT_FOUND" in e for e in state.errors)


async def test_validation_warnings_make_the_run_partial_but_keep_the_report() -> None:
    warn = ValidationReport(valid=True, issues=[ValidationIssue(severity="warning", stage="reconciliation", field="Due Date", code="UNRESOLVED_CONFLICT", message="two dates")])
    state = await make(validation=StubValidation(Recorder(), warn))[0].run("Bid1")
    assert state.status == "partial" and state.report is not None


async def test_unknown_bid_raises_before_any_work() -> None:
    rec = Recorder()
    orchestrator, _, _ = make(rec, ingestion=StubIngestion(rec, exists=False))
    with pytest.raises(UnknownBidError):
        await orchestrator.run("Nope")
    assert rec.order == []


async def test_running_twice_gives_the_same_outcome() -> None:
    orchestrator, rec, _ = make()
    first, second = await orchestrator.run("Bid1"), await orchestrator.run("Bid1")
    assert first.model_dump(exclude={"stages", "run_id", "timings_ms"}) == second.model_dump(exclude={"stages", "run_id", "timings_ms"})
    assert rec.order == ["ingestion", "retrieval", "extraction", "reconciliation", "validation"] * 2
