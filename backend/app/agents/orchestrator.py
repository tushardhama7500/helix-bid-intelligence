"""Orchestrator: a fixed, linear workflow over the agents. No loops, no agent-to-agent calls.

    ingestion -> retrieval -> extraction -> reconciliation -> validation -> report

The orchestrator never parses, searches, extracts, reconciles or calls Groq itself; it only
sequences the agents, classifies failures and combines their outcomes into one status.

Failure policy
  fatal        (stop; status "failed"): the bid is unknown or has no ingestible documents, the
               database or embedding model is unavailable, or validation finds invalid output
               (the report is then withheld)
  recoverable  (continue; status "partial"): some files failed to ingest, Groq is not configured
               or unavailable (extraction is skipped; reconciliation still runs deterministically),
               some fields could not be extracted, or addendum wording could not be resolved
"""

import logging
import time
import uuid

from sqlalchemy.exc import SQLAlchemyError

from app.agents.extraction_agent import ExtractionAgent, ReconciliationAgent
from app.agents.ingestion_agent import IngestionAgent
from app.agents.report_qa_agent import ReportQAAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.schema import STAGES, BidPipelineState, PipelineResponse, StageResult
from app.agents.validation_agent import ValidationAgent
from app.core.context import run_id_var
from app.embeddings.model import EmbeddingModelError
from app.extraction.service import LLMProviderError, UnknownBidError

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        *,
        ingestion: IngestionAgent,
        retrieval: RetrievalAgent,
        extraction: ExtractionAgent,
        reconciliation: ReconciliationAgent,
        validation: ValidationAgent,
    ) -> None:
        self.ingestion = ingestion
        self.retrieval = retrieval
        self.extraction = extraction
        self.reconciliation = reconciliation
        self.validation = validation

    async def run(self, bid_id: str, ingest: bool = True) -> BidPipelineState:
        """Run the whole workflow. Idempotent: unchanged files and embedded chunks are skipped,
        and the analysis stages are recomputed from the same stored data."""
        run_id = uuid.uuid4().hex[:8]
        token = run_id_var.set(run_id)
        try:
            return await self._run(bid_id, ingest, run_id)
        finally:
            run_id_var.reset(token)

    async def _run(self, bid_id: str, ingest: bool, run_id: str) -> BidPipelineState:
        state = BidPipelineState(bid_id=bid_id, run_id=run_id, stages={n: StageResult(name=n) for n in STAGES})
        logger.info("pipeline start run=%s bid=%s ingest=%s", run_id, bid_id, ingest)
        if not await self.ingestion.bid_exists(bid_id):
            raise UnknownBidError(f"Unknown bid_id {bid_id!r}: no source files and no stored documents.")

        run_started = time.perf_counter()
        evidence = None
        for name in STAGES:
            if state.fatal:
                break
            stage = state.stages[name]
            started = time.perf_counter()
            try:
                evidence = await getattr(self, f"_{name}")(state, stage, ingest, evidence)
            except (SQLAlchemyError, EmbeddingModelError) as exc:
                stage.state = "failed"
                self._fatal(state, stage, f"{name}: {_describe(exc)}", "infrastructure")
            except Exception as exc:  # an agent bug or an unexpected failure must not look like success
                logger.exception("Unexpected failure in stage %s", name)
                stage.state = "failed"
                self._fatal(state, stage, f"{name}: unexpected {type(exc).__name__}", "internal")
            stage.duration_ms = int((time.perf_counter() - started) * 1000)

        state.timings_ms = {n: s.duration_ms for n, s in state.stages.items()}
        state.timings_ms["total"] = int((time.perf_counter() - run_started) * 1000)
        state.errors += [e for s in state.stages.values() for e in s.errors if e not in state.errors]
        state.warnings += [w for s in state.stages.values() for w in s.warnings if w not in state.warnings]
        if state.fatal:
            state.status = "failed"
        elif any(s.state != "complete" for s in state.stages.values()) or state.warnings:
            state.status = "partial"
        else:
            state.status = "complete"
        state.complete = state.status == "complete"
        logger.info(
            "pipeline end run=%s bid=%s status=%s failure=%s total_ms=%d errors=%d warnings=%d stages=%s",
            run_id, bid_id, state.status, state.failure, state.timings_ms.get("total", 0), len(state.errors),
            len(state.warnings), {n: s.state for n, s in state.stages.items()},
        )  # fmt: skip
        return state

    # --------------------------------------------------------------- stages
    async def _ingestion(self, state, stage, ingest, evidence):
        result = state.ingestion = await self.ingestion.run(state.bid_id, ingest)
        stage.detail = {
            "files_found": result.files_found, "ingested": result.ingested, "skipped": result.skipped,
            "documents": len(result.documents), "chunks_embedded": result.chunks_embedded,
        }  # fmt: skip
        if result.failed_files:
            stage.warnings.append(f"Documents failed to ingest: {', '.join(result.failed_files)}")
        if result.embedding_failures:
            stage.warnings.append(f"{result.embedding_failures} chunk(s) could not be embedded and are not searchable.")
        if not result.completed_documents:
            stage.state = "failed"
            return self._fatal(state, stage, "ingestion: no document of this bid could be ingested", "no_documents")
        stage.state = "partial" if stage.warnings else "complete"
        return evidence

    async def _retrieval(self, state, stage, ingest, evidence):
        evidence, result = await self.retrieval.gather(state.bid_id)
        state.retrieval = result
        stage.detail = {"fields": result.fields, "chunks": result.total_chunks, "empty_fields": len(result.empty_fields)}
        if result.total_chunks == 0:
            stage.state = "failed"
            return self._fatal(state, stage, "retrieval: no evidence could be retrieved (are the chunks embedded?)", "no_evidence")
        if result.dropped_foreign_chunks:
            stage.warnings.append(f"{result.dropped_foreign_chunks} chunk(s) from another bid were discarded.")
        stage.state = "partial" if stage.warnings else "complete"
        return evidence

    async def _extraction(self, state, stage, ingest, evidence):
        if not self.extraction.available:
            stage.state = "skipped"
            stage.warnings.append("Groq is not configured: structured extraction was skipped; only deterministic reconciliation ran.")
            return evidence
        llm = getattr(getattr(self.extraction, "service", None), "llm", None)
        mark = len(getattr(llm, "calls", []))
        try:
            extraction = state.extraction = await self.extraction.run(state.bid_id, evidence)
        except LLMProviderError as exc:
            self._record_usage(stage, llm, mark)
            stage.state = "failed"
            stage.errors.append(f"extraction: {exc}")
            stage.warnings.append("Extraction was unavailable (Groq error); results rely on deterministic reconciliation only.")
            return evidence
        stage.detail = {
            "conflicts": len(extraction.reconciliation_required), "model": extraction.model,
            "schema_retries": extraction.schema_retries, "citation_retries": extraction.citation_retries,
        }  # fmt: skip
        self._record_usage(stage, llm, mark)
        stage.errors += [f"extraction: {e}" for e in extraction.errors]
        stage.state = "partial" if extraction.errors else "complete"
        return evidence

    async def _reconciliation(self, state, stage, ingest, evidence):
        llm = getattr(getattr(self.reconciliation, "service", None), "llm", None)
        mark = len(getattr(llm, "calls", []))
        result, warnings = await self.reconciliation.run(state.bid_id, state.extraction)
        state.reconciliation = result
        stage.warnings += warnings
        stage.detail = {
            "addenda": len(result.timeline.addenda), "ordering": result.timeline.ordering,
            "resolved": sum(f.status == "resolved" for f in result.fields), "used_llm": result.used_llm,
        }  # fmt: skip
        self._record_usage(stage, llm, mark)
        stage.errors += [f"reconciliation: {e}" for e in result.errors]
        if result.reconciliation_required:
            stage.warnings.append("Unresolved after reconciliation: " + ", ".join(result.reconciliation_required))
        stage.state = "partial" if (result.errors or result.reconciliation_required or warnings) else "complete"
        return evidence

    async def _validation(self, state, stage, ingest, evidence):
        report = state.validation = await self.validation.validate(state.bid_id, state.extraction, state.reconciliation)
        stage.detail = {"valid": report.valid, "errors": len(report.errors), "warnings": len(report.warnings), **report.checked}
        stage.warnings += [f"{i.field or 'pipeline'}: {i.message}" for i in report.warnings]
        if not report.valid:
            stage.state = "failed"
            stage.errors += [f"validation: {i.field or 'pipeline'}: {i.code}" for i in report.errors]
            return self._fatal(state, stage, "validation found invalid output; the report was withheld", "validation")
        state.report = ReportQAAgent.build_report(state.bid_id, state.extraction, state.reconciliation)
        stage.state = "partial" if stage.warnings else "complete"
        return evidence

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _record_usage(stage: StageResult, llm, mark: int) -> None:
        """Groq calls/tokens/latency for this stage (only a real client keeps records)."""
        if hasattr(llm, "usage_since"):
            stage.detail["groq"] = llm.usage_since(mark)

    @staticmethod
    def _fatal(state: BidPipelineState, stage: StageResult, message: str, kind: str) -> None:
        state.fatal = True
        state.failure = state.failure or kind  # type: ignore[assignment]
        if message not in stage.errors:
            stage.errors.append(message)
        return None

    @staticmethod
    def response(state: BidPipelineState) -> PipelineResponse:
        """The API view: stage outcomes, messages and the validated report (null if invalid)."""
        return PipelineResponse(
            bid_id=state.bid_id,
            run_id=state.run_id,
            status=state.status,
            failure=state.failure,
            stages={n: s.state for n, s in state.stages.items()},
            errors=state.errors,
            warnings=state.warnings,
            validation=state.validation,
            result=state.report if state.validation and state.validation.valid else None,
            timings_ms=state.timings_ms,
        )


def _describe(exc: Exception) -> str:
    if isinstance(exc, EmbeddingModelError):
        return "embedding model unavailable"
    return "database error"
