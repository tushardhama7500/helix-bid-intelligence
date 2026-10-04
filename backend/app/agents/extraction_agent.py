"""Extraction and reconciliation agents: thin wrappers over the Phase 5 and Phase 6 services."""

import logging

from app.agents.tracing import agent_span
from app.extraction.schema import BidExtraction
from app.extraction.service import ExtractionService, LLMProviderError
from app.reconciliation.schema import BidReconciliation
from app.reconciliation.service import ReconciliationService
from app.retrieval.schema import Candidate

logger = logging.getLogger(__name__)


class ExtractionAgent:
    """Requests structured extraction. Prompts, Groq calls, grounding, citation checks and
    retries all stay in `ExtractionService`. Without Groq there is nothing to run."""

    def __init__(self, service: ExtractionService | None) -> None:
        self.service = service

    @property
    def available(self) -> bool:
        return self.service is not None

    async def run(self, bid_id: str, evidence: dict[str, list[Candidate]] | None = None) -> BidExtraction:
        if self.service is None:
            raise RuntimeError("Extraction requires Groq (GROQ_API_KEY and GROQ_MODEL)")
        async with agent_span("extraction", bid_id, "extract") as span:
            result = await self.service.extract_bid(bid_id, evidence)
            span["note"] = f"conflicts={len(result.reconciliation_required)} errors={len(result.errors)}"
            return result


class ReconciliationAgent:
    """Runs addendum reconciliation. The deterministic rules, timeline and Groq fallback stay in
    `ReconciliationService`; this only adds graceful degradation if Groq becomes unavailable."""

    def __init__(self, service: ReconciliationService) -> None:
        self.service = service

    async def run(self, bid_id: str, extraction: BidExtraction | None) -> tuple[BidReconciliation, list[str]]:
        """Returns (reconciliation, warnings). If the provider fails, retries deterministically."""
        warnings: list[str] = []
        async with agent_span("reconciliation", bid_id, "reconcile") as span:
            try:
                result = await self.service.reconcile_bid(bid_id, extraction)
            except LLMProviderError as exc:
                if self.service.llm is None:
                    raise
                logger.warning("Groq unavailable during reconciliation (%s); using deterministic rules only", exc)
                warnings.append(f"Groq was unavailable ({exc}); addenda were reconciled with deterministic rules only.")
                deterministic = ReconciliationService(
                    self.service.retrieval, self.service.settings, None, self.service._timeline_loader, self.service._sleep
                )  # fmt: skip
                result = await deterministic.reconcile_bid(bid_id, extraction)
            span["note"] = f"unresolved={len(result.reconciliation_required)} errors={len(result.errors)}"
            return result, warnings
