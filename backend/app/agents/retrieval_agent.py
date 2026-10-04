"""Retrieval agent: builds field-specific retrieval requests on top of the Phase 4/5 retrieval."""

import logging

from app.agents.schema import RetrievalStageResult
from app.agents.tracing import agent_span
from app.core.config import Settings
from app.extraction.evidence import gather_evidence, gather_field_evidence
from app.extraction.fields import FieldSpec
from app.retrieval.schema import Candidate, FusedResult
from app.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """The hybrid search stays in `RetrievalService`; this adds the bid-isolation guarantee."""

    def __init__(self, retrieval: RetrievalService, settings: Settings) -> None:
        self.retrieval = retrieval
        self.settings = settings

    async def retrieve(
        self, bid_id: str, query: str, *, top_k: int = 5, document_type: str | None = None, document_id: int | None = None
    ) -> list[FusedResult]:
        """Free-text retrieval scoped to one bid; any chunk from another bid is discarded."""
        found = await self.retrieval.retrieve(
            query, bid_id=bid_id, document_type=document_type, document_id=document_id, top_k=top_k
        )
        return [r for r in found.results if r.candidate.bid_id == bid_id]

    async def retrieve_field(self, bid_id: str, spec: FieldSpec) -> list[Candidate]:
        """Evidence for one extraction field, using that field's own retrieval queries."""
        chunks = await gather_field_evidence(self.retrieval, bid_id, spec, self.settings)
        return [c for c in chunks if c.bid_id == bid_id]

    async def gather(self, bid_id: str) -> tuple[dict[str, list[Candidate]], RetrievalStageResult]:
        """Evidence for all fields (reused by extraction so nothing is retrieved twice)."""
        async with agent_span("retrieval", bid_id, "gather") as span:
            evidence = await gather_evidence(self.retrieval, bid_id, self.settings)
            dropped = 0
            for key, chunks in evidence.items():
                kept = [c for c in chunks if c.bid_id == bid_id]
                dropped += len(chunks) - len(kept)
                evidence[key] = kept
            ids = {c.chunk_id for chunks in evidence.values() for c in chunks}
            stage = RetrievalStageResult(
                bid_id=bid_id,
                fields=len(evidence),
                total_chunks=len(ids),
                evidence_counts={k: len(v) for k, v in evidence.items()},
                empty_fields=[k for k, v in evidence.items() if not v],
                dropped_foreign_chunks=dropped,
            )  # fmt: skip
            span["note"] = f"fields={stage.fields} chunks={stage.total_chunks}"
            return evidence, stage
