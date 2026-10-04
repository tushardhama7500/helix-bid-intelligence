"""Gather evidence per field by calling the existing hybrid retrieval service."""

import logging

from app.core.config import Settings
from app.extraction.fields import FIELDS, FieldSpec
from app.retrieval.schema import Candidate, FusedResult
from app.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

SUMMARY_CHUNKS_PER_FIELD = 1
SUMMARY_MAX_CHUNKS = 10


def _merge(*result_lists: list[FusedResult]) -> list[Candidate]:
    """Merge result lists into candidates ordered by RRF score; each chunk once."""
    best: dict[int, FusedResult] = {}
    for results in result_lists:
        for result in results:
            current = best.get(result.candidate.chunk_id)
            if current is None or result.rrf_score > current.rrf_score:
                best[result.candidate.chunk_id] = result
    return [r.candidate for r in sorted(best.values(), key=lambda r: (-r.rrf_score, r.candidate.chunk_id))]


async def gather_field_evidence(
    retrieval: RetrievalService, bid_id: str, spec: FieldSpec, settings: Settings
) -> list[Candidate]:
    """Evidence for one field: its own queries, plus (for amendable fields) addendum text.

    The addendum-only query just makes sure amendments are *visible* to the model
    even when the original document outranks them. It does not decide anything.
    """
    per_field = spec.evidence_limit or settings.extraction_evidence_per_field
    results = [
        (await retrieval.retrieve(query, bid_id=bid_id, top_k=per_field)).results for query in spec.queries
    ]
    primary = _merge(*results)[:per_field]

    extras: list[Candidate] = []
    if spec.addenda and settings.extraction_addendum_evidence:
        addendum = await retrieval.retrieve(
            spec.queries[0], bid_id=bid_id, document_type="addendum", top_k=settings.extraction_addendum_evidence
        )
        seen = {c.chunk_id for c in primary}
        extras = [r.candidate for r in addendum.results if r.candidate.chunk_id not in seen]
    return primary + extras


async def gather_evidence(
    retrieval: RetrievalService, bid_id: str, settings: Settings
) -> dict[str, list[Candidate]]:
    """Field key -> ranked evidence chunks (deduplicated within each field)."""
    evidence: dict[str, list[Candidate]] = {}
    for spec in FIELDS:
        if spec.kind == "text":
            continue
        evidence[spec.key] = await gather_field_evidence(retrieval, bid_id, spec, settings)
        logger.info("Evidence for bid=%s field=%s: %d chunks", bid_id, spec.key, len(evidence[spec.key]))

    # The summary draws on the best evidence of every other field plus its own query.
    summary = next(f for f in FIELDS if f.kind == "text")
    own = await retrieval.retrieve(summary.queries[0], bid_id=bid_id, top_k=3)
    pool: dict[int, Candidate] = {}
    for chunks in evidence.values():
        for chunk in chunks[:SUMMARY_CHUNKS_PER_FIELD]:
            pool.setdefault(chunk.chunk_id, chunk)
    for result in own.results:
        pool.setdefault(result.candidate.chunk_id, result.candidate)
    evidence[summary.key] = list(pool.values())[:SUMMARY_MAX_CHUNKS]
    logger.info("Evidence for bid=%s field=%s: %d chunks", bid_id, summary.key, len(evidence[summary.key]))
    return evidence


def select_group_evidence(
    field_evidence: dict[str, list[Candidate]], max_chunks: int, extra: dict[str, int] | None = None
) -> tuple[list[Candidate], dict[str, dict[int, Candidate]]]:
    """Pick the chunks to send in one LLM call.

    Chunks are taken round-robin by rank across the group's fields (every field
    gets its best chunk first), deduplicated, until `max_chunks`. Returns the chunk
    list plus, per field, the subset of its own evidence that was actually sent;
    that subset is what the field's citations are validated against.
    """
    chosen: dict[int, Candidate] = {}
    depth = max((len(v) for v in field_evidence.values()), default=0)
    for rank in range(depth):
        for chunks in field_evidence.values():
            if rank < len(chunks) and len(chosen) < max_chunks:
                chosen.setdefault(chunks[rank].chunk_id, chunks[rank])
    # Fields that need deeper evidence (e.g. one spec per product line) get extra chunks of their own, so the
    # other fields' top-ranked chunks cannot crowd them out. The group's budget grows only by these amounts.
    for key, n in (extra or {}).items():
        for chunk in field_evidence.get(key, []):
            if n <= 0:
                break
            if chunk.chunk_id not in chosen:
                chosen[chunk.chunk_id] = chunk
                n -= 1
    allowed = {
        key: {c.chunk_id: c for c in chunks if c.chunk_id in chosen} for key, chunks in field_evidence.items()
    }
    return list(chosen.values()), allowed
