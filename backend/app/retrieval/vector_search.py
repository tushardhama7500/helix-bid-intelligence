"""Semantic search: nearest chunks by cosine distance, executed by pgvector."""

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.filters import SearchFilters
from app.retrieval.schema import Candidate

logger = logging.getLogger(__name__)

_SQL = """
SELECT c.id AS chunk_id, c.content, c.chunk_index, c.bid_id, c.metadata,
       d.id AS document_id, d.file_name, d.document_type, p.page_number,
       c.embedding <=> CAST(:qvec AS vector) AS score
FROM document_chunks c
JOIN documents d ON d.id = c.document_id
JOIN document_pages p ON p.id = c.page_id
WHERE c.embedding IS NOT NULL AND c.embedding_model = :model{filters}
ORDER BY c.embedding <=> CAST(:qvec AS vector), c.id
LIMIT :limit
"""


def row_to_candidate(row: Any, rank: int) -> Candidate:
    return Candidate(
        chunk_id=row["chunk_id"],
        content=row["content"],
        document_id=row["document_id"],
        bid_id=row["bid_id"],
        file_name=row["file_name"],
        document_type=row["document_type"],
        page_number=row["page_number"],
        chunk_index=row["chunk_index"],
        metadata=row["metadata"] or {},
        score=float(row["score"]),
        rank=rank,
    )


async def vector_search(
    session: AsyncSession,
    query_vector: list[float],
    model_name: str,
    filters: SearchFilters,
    limit: int,
) -> list[Candidate]:
    """Top `limit` chunks by cosine distance (smaller = closer; similarity = 1 - distance).

    Only chunks embedded by `model_name` are compared, so query and document
    vectors always come from the same model.
    """
    clause, params = filters.to_sql()
    if filters.active:
        # With a WHERE clause the HNSW index can run out of candidates before
        # `limit` matching rows are found; iterative scan (pgvector >= 0.8) keeps
        # scanning. Best effort: older pgvector simply doesn't know the setting.
        try:
            async with session.begin_nested():
                await session.execute(
                    text("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
                )
        except Exception:
            logger.debug("hnsw.iterative_scan not available; continuing without it")

    result = await session.execute(
        text(_SQL.format(filters=clause)),
        {"qvec": "[" + ",".join(map(repr, query_vector)) + "]", "model": model_name, "limit": limit, **params},
    )
    return [row_to_candidate(row, rank) for rank, row in enumerate(result.mappings(), start=1)]
