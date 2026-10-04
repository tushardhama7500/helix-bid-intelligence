"""Database access for embeddings. No model logic lives here."""

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentChunk


@dataclass
class ChunkCounts:
    total: int
    embedded: int
    pending: int  # no embedding yet, no recorded error
    failed: int  # no embedding, last attempt failed


async def get_counts(session: AsyncSession) -> ChunkCounts:
    no_vector = DocumentChunk.embedding.is_(None)
    row = (
        await session.execute(
            select(
                func.count(),
                func.count().filter(DocumentChunk.embedding.is_not(None)),
                func.count().filter(no_vector, DocumentChunk.embedding_error.is_(None)),
                func.count().filter(no_vector, DocumentChunk.embedding_error.is_not(None)),
            )
        )
    ).one()
    return ChunkCounts(*row)


async def next_batch(
    session: AsyncSession, after_id: int, limit: int
) -> list[tuple[int, str]]:
    """Next chunks without an embedding, in id order (keyset pagination)."""
    result = await session.execute(
        select(DocumentChunk.id, DocumentChunk.content)
        .where(DocumentChunk.embedding.is_(None), DocumentChunk.id > after_id)
        .order_by(DocumentChunk.id)
        .limit(limit)
    )
    return [(row.id, row.content) for row in result]


async def store_embeddings(
    session: AsyncSession, model_name: str, rows: Sequence[tuple[int, list[float]]]
) -> None:
    if not rows:
        return
    await session.execute(
        update(DocumentChunk),
        [
            {"id": chunk_id, "embedding": vector, "embedding_model": model_name, "embedding_error": None}
            for chunk_id, vector in rows
        ],
    )


async def store_failures(session: AsyncSession, rows: Sequence[tuple[int, str]]) -> None:
    if not rows:
        return
    await session.execute(
        update(DocumentChunk),
        [{"id": chunk_id, "embedding_error": message} for chunk_id, message in rows],
    )
