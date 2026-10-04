"""Idempotent PostgreSQL DDL for the embedding column and its vector index."""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

logger = logging.getLogger(__name__)

INDEX_NAME = "ix_document_chunks_embedding_hnsw"


async def ensure_embedding_schema(
    conn: AsyncConnection, model_name: str, dimension: int
) -> None:
    """Make `document_chunks.embedding` a `vector(dimension)` with a cosine HNSW index.

    - Adds the columns to databases created by Phase 2.
    - If the dimension changed, or vectors came from a different model, the old
      vectors are cleared so they get regenerated; they are never mixed.
    """
    dimension = int(dimension)  # interpolated into DDL below
    await conn.execute(
        text(f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector({dimension})")
    )
    await conn.execute(
        text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(200)")
    )
    await conn.execute(
        text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_error TEXT")
    )

    # For pgvector, atttypmod is the dimension (-1 when unconstrained).
    typmod = (
        await conn.execute(
            text(
                "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'document_chunks'::regclass "
                "AND attname = 'embedding' AND NOT attisdropped"
            )
        )
    ).scalar_one()

    if typmod != dimension:
        if typmod > 0:
            logger.warning(
                "Embedding dimension changed %d -> %d; clearing existing embeddings", typmod, dimension
            )
        await conn.execute(text(f"DROP INDEX IF EXISTS {INDEX_NAME}"))
        await conn.execute(
            text("UPDATE document_chunks SET embedding = NULL, embedding_model = NULL, embedding_error = NULL")
        )
        await conn.execute(
            text(f"ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector({dimension}) USING NULL")
        )
    else:
        cleared = await conn.execute(
            text(
                "UPDATE document_chunks SET embedding = NULL, embedding_model = NULL "
                "WHERE embedding IS NOT NULL AND embedding_model IS DISTINCT FROM :model"
            ),
            {"model": model_name},
        )
        if cleared.rowcount:
            logger.warning("Cleared %d embeddings produced by a different model", cleared.rowcount)

    # HNSW needs no training data (unlike IVFFlat, which wants rows to build good
    # lists) and stays valid as chunks are added. Cosine ops match how we search.
    await conn.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} "
            "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
        )
    )
