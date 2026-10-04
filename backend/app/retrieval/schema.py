"""Internal retrieval types and the PostgreSQL full-text-search DDL."""

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

logger = logging.getLogger(__name__)

FTS_CONFIG = "english"
SEARCH_INDEX_NAME = "ix_document_chunks_search_vector"

VECTOR = "vector"
KEYWORD = "keyword"


class RetrievalError(Exception):
    """Base class for errors caused by the caller's request (HTTP 422)."""


class InvalidQueryError(RetrievalError):
    pass


class InvalidFilterError(RetrievalError):
    pass


@dataclass
class Candidate:
    """One chunk returned by one search method, with everything needed to cite it."""

    chunk_id: int
    content: str
    document_id: int
    bid_id: str
    file_name: str
    document_type: str
    page_number: int | None  # None for HTML: page numbers are never invented
    chunk_index: int
    metadata: dict[str, Any]
    score: float  # method-specific: cosine distance (vector) or ts_rank_cd (keyword)
    rank: int = 0  # 1-based position in that method's result list


@dataclass
class FusedResult:
    candidate: Candidate  # content + citation metadata
    rrf_score: float  # ranking score, NOT a probability or confidence
    methods: list[str] = field(default_factory=list)
    ranks: dict[str, int] = field(default_factory=dict)  # method -> 1-based rank
    scores: dict[str, float] = field(default_factory=dict)  # method -> raw score


async def ensure_search_schema(conn: AsyncConnection) -> None:
    """Add a generated `tsvector` column and a GIN index (idempotent).

    A STORED generated column keeps the vector in sync with `content` without
    triggers or application code, so re-ingested chunks are searchable at once.
    GIN is the standard index for tsvector: it maps each lexeme to the rows that
    contain it, which is exactly the lookup `@@` needs. (GiST is smaller but
    slower to query and lossy; there is no reason to prefer it here.)
    """
    await conn.execute(
        text(
            "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS search_vector tsvector "
            f"GENERATED ALWAYS AS (to_tsvector('{FTS_CONFIG}', content)) STORED"
        )
    )
    await conn.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS {SEARCH_INDEX_NAME} "
            "ON document_chunks USING gin (search_vector)"
        )
    )
