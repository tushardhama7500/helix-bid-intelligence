"""Hybrid retrieval: query -> (vector search + keyword search) -> RRF -> top-K evidence."""

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.embeddings.service import EmbeddingService
from app.ingestion.metadata import DOCUMENT_TYPES
from app.retrieval.filters import SearchFilters
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.keyword_search import keyword_search
from app.retrieval.query import preprocess_query
from app.retrieval.schema import (
    KEYWORD,
    VECTOR,
    Candidate,
    FusedResult,
    InvalidFilterError,
    InvalidQueryError,
)
from app.retrieval.vector_search import vector_search

logger = logging.getLogger(__name__)

HYBRID = "hybrid"
MODES = (HYBRID, VECTOR, KEYWORD)


@dataclass
class RetrievalResult:
    query: str
    top_k: int
    results: list[FusedResult]


class RetrievalService:
    def __init__(self, session: AsyncSession, embedder: EmbeddingService, settings: Settings) -> None:
        self.session = session
        self.embedder = embedder
        self.settings = settings

    @classmethod
    def from_settings(cls, session: AsyncSession, settings: Settings) -> "RetrievalService":
        return cls(session, EmbeddingService.from_settings(session, settings), settings)

    async def available_filters(self) -> dict[str, list[str]]:
        async def column(sql: str) -> list[str]:
            return [row[0] for row in await self.session.execute(text(sql))]

        return {
            "bids": await column("SELECT DISTINCT bid_id FROM documents ORDER BY 1"),
            "document_types": await column("SELECT DISTINCT document_type FROM documents ORDER BY 1"),
            "file_names": await column("SELECT DISTINCT file_name FROM documents ORDER BY 1"),
        }

    async def retrieve(
        self,
        query: str,
        bid_id: str | None = None,
        document_type: str | None = None,
        file_name: str | None = None,
        document_id: int | None = None,
        page_number: int | None = None,
        top_k: int | None = None,
        mode: str = HYBRID,
    ) -> RetrievalResult:
        """Return the top-K evidence chunks for `query`, each with its citation metadata.

        `mode` is `hybrid` (default), or `vector` / `keyword` to run one method
        alone (used to evaluate them separately). Raises `RetrievalError`
        subclasses for bad input.
        """
        settings = self.settings
        processed = preprocess_query(query, settings.max_query_length)
        top_k = settings.default_top_k if top_k is None else top_k
        if not 1 <= top_k <= settings.max_top_k:
            raise InvalidQueryError(f"top_k must be between 1 and {settings.max_top_k}")
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}")
        filters = SearchFilters(bid_id, document_type, file_name, document_id, page_number)
        await self._validate_filters(filters)

        rankings: dict[str, list[Candidate]] = {}
        if mode in (HYBRID, VECTOR):
            query_vector = await self.embedder.embed_query(processed.text)
            rankings[VECTOR] = await vector_search(
                self.session, query_vector, self.embedder.model.name, filters,
                settings.vector_search_limit,
            )  # fmt: skip
        if mode in (HYBRID, KEYWORD):
            rankings[KEYWORD] = await keyword_search(
                self.session, processed.terms, filters, settings.keyword_search_limit,
                settings.keyword_min_term_coverage,
            )  # fmt: skip

        fused = reciprocal_rank_fusion(rankings, settings.rrf_k)[:top_k]
        logger.info(
            "Search mode=%s filters=%s -> %s candidates, returning %d",
            mode, filters.active, {m: len(c) for m, c in rankings.items()}, len(fused),
        )  # fmt: skip
        return RetrievalResult(query=processed.text, top_k=top_k, results=fused)

    async def _validate_filters(self, filters: SearchFilters) -> None:
        if filters.document_type is not None and filters.document_type not in DOCUMENT_TYPES:
            raise InvalidFilterError(
                f"Unknown document_type {filters.document_type!r}. Valid: {', '.join(DOCUMENT_TYPES)}"
            )
        if filters.bid_id is not None:
            bids = (await self.available_filters())["bids"]
            if filters.bid_id not in bids:
                raise InvalidFilterError(
                    f"Unknown bid_id {filters.bid_id!r}. Available: {', '.join(bids) or 'none'}"
                )
