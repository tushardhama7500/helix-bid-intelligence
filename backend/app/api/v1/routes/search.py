import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.embeddings.model import EmbeddingModelError
from app.retrieval.schema import FusedResult, RetrievalError
from app.retrieval.service import RetrievalService
from app.schemas.search import (
    FiltersResponse,
    KeywordScore,
    Scores,
    SearchRequest,
    SearchResponse,
    SearchResult,
    Source,
    VectorScore,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


def get_retrieval_service(
    session: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RetrievalService:
    return RetrievalService.from_settings(session, settings)


ServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]


def to_result(fused: FusedResult) -> SearchResult:
    c = fused.candidate
    vector = keyword = None
    if "vector" in fused.ranks:
        distance = fused.scores["vector"]
        vector = VectorScore(rank=fused.ranks["vector"], cosine_distance=distance, cosine_similarity=1 - distance)
    if "keyword" in fused.ranks:
        keyword = KeywordScore(rank=fused.ranks["keyword"], ts_rank=fused.scores["keyword"])
    return SearchResult(
        chunk_id=c.chunk_id,
        content=c.content,
        score=fused.rrf_score,
        retrieval_methods=fused.methods,
        scores=Scores(rrf=fused.rrf_score, vector=vector, keyword=keyword),
        source=Source(
            document_id=c.document_id,
            bid_id=c.bid_id,
            file_name=c.file_name,
            document_type=c.document_type,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            metadata=c.metadata,
        ),
    )


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest, service: ServiceDep) -> SearchResponse:
    """Hybrid retrieval (pgvector + PostgreSQL full-text, fused with RRF).

    Returns retrieved evidence only; no answer is generated.
    """
    try:
        found = await service.retrieve(**request.model_dump())
    except RetrievalError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except EmbeddingModelError as exc:
        logger.error("Embedding model unavailable: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Embedding model unavailable") from exc
    except SQLAlchemyError as exc:
        logger.exception("Search failed on the database")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database error") from exc
    return SearchResponse(query=found.query, top_k=found.top_k, results=[to_result(r) for r in found.results])


@router.get("/filters", response_model=FiltersResponse)
async def filters(service: ServiceDep) -> FiltersResponse:
    """Values available for the metadata filters."""
    try:
        return FiltersResponse(**await service.available_filters())
    except SQLAlchemyError as exc:
        logger.exception("Could not load filters")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database error") from exc
