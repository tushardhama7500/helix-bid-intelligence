import logging
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.embeddings.model import EmbeddingModelError
from app.embeddings.service import EmbeddingService
from app.schemas.embeddings import GenerateResponse, StatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


def get_embedding_service(
    session: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingService:
    return EmbeddingService.from_settings(session, settings)


ServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]


@router.post("/generate", response_model=GenerateResponse)
async def generate(service: ServiceDep) -> GenerateResponse:
    """Embed every chunk that has no embedding yet. Safe to call repeatedly."""
    try:
        return GenerateResponse(**asdict(await service.generate()))
    except EmbeddingModelError as exc:
        logger.error("Embedding model unavailable: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Embedding model unavailable") from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database error") from exc


@router.get("/status", response_model=StatusResponse)
async def embedding_status(
    service: ServiceDep, settings: Annotated[Settings, Depends(get_settings)]
) -> StatusResponse:
    try:
        counts = await service.status()
    except SQLAlchemyError as exc:
        logger.exception("Could not read embedding status")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database error") from exc
    return StatusResponse(
        total_chunks=counts.total,
        embedded_chunks=counts.embedded,
        pending_chunks=counts.pending,
        failed_chunks=counts.failed,
        embedding_model=settings.embedding_model,
        embedding_dimension=service.model.dimension,
    )
