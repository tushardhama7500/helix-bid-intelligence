import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.params import BidId
from app.core.config import ConfigurationError, Settings, get_settings
from app.core.database import get_db
from app.embeddings.model import EmbeddingModelError
from app.extraction.schema import BidExtraction
from app.extraction.service import ExtractionService, LLMProviderError, UnknownBidError
from app.llm.groq_client import GroqClient
from app.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extraction", tags=["extraction"])


def get_extraction_service(
    session: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ExtractionService:
    try:
        llm = GroqClient.from_settings(settings)
    except ConfigurationError as exc:
        # Say which setting is missing, but never anything secret.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return ExtractionService(RetrievalService.from_settings(session, settings), llm, settings)


ServiceDep = Annotated[ExtractionService, Depends(get_extraction_service)]


@router.post("/{bid_id}", response_model=BidExtraction)
async def extract_bid(bid_id: BidId, service: ServiceDep) -> BidExtraction:
    """Extract the structured bid fields for one bid from its retrieved evidence.

    Stateless: every call retrieves, calls Groq and validates afresh. Conflicting
    values (e.g. original vs addendum due date) are reported, not resolved.
    """
    try:
        return await service.extract_bid(bid_id)
    except UnknownBidError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except EmbeddingModelError as exc:
        logger.error("Embedding model unavailable: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Embedding model unavailable") from exc
    except SQLAlchemyError as exc:
        logger.exception("Extraction failed on the database")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database error") from exc
