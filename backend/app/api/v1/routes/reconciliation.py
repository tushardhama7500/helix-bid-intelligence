import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.params import BidId
from app.core.config import ConfigurationError, Settings, get_settings
from app.core.database import get_db
from app.embeddings.model import EmbeddingModelError
from app.extraction.service import ExtractionService, LLMProviderError, UnknownBidError
from app.llm.groq_client import GroqClient
from app.reconciliation.schema import BidReconciliation
from app.reconciliation.service import ReconciliationService
from app.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reconciliation", tags=["reconciliation"])


def get_reconciliation_service(
    session: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReconciliationService:
    try:
        llm = GroqClient.from_settings(settings)
    except ConfigurationError:
        llm = None  # deterministic rules still work; Groq is only needed for ambiguous wording
    return ReconciliationService(RetrievalService.from_settings(session, settings), settings, llm)


ServiceDep = Annotated[ReconciliationService, Depends(get_reconciliation_service)]


@router.post("/{bid_id}", response_model=BidReconciliation)
async def reconcile_bid(bid_id: BidId, service: ServiceDep, use_extraction: bool = False) -> BidReconciliation:
    """Resolve fields that addenda change, keeping the original value and every change in the history.

    Explicit change language is resolved with deterministic rules. Groq is used only for
    ambiguous wording, and only if configured; without it those fields are reported
    unresolved. `use_extraction=true` additionally runs Phase 5 extraction to find original
    values for fields without a built-in locator (requires Groq).
    """
    try:
        extraction = None
        if use_extraction:
            if service.llm is None:
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    "use_extraction requires Groq: set GROQ_API_KEY and GROQ_MODEL.",
                )
            extraction = await ExtractionService(service.retrieval, service.llm, service.settings).extract_bid(bid_id)
        return await service.reconcile_bid(bid_id, extraction)
    except UnknownBidError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except EmbeddingModelError as exc:
        logger.error("Embedding model unavailable: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Embedding model unavailable") from exc
    except SQLAlchemyError as exc:
        logger.exception("Reconciliation failed on the database")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database error") from exc
