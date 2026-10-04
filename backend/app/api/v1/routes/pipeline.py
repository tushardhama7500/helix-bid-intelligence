import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.factory import build_orchestrator, build_qa_agent
from app.agents.orchestrator import Orchestrator
from app.agents.report_qa_agent import QAOutputError, ReportQAAgent
from app.agents.schema import PipelineResponse, QAAnswer, QARequest
from app.api.v1.params import BidId
from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.embeddings.model import EmbeddingModelError
from app.extraction.service import LLMProviderError, UnknownBidError

logger = logging.getLogger(__name__)

pipeline_router = APIRouter(prefix="/pipeline", tags=["pipeline"])
qa_router = APIRouter(prefix="/qa", tags=["qa"])


def get_orchestrator(
    session: Annotated[AsyncSession, Depends(get_db)], settings: Annotated[Settings, Depends(get_settings)]
) -> Orchestrator:
    return build_orchestrator(session, settings)


def get_qa(
    session: Annotated[AsyncSession, Depends(get_db)], settings: Annotated[Settings, Depends(get_settings)]
) -> ReportQAAgent:
    return build_qa_agent(session, settings)


# Failed pipelines keep their structured body but use the status codes the rest of the API uses.
FAILURE_HTTP_STATUS = {
    "infrastructure": status.HTTP_503_SERVICE_UNAVAILABLE,  # database / embedding model, as in /search and /qa
    "no_documents": 422,
    "no_evidence": 422,
    "validation": status.HTTP_500_INTERNAL_SERVER_ERROR,  # our own output failed its checks
    "internal": status.HTTP_500_INTERNAL_SERVER_ERROR,
}


@pipeline_router.post(
    "/{bid_id}",
    response_model=PipelineResponse,
    responses={r: {"model": PipelineResponse, "description": "Pipeline failed; body explains why"} for r in (422, 500, 503)},
)
async def run_pipeline(
    bid_id: BidId, orchestrator: Annotated[Orchestrator, Depends(get_orchestrator)], ingest: bool = True
) -> PipelineResponse:
    """Run ingestion -> retrieval -> extraction -> reconciliation -> validation for one bid.

    `status` is `complete`, `partial` (recoverable problems: see `errors`/`warnings`; HTTP 200) or `failed`
    (HTTP 503 infrastructure, 422 nothing to analyse, 500 invalid output; the body still explains why).
    Without Groq the pipeline still runs and reports `partial`. When validation finds errors the
    report is withheld (`result: null`).
    """
    try:
        state = await orchestrator.run(bid_id, ingest)
        response = Orchestrator.response(state)
        if state.status == "failed":
            code = FAILURE_HTTP_STATUS.get(state.failure or "internal", 500)
            return JSONResponse(status_code=code, content=response.model_dump(mode="json"))
        return response
    except UnknownBidError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.exception("Pipeline failed on the database")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database error") from exc
    except EmbeddingModelError as exc:
        logger.error("Embedding model unavailable: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Embedding model unavailable") from exc


@qa_router.post("/{bid_id}", response_model=QAAnswer)
async def ask(bid_id: BidId, request: QARequest, agent: Annotated[ReportQAAgent, Depends(get_qa)]) -> QAAnswer:
    """Answer a question from this bid's retrieved evidence, with citations, or refuse."""
    if agent.llm is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Q&A requires Groq: set GROQ_API_KEY and GROQ_MODEL.")
    try:
        bids = (await agent.retrieval.retrieval.available_filters())["bids"]
        if bid_id not in bids:
            raise UnknownBidError(f"Unknown bid_id {bid_id!r}. Available: {', '.join(bids) or 'none'}")
        return await agent.answer(bid_id, request.question)
    except UnknownBidError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except QAOutputError as exc:
        logger.error("Q&A model output invalid: %s", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The model returned an invalid answer; please try again.") from exc
    except SQLAlchemyError as exc:
        logger.exception("Q&A failed on the database")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database error") from exc
    except EmbeddingModelError as exc:
        logger.error("Embedding model unavailable: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Embedding model unavailable") from exc
