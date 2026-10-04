"""Wires the agents onto the existing services for one database session."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.extraction_agent import ExtractionAgent, ReconciliationAgent
from app.agents.ingestion_agent import IngestionAgent
from app.agents.orchestrator import Orchestrator
from app.agents.report_qa_agent import ReportQAAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.validation_agent import ValidationAgent, db_chunk_lookup
from app.core.config import ConfigurationError, Settings
from app.embeddings.service import EmbeddingService
from app.extraction.service import ExtractionService
from app.ingestion.service import IngestionService
from app.llm.groq_client import GroqClient
from app.reconciliation.service import ReconciliationService
from app.retrieval.service import RetrievalService


def optional_groq(settings: Settings) -> GroqClient | None:
    """The shared Groq client, or None when it is not configured."""
    try:
        return GroqClient.from_settings(settings)
    except ConfigurationError:
        return None


def build_orchestrator(session: AsyncSession, settings: Settings, llm: GroqClient | None = None) -> Orchestrator:
    llm = llm if llm is not None else optional_groq(settings)
    retrieval = RetrievalService.from_settings(session, settings)
    lookup = db_chunk_lookup(session)
    return Orchestrator(
        ingestion=IngestionAgent(IngestionService.from_settings(session, settings), EmbeddingService.from_settings(session, settings)),
        retrieval=RetrievalAgent(retrieval, settings),
        extraction=ExtractionAgent(ExtractionService(retrieval, llm, settings) if llm else None),
        reconciliation=ReconciliationAgent(ReconciliationService(retrieval, settings, llm)),
        validation=ValidationAgent(lookup),
    )


def build_qa_agent(session: AsyncSession, settings: Settings, llm: GroqClient | None = None) -> ReportQAAgent:
    llm = llm if llm is not None else optional_groq(settings)
    retrieval = RetrievalService.from_settings(session, settings)
    return ReportQAAgent(
        RetrievalAgent(retrieval, settings), db_chunk_lookup(session), settings, llm,
        reconciliation=ReconciliationService(retrieval, settings, llm),
    )  # fmt: skip
