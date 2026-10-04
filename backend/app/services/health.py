"""Health-check logic, kept out of the route handlers."""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import ConfigurationError, Settings
from app.llm.groq_client import GroqClient
from app.schemas.health import EmbeddingsHealth, GroqHealthResponse, HealthResponse

logger = logging.getLogger(__name__)


async def check_database(session: AsyncSession) -> HealthResponse:
    """Verify the DB answers queries and that pgvector is installed."""
    try:
        await session.execute(text("SELECT 1"))
        result = await session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        )
        pgvector = result.scalar_one_or_none() is not None
    except Exception:
        logger.exception("Database health check failed")
        return HealthResponse(status="degraded", database=False, pgvector=False)

    return HealthResponse(
        status="ok" if pgvector else "degraded", database=True, pgvector=pgvector
    )


async def check_groq(settings: Settings, probe: bool = False) -> GroqHealthResponse:
    """Check Groq configuration; optionally make a 1-token live request."""
    try:
        client = GroqClient.from_settings(settings)
    except ConfigurationError as exc:
        return GroqHealthResponse(
            status="unconfigured",
            configured=False,
            model=settings.groq_model,
            detail=str(exc),
        )

    if not probe:
        return GroqHealthResponse(status="ok", configured=True, model=client.model)

    try:
        await client.generate([{"role": "user", "content": "ping"}], max_tokens=1)
    except Exception as exc:
        logger.warning("Groq probe failed: %s", type(exc).__name__)
        return GroqHealthResponse(
            status="error",
            configured=True,
            model=client.model,
            reachable=False,
            detail=type(exc).__name__,
        )
    return GroqHealthResponse(
        status="ok", configured=True, model=client.model, reachable=True
    )


async def check_embeddings(settings: Settings, probe: bool = False) -> EmbeddingsHealth:
    """Is the embedding model usable? Without `probe` this only checks the configured model name."""
    import asyncio

    from app.embeddings.model import EmbeddingModelError, declared_dimension, get_embedding_model

    try:
        dimension = declared_dimension(settings.embedding_model)
        model = get_embedding_model(settings.embedding_model, settings.embedding_batch_size)
        if probe:
            await asyncio.to_thread(model.load)
        return EmbeddingsHealth(
            status="ok", model=settings.embedding_model, dimension=dimension, loaded=model._model is not None
        )
    except EmbeddingModelError as exc:
        logger.warning("Embedding health check failed: %s", type(exc).__name__)
        # The message can name a path or URL; report only what the operator needs.
        return EmbeddingsHealth(
            status="unavailable", model=settings.embedding_model, detail="embedding model cannot be used"
        )
