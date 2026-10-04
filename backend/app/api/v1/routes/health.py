from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.schemas.health import DatabaseHealth, GroqHealthResponse, HealthResponse, SystemStatusResponse
from app.services import health as health_service

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health(
    response: Response, session: Annotated[AsyncSession, Depends(get_db)]
) -> HealthResponse:
    """Liveness plus a real database / pgvector check. Returns 503 if unhealthy."""
    result = await health_service.check_database(session)
    if result.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get("/groq", response_model=GroqHealthResponse)
async def groq_health(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    probe: bool = False,
) -> GroqHealthResponse:
    """Check Groq configuration. Add `?probe=true` to make a minimal live request."""
    result = await health_service.check_groq(settings, probe=probe)
    if result.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get("/status", response_model=SystemStatusResponse)
async def system_status(
    response: Response,
    session: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    probe_embeddings: bool = False,
) -> SystemStatusResponse:
    """Health per component. Only the database and embedding model decide the overall status.

    A missing Groq key degrades only the LLM features (extraction, reconciliation fallback, Q&A);
    ingestion, embeddings and search keep working, so it does not make the system unhealthy.
    `probe_embeddings=true` actually loads the model (slow the first time).
    """
    db = await health_service.check_database(session)
    database = DatabaseHealth(status="ok" if db.database else "unavailable", pgvector=db.pgvector)
    embeddings = await health_service.check_embeddings(settings, probe=probe_embeddings)
    groq_health = await health_service.check_groq(settings, probe=False)

    core_ok = db.database and db.pgvector and embeddings.status == "ok"
    if not core_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return SystemStatusResponse(
        status="ok" if core_ok else "degraded",
        database=database,
        embeddings=embeddings,
        groq=groq_health,
        features={
            "ingestion": db.database and embeddings.status == "ok",
            "search": core_ok,
            "llm_features": core_ok and groq_health.status == "ok",
        },
    )
