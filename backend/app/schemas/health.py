from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    pgvector: bool


class GroqHealthResponse(BaseModel):
    status: Literal["ok", "unconfigured", "error"]
    configured: bool
    model: str | None = None
    # None when no live request was made (default); see `?probe=true`.
    reachable: bool | None = None
    detail: str | None = None


class EmbeddingsHealth(BaseModel):
    status: Literal["ok", "unavailable"]
    model: str
    dimension: int | None = None
    loaded: bool = False  # the model loads on first use unless `?probe=true` was requested
    detail: str | None = None


class DatabaseHealth(BaseModel):
    status: Literal["ok", "unavailable"]
    pgvector: bool


class SystemStatusResponse(BaseModel):
    """Per-component health. Groq is optional: without it ingestion and search still work."""

    status: Literal["ok", "degraded"]
    application: Literal["ok"] = "ok"
    database: DatabaseHealth
    embeddings: EmbeddingsHealth
    groq: GroqHealthResponse
    features: dict[str, bool]  # what works right now
