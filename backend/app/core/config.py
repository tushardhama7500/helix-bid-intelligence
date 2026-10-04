"""Application settings, loaded from environment variables (and an optional .env)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# backend/app/core/config.py -> repository root (or "/" inside the Docker image).
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str

    # SecretStr: printing or logging the settings object shows '**********', never the key.
    groq_api_key: SecretStr | None = None
    groq_model: str | None = None

    # Ingestion: relative paths are resolved against the repository root.
    data_dir: str = "data"
    # Files larger than this are skipped (and reported) rather than parsed; the real PDFs are <= 9 MB.
    max_ingest_file_mb: int = Field(default=50, gt=0)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # Local FastEmbed model; the pgvector column dimension is derived from it.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = Field(default=32, gt=0)

    # Retrieval (Phase 4)
    vector_search_limit: int = Field(default=20, gt=0)
    keyword_search_limit: int = Field(default=20, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    default_top_k: int = Field(default=5, gt=0)
    max_top_k: int = Field(default=20, gt=0)
    max_query_length: int = Field(default=500, gt=0)
    # A keyword hit must contain at least this share of the meaningful query terms.
    keyword_min_term_coverage: float = Field(default=0.6, gt=0, le=1)

    # Extraction (Phase 5): how much evidence goes to the LLM.
    extraction_evidence_per_field: int = Field(default=4, gt=0)
    extraction_addendum_evidence: int = Field(default=2, ge=0)
    extraction_max_chunks_per_call: int = Field(default=14, gt=0)
    extraction_chunk_char_limit: int = Field(default=1200, gt=0)
    extraction_max_output_tokens: int = Field(default=3000, gt=0)

    # Reconciliation (Phase 6): evidence examined per addendum and per Groq fallback call.
    reconciliation_chunks_per_addendum: int = Field(default=6, gt=0)
    reconciliation_llm_max_chunks: int = Field(default=8, gt=0)

    # Origin of the React app allowed by CORS.
    frontend_origin: str = "http://localhost:5173"

    @field_validator("groq_api_key", "groq_model", mode="before")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        """Treat `GROQ_MODEL=` (empty) the same as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.default_top_k > self.max_top_k:
            raise ValueError("DEFAULT_TOP_K must not exceed MAX_TOP_K")
        return self

    @property
    def data_path(self) -> Path:
        path = Path(self.data_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def async_database_url(self) -> str:
        """DATABASE_URL normalised to the async psycopg 3 driver."""
        url = self.database_url
        for prefix in ("postgresql://", "postgres://", "postgresql+psycopg2://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
