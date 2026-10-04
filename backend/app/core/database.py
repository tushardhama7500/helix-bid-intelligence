"""Async SQLAlchemy engine, session dependency and startup initialisation."""

import logging
from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(get_settings().async_database_url, pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a database session."""
    async with get_sessionmaker()() as session:
        yield session


async def init_db(engine: AsyncEngine | None = None) -> None:
    """Enable pgvector, create tables and bring the embedding column up to date.

    `create_all` never alters existing tables, so the embedding columns and the
    vector index are managed by `ensure_embedding_schema`. A migration tool can
    replace all of this later.
    """
    import app.models  # noqa: F401  (registers models on Base.metadata)
    from app.embeddings.model import declared_dimension
    from app.embeddings.schema import ensure_embedding_schema
    from app.retrieval.schema import ensure_search_schema

    settings = get_settings()
    dimension = declared_dimension(settings.embedding_model)
    async with (engine or get_engine()).begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await ensure_embedding_schema(conn, settings.embedding_model, dimension)
        await ensure_search_schema(conn)


async def dispose_engine() -> None:
    await get_engine().dispose()
