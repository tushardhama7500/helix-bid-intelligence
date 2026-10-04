"""Integration test against a real PostgreSQL. Skipped if none is reachable."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings


@pytest.fixture
async def engine():
    eng = create_async_engine(get_settings().async_database_url)
    try:
        async with eng.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        await eng.dispose()
        pytest.skip("PostgreSQL is not reachable")
    yield eng
    await eng.dispose()


async def test_database_connectivity(engine) -> None:
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT 1"))).scalar_one() == 1


async def test_pgvector_extension_is_available(engine) -> None:
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
        )
        assert result.scalar_one_or_none() == 1
