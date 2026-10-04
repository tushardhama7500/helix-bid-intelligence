import asyncio
import os
import sys
from pathlib import Path

# Must be set before `app` modules read settings. If backend/.env exists it
# provides the real DATABASE_URL, so don't shadow it with a placeholder.
if not (Path(__file__).resolve().parents[1] / ".env").exists():
    os.environ.setdefault(
        "DATABASE_URL", "postgresql+psycopg://helix:helix@localhost:5432/helix"
    )

# psycopg's async mode does not work with the Windows default (Proactor) loop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

TEST_DB = "helix_test"


@pytest.fixture
async def pg_engine():
    """Fresh schema in a dedicated `helix_test` PostgreSQL database.

    Uses the server from DATABASE_URL but never touches the real database.
    Skipped when PostgreSQL / pgvector is unavailable.
    """
    import app.models  # noqa: F401
    from app.core.config import get_settings
    from app.core.database import Base, init_db

    url = make_url(get_settings().async_database_url)
    if not url.drivername.startswith("postgresql"):
        pytest.skip("DATABASE_URL is not PostgreSQL")

    admin = create_async_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": TEST_DB}
            )
            if not exists:
                await conn.execute(text(f"CREATE DATABASE {TEST_DB}"))
    except Exception:
        await admin.dispose()
        pytest.skip("PostgreSQL is not reachable")
    await admin.dispose()

    engine = create_async_engine(url.set(database=TEST_DB))
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.drop_all)
        await init_db(engine)
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"Cannot prepare pgvector test database: {exc}")
    yield engine
    await engine.dispose()


@pytest.fixture
async def pg_session(pg_engine):
    async with async_sessionmaker(pg_engine, expire_on_commit=False)() as session:
        yield session
