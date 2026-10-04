import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401
from app.core.config import Settings, get_settings
from app.core.database import Base, get_db
from app.main import app
from tests.helpers import lorem, make_pdf


def _settings(data_dir: str) -> Settings:
    return Settings(_env_file=None, database_url="postgresql+psycopg://x/x", data_dir=data_dir)


@pytest.fixture
async def client(tmp_path):
    root = tmp_path / "data"
    make_pdf(root / "Bid1" / "Affidavit.pdf", [lorem(10)])
    (root / "Bid2").mkdir(parents=True)
    (root / "Bid2" / "page.html").write_text("<h1>Hi</h1><p>Body text.</p>", encoding="utf-8")

    engine = create_async_engine(
        "sqlite+aiosqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async def override_db():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_settings] = lambda: _settings(str(root))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        yield c
    app.dependency_overrides.clear()
    await engine.dispose()


async def test_scan(client) -> None:
    body = (await client.post("/api/v1/ingestion/scan")).json()
    assert body == {
        "files_found": 2,
        "pdf_files": 1,
        "html_files": 1,
        "ignored_files": 0,
        "bids": {"Bid1": 1, "Bid2": 1},
    }


async def test_run_list_and_detail(client) -> None:
    run = (await client.post("/api/v1/ingestion/run")).json()
    assert (run["total"], run["processed"], run["skipped"], run["failed"]) == (2, 2, 0, 0)

    again = (await client.post("/api/v1/ingestion/run")).json()
    assert (again["processed"], again["skipped"]) == (0, 2)

    docs = (await client.get("/api/v1/ingestion/documents")).json()
    assert [(d["bid_id"], d["document_type"], d["status"]) for d in docs] == [
        ("Bid1", "affidavit", "completed"),
        ("Bid2", "bid_document", "completed"),
    ]

    detail = (await client.get(f"/api/v1/ingestion/documents/{docs[0]['id']}")).json()
    assert detail["stored_pages"] == 1 and detail["chunk_count"] >= 1
    assert detail["pages"][0]["page_number"] == 1
    assert (await client.get("/api/v1/ingestion/documents/9999")).status_code == 404


async def test_missing_data_dir_is_404(client) -> None:
    app.dependency_overrides[get_settings] = lambda: _settings("/no/such/dir")
    assert (await client.post("/api/v1/ingestion/scan")).status_code == 404
