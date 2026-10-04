import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.v1.routes.embeddings import get_embedding_service
from app.core.database import get_db
from app.embeddings.service import EmbeddingService
from app.main import app
from tests.fakes import FakeEmbeddingModel, seed_chunks


@pytest.fixture
async def api(pg_engine):
    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    model = FakeEmbeddingModel()

    async def override_db():
        async with maker() as session:
            yield session

    async def override_service():
        async with maker() as session:
            yield EmbeddingService(session, model, 4)

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_embedding_service] = override_service
    async with maker() as session:
        await seed_chunks(session, [f"chunk {i}" for i in range(5)])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        client.model = model
        yield client
    app.dependency_overrides.clear()


async def test_status_before_and_after_generate(api) -> None:
    before = (await api.get("/api/v1/embeddings/status")).json()
    assert before == {
        "total_chunks": 5, "embedded_chunks": 0, "pending_chunks": 5, "failed_chunks": 0,
        "embedding_model": before["embedding_model"], "embedding_dimension": 384,
    }  # fmt: skip

    run = (await api.post("/api/v1/embeddings/generate")).json()
    assert (run["total_chunks"], run["embedded"], run["skipped"], run["failed"]) == (5, 5, 0, 0)

    again = (await api.post("/api/v1/embeddings/generate")).json()
    assert (again["embedded"], again["skipped"]) == (0, 5)

    after = (await api.get("/api/v1/embeddings/status")).json()
    assert (after["embedded_chunks"], after["pending_chunks"], after["failed_chunks"]) == (5, 0, 0)


async def test_generate_reports_model_unavailable_as_503(api) -> None:
    api.model.load_error = "cannot download"
    res = await api.post("/api/v1/embeddings/generate")
    assert res.status_code == 503
    assert "cannot download" not in res.text  # internals are not leaked
