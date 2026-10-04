import httpx
import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.v1.routes.search import get_retrieval_service
from app.core.config import Settings
from app.core.database import get_db
from app.embeddings.model import EmbeddingModelError
from app.embeddings.service import EmbeddingService
from app.main import app
from app.retrieval.service import RetrievalService
from tests.fakes import TopicEmbeddingModel, seed_corpus

SETTINGS = Settings(_env_file=None, database_url="postgresql+psycopg://x/x", max_top_k=10)


@pytest.fixture
async def api(pg_engine):
    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    model = TopicEmbeddingModel()
    async with maker() as session:
        await seed_corpus(session, model)

    async def override_service():
        async with maker() as session:
            yield RetrievalService(session, EmbeddingService(session, model, 8), SETTINGS)

    app.dependency_overrides[get_retrieval_service] = override_service
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        client.model = model
        yield client
    app.dependency_overrides.clear()


async def test_valid_search_returns_evidence_with_citations(api) -> None:
    res = await api.post("/api/v1/search", json={"query": "What is the deadline for JA-207652?", "bid_id": "Bid1", "top_k": 2})

    assert res.status_code == 200
    body = res.json()
    assert body["query"] == "What is the deadline for JA-207652?" and body["top_k"] == 2
    assert len(body["results"]) == 2
    first = body["results"][0]
    assert first["source"] == {
        "document_id": first["source"]["document_id"], "bid_id": "Bid1",
        "file_name": "JA-207652 FINAL.pdf", "document_type": "rfp", "page_number": 2,
        "chunk_index": 1, "metadata": first["source"]["metadata"],
    }  # fmt: skip
    assert first["retrieval_methods"] == ["vector", "keyword"]
    # Scores are separate and labelled; `score` is the RRF ranking score.
    assert first["score"] == first["scores"]["rrf"]
    assert first["scores"]["vector"]["cosine_similarity"] == pytest.approx(1 - first["scores"]["vector"]["cosine_distance"])
    assert first["scores"]["keyword"]["ts_rank"] > 0
    assert "answer" not in body and "content" in first


async def test_html_result_has_null_page_and_section(api) -> None:
    res = await api.post("/api/v1/search", json={"query": "BPM044557"})
    top = res.json()["results"][0]
    assert top["source"]["page_number"] is None
    assert top["source"]["metadata"]["section"] == "Bid Information"


async def test_filters_restrict_results(api) -> None:
    res = await api.post("/api/v1/search", json={"query": "deadline", "bid_id": "Bid2", "document_type": "rfp", "top_k": 10})
    assert [r["source"]["file_name"] for r in res.json()["results"]] == ["PORFP_Dell_Laptop.pdf"]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 501},
        {"query": "deadline", "top_k": 0},
        {"query": "deadline", "top_k": 11},
        {"query": "deadline", "top_k": "many"},
        {"query": "deadline", "bid_id": "Bid9"},
        {"query": "deadline", "document_type": "memo"},
        {"query": "deadline", "page_number": 0},
        {},
    ],
)
async def test_invalid_requests_are_422(api, payload) -> None:
    res = await api.post("/api/v1/search", json=payload)
    assert res.status_code == 422
    assert "Traceback" not in res.text


async def test_filters_endpoint(api) -> None:
    body = (await api.get("/api/v1/search/filters")).json()
    assert body["bids"] == ["Bid1", "Bid2"] and "specification" in body["document_types"]


async def test_embedding_model_unavailable_is_503_without_internals(api) -> None:
    api.model.load_error = None

    def boom(_: str):
        raise EmbeddingModelError("secret /path/to/model failed")

    api.model.embed_query = boom
    res = await api.post("/api/v1/search", json={"query": "deadline"})
    assert res.status_code == 503
    assert res.json() == {"detail": "Embedding model unavailable"}


async def test_database_failure_is_503_without_stack_trace(api) -> None:
    class BrokenService:
        async def retrieve(self, **_):
            raise OperationalError("SELECT secret", {}, Exception("connection refused host=10.0.0.5"))

    app.dependency_overrides[get_retrieval_service] = lambda: BrokenService()
    res = await api.post("/api/v1/search", json={"query": "deadline"})
    assert res.status_code == 503
    assert res.json() == {"detail": "Database error"}
    assert "10.0.0.5" not in res.text and "SELECT" not in res.text
