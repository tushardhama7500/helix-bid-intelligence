import groq
import httpx
import pytest
from sqlalchemy.exc import OperationalError

from app.api.v1.routes.extraction import get_extraction_service
from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.embeddings.model import EmbeddingModelError
from app.extraction.service import ExtractionService
from app.main import app
from tests.extraction_fakes import FakeLLM, StubRetrieval

SETTINGS = Settings(_env_file=None, database_url="postgresql+psycopg://x/x")
CONFLICT = {
    "due_date": ([{"value": "27-JUN-2024 14:00:00", "chunk_ids": [1]},
                  {"value": "July 9, 2024 at 2:00 PM CST", "chunk_ids": [2]}], 0.9),
    "bid_number": ([{"value": "JA-207652", "chunk_ids": [1]}], 0.9),
}  # fmt: skip


def make_client(llm=None, retrieval=None):
    service = ExtractionService(retrieval or StubRetrieval(), llm or FakeLLM(answers=CONFLICT), SETTINGS)
    app.dependency_overrides[get_extraction_service] = lambda: service
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t")


@pytest.fixture(autouse=True)
def _clear():
    yield
    app.dependency_overrides.clear()


async def test_bid1_extraction_response() -> None:
    async with make_client() as client:
        res = await client.post("/api/v1/extraction/Bid1")

    assert res.status_code == 200
    body = res.json()
    assert body["bid_id"] == "Bid1" and len(body["fields"]) == 20
    due = body["fields"]["Due Date"]
    assert due["status"] == "conflict" and due["value"] is None and due["confidence"] == 0.0
    assert {c["value"] for c in due["conflicting_values"]} == {"27-JUN-2024 14:00:00", "July 9, 2024 at 2:00 PM CST"}
    assert "reconciliation is required" in due["notes"]
    number = body["fields"]["Bid Number"]
    assert number["value"] == ["JA-207652"] and number["sources"][0]["chunk_id"] == 1
    assert body["reconciliation_required"] == ["Due Date"] and body["complete"] is True
    assert "prompt" not in res.text.lower() and "api_key" not in res.text.lower()


async def test_bid2_extraction_response() -> None:
    answers = {"bid_number": ([{"value": "E20P4600040", "chunk_ids": [11]}], 0.9)}
    async with make_client(llm=FakeLLM(answers=answers)) as client:
        body = (await client.post("/api/v1/extraction/Bid2")).json()
    assert body["fields"]["Bid Number"]["value"] == ["E20P4600040"]
    assert body["reconciliation_required"] == []


async def test_unknown_bid_is_404() -> None:
    async with make_client() as client:
        res = await client.post("/api/v1/extraction/Bid9")
    assert res.status_code == 404 and "Bid9" in res.json()["detail"]


async def test_groq_failure_is_502_without_internals() -> None:
    down = groq.APIConnectionError(request=httpx.Request("POST", "http://secret-host"))
    async with make_client(llm=FakeLLM(script=[down])) as client:
        res = await client.post("/api/v1/extraction/Bid1")
    assert res.status_code == 502
    assert "secret-host" not in res.text and "Traceback" not in res.text


async def test_retrieval_database_failure_is_503() -> None:
    error = OperationalError("SELECT 1", {}, Exception("password authentication failed for user root"))
    async with make_client(retrieval=StubRetrieval(error=error)) as client:
        res = await client.post("/api/v1/extraction/Bid1")
    assert res.status_code == 503 and res.json() == {"detail": "Database error"}


async def test_embedding_model_failure_is_503() -> None:
    async with make_client(retrieval=StubRetrieval(error=EmbeddingModelError("/secret/path"))) as client:
        res = await client.post("/api/v1/extraction/Bid1")
    assert res.status_code == 503 and "/secret/path" not in res.text


async def test_missing_groq_configuration_is_503_and_names_only_the_setting() -> None:
    async def no_db():
        yield None

    app.dependency_overrides[get_db] = no_db
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, database_url="postgresql+psycopg://x/x", groq_api_key="super-secret-key"
    )
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        res = await client.post("/api/v1/extraction/Bid1")
    assert res.status_code == 503
    assert "GROQ_MODEL" in res.json()["detail"] and "super-secret-key" not in res.text
