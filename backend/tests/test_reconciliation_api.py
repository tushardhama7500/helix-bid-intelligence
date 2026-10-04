import groq
import httpx
import pytest
from sqlalchemy.exc import OperationalError

from app.api.v1.routes.reconciliation import get_reconciliation_service
from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.embeddings.model import EmbeddingModelError
from app.main import app
from app.reconciliation.service import ReconciliationService
from tests.extraction_fakes import FakeLLM
from tests.test_reconciliation_service import ADD1, ADD2, BOND, RFP, SETTINGS, Retrieval, loader


def client_for(chunks, llm=None, error=None):
    retrieval = Retrieval(chunks, error)
    addenda = [c for c in chunks if c.document_type == "addendum"]
    service = ReconciliationService(retrieval, SETTINGS, llm, loader(addenda))
    app.dependency_overrides[get_reconciliation_service] = lambda: service
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t")


@pytest.fixture(autouse=True)
def _clear():
    yield
    app.dependency_overrides.clear()


async def test_bid1_reconciliation_response() -> None:
    async with client_for([RFP, ADD1, ADD2]) as client:
        res = await client.post("/api/v1/reconciliation/Bid1")

    assert res.status_code == 200
    body = res.json()
    due = next(f for f in body["fields"] if f["key"] == "due_date")
    assert due["original_value"] == "27-JUN-2024 14:00:00" and due["final_value"] == "July 9, 2024 at 2:00 PM CST"
    assert (due["status"], due["resolution_method"]) == ("resolved", "explicit_addendum_change")
    assert due["changes"][0]["addendum_number"] == 2 and due["changes"][0]["sources"][0]["page_number"] == 1
    assert body["complete"] is True and body["reconciliation_required"] == []
    assert [e["addendum_number"] for e in body["timeline"]["addenda"]] == [1, 2]
    assert "api_key" not in res.text.lower()


async def test_unknown_bid_is_404() -> None:
    async with client_for([RFP, ADD2]) as client:
        res = await client.post("/api/v1/reconciliation/Bid9")
    assert res.status_code == 404 and "Bid9" in res.json()["detail"]


async def test_database_and_embedding_failures_are_503() -> None:
    db_error = OperationalError("SELECT 1", {}, Exception("password authentication failed host=10.0.0.5"))
    async with client_for([RFP], error=db_error) as client:
        res = await client.post("/api/v1/reconciliation/Bid1")
    assert res.status_code == 503 and res.json() == {"detail": "Database error"} and "10.0.0.5" not in res.text

    async with client_for([RFP], error=EmbeddingModelError("/secret/path")) as client:
        res = await client.post("/api/v1/reconciliation/Bid1")
    assert res.status_code == 503 and "/secret/path" not in res.text


async def test_groq_failure_is_502() -> None:
    down = groq.APIConnectionError(request=httpx.Request("POST", "http://secret-host"))
    async with client_for([RFP, BOND], llm=FakeLLM(script=[down])) as client:
        res = await client.post("/api/v1/reconciliation/Bid1")
    assert res.status_code == 502 and "secret-host" not in res.text


async def test_invalid_groq_output_is_reported_in_the_body_not_hidden() -> None:
    bad = '{"changes": 5}'
    async with client_for([RFP, BOND], llm=FakeLLM(script=[bad, bad])) as client:
        res = await client.post("/api/v1/reconciliation/Bid1")
    body = res.json()
    assert res.status_code == 200 and body["complete"] is False and "invalid after 2 attempts" in body["errors"][0]
    bond = next(f for f in body["fields"] if f["key"] == "bid_bond")
    assert bond["final_value"] is None and bond["status"] == "insufficient_evidence"


async def test_missing_groq_is_not_fatal_unless_extraction_is_requested() -> None:
    async with client_for([RFP, ADD2, BOND], llm=None) as client:
        ok = await client.post("/api/v1/reconciliation/Bid1")
        needs_llm = await client.post("/api/v1/reconciliation/Bid1?use_extraction=true")

    assert ok.status_code == 200 and ok.json()["complete"] is False
    assert needs_llm.status_code == 503 and "GROQ_API_KEY" in needs_llm.json()["detail"]


async def test_real_dependency_without_groq_still_serves_deterministic_requests() -> None:
    async def no_db():
        yield None

    app.dependency_overrides[get_db] = no_db
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, database_url="postgresql+psycopg://x/x")
    service = get_reconciliation_service(None, Settings(_env_file=None, database_url="postgresql+psycopg://x/x", groq_api_key="k"))
    assert service.llm is None  # GROQ_MODEL missing -> no client, no crash, no key leakage
