import json

import groq
import httpx
import pytest
from sqlalchemy.exc import OperationalError

from app.agents.schema import REFUSAL, ValidationIssue, ValidationReport
from app.api.v1.routes.pipeline import get_orchestrator, get_qa
from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.main import app
from tests.extraction_fakes import FakeLLM
from tests.test_agents_orchestrator import Recorder, StubExtraction, StubIngestion, StubValidation, make
from tests.test_agents_qa import DEADLINE, build, say
from tests.test_reconciliation_service import ADD1, ADD2, RFP


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t")


@pytest.fixture(autouse=True)
def _clear():
    yield
    app.dependency_overrides.clear()


def use_pipeline(**overrides):
    orchestrator, _, _ = make(**overrides)
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator


def use_qa(llm, chunks=(RFP, ADD1, ADD2), retrieval_cls=None):
    agent, _ = build(list(chunks), llm) if retrieval_cls is None else build(list(chunks), llm, retrieval_cls)
    app.dependency_overrides[get_qa] = lambda: agent


# -------------------------------------------------------------------------- pipeline
async def test_pipeline_success_response() -> None:
    use_pipeline()
    async with client() as c:
        res = await c.post("/api/v1/pipeline/Bid1")

    body = res.json()
    assert res.status_code == 200 and body["status"] == "complete" and body["errors"] == [] and body["warnings"] == []
    assert body["stages"] == dict.fromkeys(["ingestion", "retrieval", "extraction", "reconciliation", "validation"], "complete")
    assert body["result"]["bid_id"] == "Bid1" and len(body["result"]["fields"]) == 20
    assert body["validation"]["valid"] is True


async def test_pipeline_without_groq_is_partial_not_an_error() -> None:
    use_pipeline(extraction=StubExtraction(Recorder(), available=False))
    async with client() as c:
        body = (await c.post("/api/v1/pipeline/Bid1")).json()
    assert body["status"] == "partial" and body["stages"]["extraction"] == "skipped"
    assert any("Groq is not configured" in w for w in body["warnings"]) and body["result"] is not None


async def test_pipeline_unknown_bid_is_404() -> None:
    use_pipeline(ingestion=StubIngestion(Recorder(), exists=False))
    async with client() as c:
        res = await c.post("/api/v1/pipeline/Bid9")
    assert res.status_code == 404 and "Bid9" in res.json()["detail"]


async def test_pipeline_validation_failure_withholds_the_result() -> None:
    bad = ValidationReport(valid=False, issues=[ValidationIssue(severity="error", stage="reconciliation", field="Due Date", code="CITATION_NOT_FOUND", message="gone")])
    use_pipeline(validation=StubValidation(Recorder(), bad))
    async with client() as c:
        res = await c.post("/api/v1/pipeline/Bid1")
    body = res.json()
    # Our own output failed its checks: HTTP 500, with the structured explanation preserved.
    assert res.status_code == 500 and body["status"] == "failed" and body["failure"] == "validation" and body["result"] is None
    assert body["validation"]["issues"][0]["code"] == "CITATION_NOT_FOUND"


async def test_pipeline_service_failure_is_reported_without_internals() -> None:
    error = OperationalError("SELECT 1", {}, Exception("password authentication failed host=10.0.0.5"))
    use_pipeline(ingestion=StubIngestion(Recorder(), error=error))
    async with client() as c:
        res = await c.post("/api/v1/pipeline/Bid1")
    body = res.json()
    assert res.status_code == 503 and body["failure"] == "infrastructure"  # same code as /search and /qa use
    assert body["status"] == "failed" and body["result"] is None and body["stages"]["retrieval"] == "skipped"
    assert "10.0.0.5" not in res.text and "password" not in res.text


# ------------------------------------------------------------------------------- Q&A
async def test_qa_returns_a_cited_answer() -> None:
    use_qa(FakeLLM(script=[say(DEADLINE, ids=[12, 1])]))
    async with client() as c:
        res = await c.post("/api/v1/qa/Bid1", json={"question": "What is the final submission deadline?"})

    body = res.json()
    assert res.status_code == 200 and body["answerable"] and body["answer"] == DEADLINE
    assert body["bid_id"] == "Bid1" and body["question"] == "What is the final submission deadline?"
    assert {(s["file"], s["page"], s["chunk_id"]) for s in body["sources"]} == {("Addendum 2.pdf", 1, 12), ("JA-207652 FINAL.pdf", 2, 1)}
    assert 0 < body["confidence"] <= 0.95 and body["used_reconciliation"] is True


async def test_qa_refuses_when_evidence_is_insufficient() -> None:
    use_qa(FakeLLM(script=[say(None, ids=[], answerable=False)]))
    async with client() as c:
        body = (await c.post("/api/v1/qa/Bid1", json={"question": "Who won the contract?"})).json()
    assert body["answer"] == REFUSAL and body["answerable"] is False and body["sources"] == [] and body["confidence"] == 0


async def test_qa_unknown_bid_and_bad_input() -> None:
    use_qa(FakeLLM())
    async with client() as c:
        assert (await c.post("/api/v1/qa/Bid9", json={"question": "x?"})).status_code == 404
        assert (await c.post("/api/v1/qa/Bid1", json={"question": ""})).status_code == 422
        assert (await c.post("/api/v1/qa/Bid1", json={"question": "x" * 1001})).status_code == 422
        assert (await c.post("/api/v1/qa/Bid1", json={})).status_code == 422


async def test_qa_without_groq_is_503_and_names_only_the_setting() -> None:
    async def no_db():
        yield None

    app.dependency_overrides[get_db] = no_db
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, database_url="postgresql+psycopg://x/x", groq_api_key="super-secret")
    async with client() as c:
        res = await c.post("/api/v1/qa/Bid1", json={"question": "x?"})
    assert res.status_code == 503 and "GROQ_API_KEY" in res.json()["detail"] and "super-secret" not in res.text


async def test_qa_groq_failure_and_malformed_output_are_502() -> None:
    down = groq.APIConnectionError(request=httpx.Request("POST", "http://secret-host"))
    use_qa(FakeLLM(script=[down]))
    async with client() as c:
        res = await c.post("/api/v1/qa/Bid1", json={"question": "What is the deadline?"})
    assert res.status_code == 502 and "secret-host" not in res.text

    use_qa(FakeLLM(script=["nope", "still nope"]))
    async with client() as c:
        res = await c.post("/api/v1/qa/Bid1", json={"question": "What is the deadline?"})
    assert res.status_code == 502 and "invalid answer" in res.json()["detail"]


async def test_qa_database_failure_is_503() -> None:
    class Broken:
        async def available_filters(self):
            raise OperationalError("SELECT 1", {}, Exception("host=10.0.0.5"))

    agent, retrieval = build([RFP], FakeLLM())
    agent.retrieval.retrieval = Broken()
    app.dependency_overrides[get_qa] = lambda: agent
    async with client() as c:
        res = await c.post("/api/v1/qa/Bid1", json={"question": "x?"})
    assert res.status_code == 503 and res.json() == {"detail": "Database error"}


async def test_pipeline_with_nothing_to_analyse_is_422_with_a_structured_body() -> None:
    from app.agents.schema import DocumentInfo, IngestionResult

    nothing = IngestionResult(bid_id="Bid1", files_found=1, failed_files=["a.pdf"], documents=[DocumentInfo(document_id=1, file_name="a.pdf", document_type="rfp", status="failed")])
    use_pipeline(ingestion=StubIngestion(Recorder(), result=nothing))
    async with client() as c:
        res = await c.post("/api/v1/pipeline/Bid1")
    body = res.json()
    assert res.status_code == 422 and body["failure"] == "no_documents" and body["status"] == "failed"
    assert any("no document of this bid could be ingested" in e for e in body["errors"])


async def test_partial_pipeline_is_still_http_200_and_timings_are_recorded() -> None:
    use_pipeline(extraction=StubExtraction(Recorder(), available=False))
    async with client() as c:
        res = await c.post("/api/v1/pipeline/Bid1")
    body = res.json()
    assert res.status_code == 200 and body["status"] == "partial" and body["failure"] is None
    assert set(body["timings_ms"]) == {"ingestion", "retrieval", "extraction", "reconciliation", "validation", "total"}
