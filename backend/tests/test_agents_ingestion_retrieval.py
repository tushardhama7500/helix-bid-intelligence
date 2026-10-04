import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401
from app.agents.ingestion_agent import IngestionAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.core.config import Settings
from app.core.database import Base
from app.extraction.fields import FIELDS_BY_KEY
from app.ingestion.chunker import Chunker
from app.ingestion.service import IngestionService
from app.models import Document, DocumentChunk
from tests.extraction_fakes import BID1, BID2, StubRetrieval
from tests.helpers import lorem, make_pdf

SETTINGS = Settings(_env_file=None, database_url="postgresql+psycopg://x/x")


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as s:
        yield s
    await engine.dispose()


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "data"
    make_pdf(root / "Bid1" / "RFP.pdf", [lorem(15, "A")])
    make_pdf(root / "Bid1" / "Addendum 1 RFP.pdf", ["ADDENDUM No. 1 " + lorem(6, "B")])
    make_pdf(root / "Bid2" / "Other.pdf", [lorem(10, "C")])
    return root


def agent(session, root) -> IngestionAgent:
    return IngestionAgent(IngestionService(session, root, Chunker(300, 50)), None)


# ------------------------------------------------------------------ ingestion agent
async def test_ingests_only_the_requested_bid_and_reports_documents(session, data_dir) -> None:
    result = await agent(session, data_dir).run("Bid1")

    assert (result.files_found, result.ingested, result.skipped, result.failed_files) == (2, 2, 0, [])
    assert sorted(d.document_type for d in result.documents) == ["addendum", "rfp"]
    assert all(d.status == "completed" and d.document_id for d in result.documents)
    assert {d.bid_id for d in (await session.execute(select(Document))).scalars()} == {"Bid1"}  # Bid2 untouched


async def test_second_run_skips_unchanged_files(session, data_dir) -> None:
    a = agent(session, data_dir)
    await a.run("Bid1")
    chunks = (await session.execute(select(func.count()).select_from(DocumentChunk))).scalar_one()

    again = await a.run("Bid1")

    assert (again.ingested, again.skipped) == (0, 2)
    assert (await session.execute(select(func.count()).select_from(DocumentChunk))).scalar_one() == chunks


async def test_ingestion_failure_is_reported_per_document(session, data_dir) -> None:
    (data_dir / "Bid1" / "broken.pdf").write_bytes(b"not a pdf")
    result = await agent(session, data_dir).run("Bid1")

    assert result.ingested == 2 and result.failed_files == ["broken.pdf"]
    assert {d.file_name: d.status for d in result.documents}["broken.pdf"] == "failed"
    assert len(result.completed_documents) == 2


async def test_ingest_false_only_reports_stored_state(session, data_dir) -> None:
    a = agent(session, data_dir)
    assert (await a.run("Bid1", ingest=False)).documents == []
    await a.run("Bid1")
    assert len((await a.run("Bid1", ingest=False)).documents) == 2


async def test_bid_exists_by_disk_or_database(session, data_dir, tmp_path) -> None:
    a = agent(session, data_dir)
    assert await a.bid_exists("Bid1") and not await a.bid_exists("Bid9")
    await a.run("Bid1")
    no_disk = agent(session, tmp_path / "missing")
    assert await no_disk.bid_exists("Bid1") and not await no_disk.bid_exists("Bid2")
    result = await no_disk.run("Bid1")
    assert result.files_found == 0 and result.notes and len(result.documents) == 2


async def test_embedding_runs_only_when_something_is_pending(session, data_dir) -> None:
    from types import SimpleNamespace

    class FakeEmbeddings:
        def __init__(self, pending):
            self.pending, self.generated = pending, 0

        async def status(self):
            return SimpleNamespace(pending=self.pending, failed=0)

        async def generate(self):
            self.generated += 1
            return SimpleNamespace(embedded=7, failed=1)

    idle, busy = FakeEmbeddings(0), FakeEmbeddings(3)
    service = IngestionService(session, data_dir, Chunker(300, 50))
    assert (await IngestionAgent(service, idle).run("Bid1")).chunks_embedded == 0 and idle.generated == 0
    result = await IngestionAgent(service, busy).run("Bid1")
    assert (result.chunks_embedded, result.embedding_failures, busy.generated) == (7, 1, 1)


# ------------------------------------------------------------------ retrieval agent
class Leaky(StubRetrieval):
    """Ignores the bid filter and returns every bid's chunks."""

    async def retrieve(self, query, bid_id=None, document_type=None, top_k=5, **_):
        return await super().retrieve(query, bid_id=None, document_type=document_type, top_k=top_k) if False else await self._all(document_type, top_k)

    async def _all(self, document_type, top_k):
        from app.retrieval.schema import FusedResult
        from tests.extraction_fakes import Found

        chunks = [c for c in [*BID1, *BID2] if document_type in (None, c.document_type)][:top_k]
        return Found([FusedResult(candidate=c, rrf_score=0.01) for c in chunks])


async def test_retrieval_agent_scopes_every_request_to_the_bid() -> None:
    stub = StubRetrieval()
    a = RetrievalAgent(stub, SETTINGS)
    found = await a.retrieve("Bid1", "due date", top_k=3)
    chunks = await a.retrieve_field("Bid2", FIELDS_BY_KEY["delivery_date"])

    assert found and {r.candidate.bid_id for r in found} == {"Bid1"}
    assert chunks and {c.bid_id for c in chunks} == {"Bid2"}
    assert {c["bid_id"] for c in stub.calls} == {"Bid1", "Bid2"}


async def test_retrieval_agent_drops_foreign_chunks_even_if_retrieval_leaks() -> None:
    a = RetrievalAgent(Leaky(), SETTINGS)
    assert {r.candidate.bid_id for r in await a.retrieve("Bid1", "x")} == {"Bid1"}

    evidence, stage = await a.gather("Bid1")
    assert {c.bid_id for chunks in evidence.values() for c in chunks} == {"Bid1"}
    assert stage.dropped_foreign_chunks > 0 and stage.fields == 20 and stage.total_chunks == 3


async def test_gather_reports_counts_and_empty_fields() -> None:
    evidence, stage = await RetrievalAgent(StubRetrieval({"Empty": []}), SETTINGS).gather("Empty")
    assert stage.total_chunks == 0 and len(stage.empty_fields) == 20


async def test_retrieval_failure_propagates() -> None:
    with pytest.raises(RuntimeError):
        await RetrievalAgent(StubRetrieval(error=RuntimeError("db down")), SETTINGS).gather("Bid1")
