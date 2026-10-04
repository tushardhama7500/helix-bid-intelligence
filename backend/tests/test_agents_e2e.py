"""End-to-end pipeline runs.

* `test_full_pipeline_from_pdfs...` builds PDFs in a temp folder and uses a SCRIPTED model for
  extraction (a stand-in for Groq) to exercise every stage together on PostgreSQL.
* The `real_*` tests run on the actual Bid1/Bid2 documents in the `helix` database with no Groq,
  i.e. only deterministic paths; no model output is fabricated.
"""

import json
import re

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.extraction_agent import ExtractionAgent, ReconciliationAgent
from app.agents.factory import build_orchestrator
from app.agents.ingestion_agent import IngestionAgent
from app.agents.orchestrator import Orchestrator
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.validation_agent import ValidationAgent, db_chunk_lookup
from app.core.config import Settings, get_settings
from app.embeddings.model import EmbeddingModelError, get_embedding_model
from app.embeddings.service import EmbeddingService
from app.extraction.service import ExtractionService
from app.ingestion.chunker import Chunker
from app.ingestion.service import IngestionService
from app.models import Document, DocumentChunk
from app.reconciliation.service import ReconciliationService
from app.retrieval.service import RetrievalService
from tests.extraction_fakes import FakeLLM, llm_json
from tests.fakes import TopicEmbeddingModel
from tests.helpers import make_pdf

SETTINGS = Settings(_env_file=None, database_url="postgresql+psycopg://x/x", max_top_k=10)


def scripted_extraction(prompt: str) -> str:
    """Stands in for Groq: answers the due date from whatever chunks the prompt shows."""
    chunks = {file.strip(): int(i) for i, file in re.findall(r"\[chunk (\d+) \| ([^|]+) \|", prompt)}
    original = next((i for f, i in chunks.items() if "Addendum" not in f and f.startswith("RFP")), None)
    addendum = next((i for f, i in chunks.items() if f.startswith("Addendum")), None)
    answers = {}
    if original and addendum:
        answers["due_date"] = ([
            {"value": "27-JUN-2024 14:00:00", "chunk_ids": [original]},
            {"value": "July 9, 2024 at 2:00 PM CST", "chunk_ids": [addendum]},
        ], 0.9)  # fmt: skip
    return llm_json(prompt, answers)


def build(session, root, llm):
    model = TopicEmbeddingModel()
    embeddings = EmbeddingService(session, model, 8)
    retrieval = RetrievalService(session, embeddings, SETTINGS)
    return Orchestrator(
        ingestion=IngestionAgent(IngestionService(session, root, Chunker(300, 50)), embeddings),
        retrieval=RetrievalAgent(retrieval, SETTINGS),
        extraction=ExtractionAgent(ExtractionService(retrieval, llm, SETTINGS) if llm else None),
        reconciliation=ReconciliationAgent(ReconciliationService(retrieval, SETTINGS, llm)),
        validation=ValidationAgent(db_chunk_lookup(session)),
    )  # fmt: skip


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "data"
    make_pdf(root / "Bid1" / "RFP JA-1.pdf", ["Request for Proposal JA-1. Solicitation Due 27-JUN-2024 14:00:00. Proposal due date and deadline."])
    make_pdf(root / "Bid1" / "Addendum 2 RFP JA-1.pdf", ["ADDENDUM No. 2\nThe new due date for this RFP will be July 9, 2024 at 2:00 PM CST."])
    make_pdf(root / "Bid2" / "Other RFP.pdf", ["Solicitation Due 01-MAY-2030 10:00:00. Delivery within 45 days. Laptop warranty."])
    return root


async def test_full_pipeline_from_pdfs_with_a_scripted_model(pg_session, root) -> None:
    orchestrator = build(pg_session, root, FakeLLM(script=[scripted_extraction] * 5))
    state = await orchestrator.run("Bid1")

    assert state.status == "complete", (state.errors, state.warnings)
    assert {n: s.state for n, s in state.stages.items()} == dict.fromkeys(
        ["ingestion", "retrieval", "extraction", "reconciliation", "validation"], "complete"
    )
    assert state.ingestion.ingested == 2 and state.ingestion.chunks_embedded > 0
    assert state.extraction.fields["Due Date"].status == "conflict"  # Phase 5 reports it, picks nothing
    assert state.reconciliation.fields[0].key == "due_date"

    due = next(f for f in state.report.fields if f.field == "Due Date")
    assert due.value == "July 9, 2024 at 2:00 PM CST" and due.original_value == "27-JUN-2024 14:00:00"
    assert due.status == "resolved" and due.changed_by_addendum
    assert [s.file_name for s in due.sources] == ["Addendum 2 RFP JA-1.pdf"]
    assert [s.file_name for s in due.original_sources] == ["RFP JA-1.pdf"]
    assert state.validation.valid and state.validation.checked["citations"] > 0
    assert "01-MAY-2030" not in json.dumps(state.report.model_dump(), default=str)  # Bid2 never leaks in


async def test_pipeline_is_idempotent(pg_session, root) -> None:
    orchestrator = build(pg_session, root, FakeLLM(script=[scripted_extraction] * 10))
    first = await orchestrator.run("Bid1")
    docs = (await pg_session.execute(select(func.count()).select_from(Document))).scalar_one()
    chunks = (await pg_session.execute(select(func.count()).select_from(DocumentChunk))).scalar_one()

    second = await orchestrator.run("Bid1")

    assert second.ingestion.ingested == 0 and second.ingestion.skipped == 2 and second.ingestion.chunks_embedded == 0
    assert (await pg_session.execute(select(func.count()).select_from(Document))).scalar_one() == docs
    assert (await pg_session.execute(select(func.count()).select_from(DocumentChunk))).scalar_one() == chunks
    assert second.report.model_dump() == first.report.model_dump() and second.status == first.status


async def test_pipeline_without_groq_still_resolves_the_deterministic_part(pg_session, root) -> None:
    state = await build(pg_session, root, None).run("Bid1")

    assert state.status == "partial" and state.stages["extraction"].state == "skipped"
    due = next(f for f in state.report.fields if f.field == "Due Date")
    assert due.value == "July 9, 2024 at 2:00 PM CST" and due.original_value == "27-JUN-2024 14:00:00"
    assert next(f for f in state.report.fields if f.field == "Title").status == "not_extracted"


# ---------------------------------------------------------------- real documents
@pytest.fixture
async def real():
    settings = get_settings().model_copy(update={"groq_api_key": None})  # deterministic: never call Groq in tests
    engine = create_async_engine(settings.async_database_url)
    try:
        async with engine.connect() as conn:
            embedded = await conn.scalar(text("SELECT count(embedding) FROM document_chunks"))
            bids = (await conn.execute(text("SELECT DISTINCT bid_id FROM documents"))).scalars().all()
    except Exception:
        await engine.dispose()
        pytest.skip("real database not available")
    if embedded < 200 or {"Bid1", "Bid2"} - set(bids) or not (settings.data_path / "Bid1").is_dir():
        await engine.dispose()
        pytest.skip("real documents are not ingested and embedded")
    try:
        get_embedding_model(settings.embedding_model, settings.embedding_batch_size).load()
    except EmbeddingModelError as exc:
        await engine.dispose()
        pytest.skip(f"embedding model unavailable: {exc}")
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session, settings
    await engine.dispose()


async def test_real_bid1_pipeline_resolves_the_due_date_with_valid_citations(real) -> None:
    session, settings = real
    state = await build_orchestrator(session, settings, None).run("Bid1")

    assert state.status == "partial" and not state.fatal  # partial only because Groq is not configured
    assert {n: s.state for n, s in state.stages.items()} == {
        "ingestion": "complete", "retrieval": "complete", "extraction": "skipped",
        "reconciliation": "complete", "validation": "complete",
    }  # fmt: skip
    assert state.ingestion.skipped == 4 and state.ingestion.chunks_embedded == 0  # nothing re-done
    assert state.validation.valid and state.validation.errors == []

    due = next(f for f in state.reconciliation.fields if f.key == "due_date")
    assert due.final_value == "July 9, 2024 at 2:00 PM CST"
    assert due.original_value == "27-JUN-2024 14:00:00"
    (orig,) = due.original_sources
    assert (orig.file_name, orig.page_number) == ("JA-207652 Student and Staff Computing Devices FINAL.pdf", 2)
    (final,) = due.sources
    assert final.file_name.startswith("Addendum 2") and final.page_number == 1 and final.document_type == "addendum"

    report_due = next(f for f in state.report.fields if f.field == "Due Date")
    assert report_due.value == due.final_value and report_due.original_sources == due.original_sources
    # every cited chunk exists, belongs to Bid1, and contains the cited value
    for source, value in [(orig, "27-JUN-2024"), (final, "July 9, 2024 at 2:00 PM CST")]:
        bid, content = (await session.execute(text("SELECT bid_id, content FROM document_chunks WHERE id = :i"), {"i": source.chunk_id})).one()
        assert bid == "Bid1" and value in " ".join(content.split())


async def test_real_pipeline_is_idempotent_and_keeps_bids_isolated(real) -> None:
    session, settings = real
    orchestrator = build_orchestrator(session, settings, None)
    before = (await session.execute(text("SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM document_chunks)"))).one()

    first, second = await orchestrator.run("Bid1"), await orchestrator.run("Bid1")
    bid2 = await orchestrator.run("Bid2")

    after = (await session.execute(text("SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM document_chunks)"))).one()
    assert before == after
    assert first.report.model_dump() == second.report.model_dump()
    assert bid2.status == "partial" and bid2.validation.valid and bid2.reconciliation.timeline.addenda == []
    names = {s.file_name for f in first.report.fields for s in [*f.sources, *f.original_sources]}
    assert names and not any(n.startswith(("PORFP", "Dell", "Mercury", "Contract_")) for n in names)


async def test_real_unknown_bid(real) -> None:
    from app.extraction.service import UnknownBidError

    session, settings = real
    with pytest.raises(UnknownBidError):
        await build_orchestrator(session, settings, None).run("Bid99")
