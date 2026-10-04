"""Reconciliation on real PostgreSQL: seeded corpus (isolation) and the REAL Bid1 documents."""

import json

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.embeddings.model import EmbeddingModelError, get_embedding_model
from app.embeddings.service import EmbeddingService
from app.extraction.validation import grounding
from app.reconciliation.addenda import load_timeline
from app.reconciliation.service import ReconciliationService
from app.retrieval.service import RetrievalService
from tests.fakes import TopicEmbeddingModel, seed_corpus

SETTINGS = Settings(_env_file=None, database_url="postgresql+psycopg://x/x", max_top_k=10)

CORPUS = [
    ("Bid1", "JA-1 FINAL.pdf", "rfp", 2, None, "Solicitation Due 27-JUN-2024 14:00:00. Proposal due date and deadline."),
    ("Bid1", "Addendum 2 RFP.pdf", "addendum", 1, None,
     "ADDENDUM No. 2\nThe new due date for this RFP will be July 9, 2024 at 2:00 PM CST."),
    ("Bid2", "PORFP.pdf", "rfp", 1, None, "Solicitation Due 01-MAY-2024 10:00:00. Proposal due date and deadline."),
    ("Bid2", "Addendum 1 PORFP.pdf", "addendum", 1, None,
     "ADDENDUM No. 1\nThe new due date for this RFP will be August 1, 2024 at 9:00 AM EST."),
    ("Bid2", "Notice.pdf", "bid_document", 1, None, "ADDENDUM No. 2\nThe pre-bid meeting is amended to 11-JUN-2024 10:00:00."),
]  # fmt: skip


async def test_isolation_and_content_based_addendum_detection_in_postgres(pg_session) -> None:
    model = TopicEmbeddingModel()
    await seed_corpus(pg_session, model, CORPUS)
    retrieval = RetrievalService(pg_session, EmbeddingService(pg_session, model, 8), SETTINGS)
    service = ReconciliationService(retrieval, SETTINGS)

    bid1 = await service.reconcile_bid("Bid1")
    bid2 = await service.reconcile_bid("Bid2")

    due1 = next(f for f in bid1.fields if f.key == "due_date")
    due2 = next(f for f in bid2.fields if f.key == "due_date")
    assert due1.final_value == "July 9, 2024 at 2:00 PM CST" and due2.final_value == "August 1, 2024 at 9:00 AM EST"
    assert "August" not in json.dumps(bid1.model_dump(), default=str)  # nothing from Bid2 reached Bid1
    assert "July 9" not in json.dumps(bid2.model_dump(), default=str)
    assert {s.file_name for s in due1.original_sources + due1.sources} == {"JA-1 FINAL.pdf", "Addendum 2 RFP.pdf"}

    # An addendum is also recognised by its header, not only by document_type / file name.
    timeline = await load_timeline(pg_session, "Bid2")
    assert [(e.addendum_number, e.identified_by) for e in timeline.addenda] == [(1, "document_type"), (2, "content")]
    assert timeline.original_documents == ["PORFP.pdf"] and timeline.ordering == "explicit_number"


# ---------------------------------------------------------------- the real documents
@pytest.fixture
async def real():
    settings = get_settings()
    engine = create_async_engine(settings.async_database_url)
    try:
        async with engine.connect() as conn:
            embedded = await conn.scalar(text("SELECT count(embedding) FROM document_chunks"))
            bids = (await conn.execute(text("SELECT DISTINCT bid_id FROM documents"))).scalars().all()
    except Exception:
        await engine.dispose()
        pytest.skip("real database not available")
    if embedded < 200 or {"Bid1", "Bid2"} - set(bids):
        await engine.dispose()
        pytest.skip("real documents are not ingested and embedded")
    try:
        get_embedding_model(settings.embedding_model, settings.embedding_batch_size).load()
    except EmbeddingModelError as exc:
        await engine.dispose()
        pytest.skip(f"embedding model unavailable: {exc}")
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session, settings, RetrievalService.from_settings(session, settings)
    await engine.dispose()


async def test_real_bid1_due_date_resolves_from_june_27_to_july_9(real) -> None:
    session, settings, retrieval = real
    result = await ReconciliationService(retrieval, settings).reconcile_bid("Bid1")
    due = next(f for f in result.fields if f.key == "due_date")

    # Resolved to Addendum 2's value, deterministically.
    assert due.status == "resolved" and due.resolution_method == "explicit_addendum_change"
    assert due.final_value == "July 9, 2024 at 2:00 PM CST"
    assert "June" not in str(due.final_value) and "27-JUN" not in str(due.final_value)
    # The original is preserved, with its own citation.
    assert due.original_value is not None and "27-JUN-2024" in due.original_value
    assert [s.file_name for s in due.original_sources] == ["JA-207652 Student and Staff Computing Devices FINAL.pdf"]
    assert due.original_sources[0].page_number == 2
    # The history cites Addendum 2, and the final value cites it too.
    (change,) = due.changes
    assert change.addendum_number == 2 and change.change_type == "replacement" and change.method == "rule"
    assert change.previous_value == due.original_value and change.new_value == due.final_value
    assert change.sources and change.sources[0].file_name.startswith("Addendum 2")
    assert change.sources[0].document_type == "addendum" and change.sources[0].page_number == 1
    assert due.sources == change.sources and 0 < due.confidence <= 0.95
    # Citations point at real stored chunks, and the cited text contains the value (grounding).
    for source in due.original_sources + due.sources:
        content = await session.scalar(text("SELECT content FROM document_chunks WHERE id = :i"), {"i": source.chunk_id})
        assert content, source
    cited = await session.scalar(text("SELECT content FROM document_chunks WHERE id = :i"), {"i": due.sources[0].chunk_id})
    assert "July 9, 2024 at 2:00 PM CST" in " ".join(cited.split())

    # Timeline: Addendum 1 -> Addendum 2, ordered by the numbers in their own headers.
    assert [e.addendum_number for e in result.timeline.addenda] == [1, 2]
    assert result.timeline.ordering == "explicit_number" and result.timeline.reliable
    assert {e.number_source for e in result.timeline.addenda} == {"content"}


async def test_real_bid2_has_no_addenda_so_nothing_is_changed(real) -> None:
    _, settings, retrieval = real
    result = await ReconciliationService(retrieval, settings).reconcile_bid("Bid2")

    assert result.timeline.addenda == [] and result.timeline.ordering == "none"
    assert not any(f.changes for f in result.fields) and not any(f.status == "resolved" for f in result.fields)


async def test_real_bid1_never_cites_bid2_documents(real) -> None:
    _, settings, retrieval = real
    result = await ReconciliationService(retrieval, settings).reconcile_bid("Bid1")
    names = {s.file_name for f in result.fields for s in (f.original_sources + f.sources + [x for c in f.changes for x in c.sources])}
    assert names and not any(n.startswith(("PORFP", "Dell", "Mercury", "Contract_")) for n in names)
