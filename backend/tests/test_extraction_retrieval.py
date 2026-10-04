"""Extraction on top of the REAL retrieval service (PostgreSQL), with a scripted LLM."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.embeddings.model import EmbeddingModelError, get_embedding_model
from app.embeddings.service import EmbeddingService
from app.extraction.evaluation import evidence_coverage, load_ground_truth, summarize
from app.extraction.evidence import gather_evidence
from app.extraction.service import ExtractionService
from app.retrieval.service import RetrievalService
from tests.extraction_fakes import FakeLLM
from tests.fakes import TopicEmbeddingModel, seed_corpus

SETTINGS = Settings(_env_file=None, database_url="postgresql+psycopg://x/x", max_top_k=10)


async def test_extraction_pulls_evidence_through_phase4_retrieval(pg_session) -> None:
    model = TopicEmbeddingModel()
    await seed_corpus(pg_session, model)
    retrieval = RetrievalService(pg_session, EmbeddingService(pg_session, model, 8), SETTINGS)
    answers = {
        "due_date": ([{"value": "Solicitation Due 27-JUN-2024", "chunk_ids": [2]},
                      {"value": "July 9, 2024", "chunk_ids": [1]}], 0.9),
    }  # fmt: skip
    llm = FakeLLM(answers=answers)

    result = await ExtractionService(retrieval, llm, SETTINGS).extract_bid("Bid1")

    prompt = next(c for c in llm.calls if 'key="due_date"' in c[1]["content"])[1]["content"]
    assert "The new due date for this RFP will be July 9, 2024" in prompt  # real chunk text from PostgreSQL
    assert "Dell Latitude" not in prompt  # Bid2 evidence is never mixed in
    due = result.fields["Due Date"]
    assert due.status == "conflict" and {s.page_number for s in due.sources} == {1, 2}
    assert result.reconciliation_required == ["Due Date"]


async def test_unknown_bid_is_rejected_by_the_real_service(pg_session) -> None:
    from app.extraction.service import UnknownBidError

    model = TopicEmbeddingModel()
    await seed_corpus(pg_session, model)
    retrieval = RetrievalService(pg_session, EmbeddingService(pg_session, model, 8), SETTINGS)
    with pytest.raises(UnknownBidError):
        await ExtractionService(retrieval, FakeLLM(), SETTINGS).extract_bid("Nope")


# ------------------------------------------------------------- real documents
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
        yield settings, RetrievalService.from_settings(session, settings)
    await engine.dispose()


async def test_real_due_date_evidence_contains_both_original_and_addendum(real) -> None:
    settings, retrieval = real
    evidence = await gather_evidence(retrieval, "Bid1", settings)
    due = evidence["due_date"]
    text_ = " ".join(" ".join(c.content.split()) for c in due)

    assert "27-JUN-2024" in text_ and "July 9, 2024" in text_
    assert {c.bid_id for c in due} == {"Bid1"}
    assert any(c.document_type == "addendum" and c.file_name.startswith("Addendum 2") for c in due)


async def test_real_evidence_coverage_against_ground_truth(real) -> None:
    """Regression guard on retrieval for extraction (needs no Groq key)."""
    settings, retrieval = real
    truth = load_ground_truth()
    for bid, checks in truth.items():
        results = await evidence_coverage(retrieval, settings, bid, checks)
        assert summarize(results)["rate"] >= 0.9, [r.id for r in results if r.scored and not r.passed]
