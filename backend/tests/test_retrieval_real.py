"""Retrieval against the REAL ingested Bid1/Bid2 documents in the `helix` database.

Read-only. Skipped when the database has not been ingested and embedded
(`python -m app.ingestion && python -m app.embeddings`) or the model is unavailable.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.embeddings.model import EmbeddingModelError, get_embedding_model
from app.retrieval.evaluation import evaluate, load_dataset
from app.retrieval.service import HYBRID, RetrievalService

MIN_CHUNKS = 200  # the 9 real documents produce ~285 chunks


@pytest.fixture
async def real(request):
    settings = get_settings()
    engine = create_async_engine(settings.async_database_url)
    try:
        async with engine.connect() as conn:
            embedded = await conn.scalar(text("SELECT count(embedding) FROM document_chunks"))
            bids = (await conn.execute(text("SELECT DISTINCT bid_id FROM documents"))).scalars().all()
    except Exception:
        await engine.dispose()
        pytest.skip("real database not available")
    if embedded < MIN_CHUNKS or {"Bid1", "Bid2"} - set(bids):
        await engine.dispose()
        pytest.skip("real documents are not ingested and embedded")
    try:
        get_embedding_model(settings.embedding_model, settings.embedding_batch_size).load()
    except EmbeddingModelError as exc:
        await engine.dispose()
        pytest.skip(f"embedding model unavailable: {exc}")
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session, RetrievalService.from_settings(session, settings)
    await engine.dispose()


async def test_real_database_has_both_indexes_and_embeddings(real) -> None:
    session, _ = real
    defs = dict((await session.execute(text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename='document_chunks'"))).all())
    assert "gin (search_vector)" in defs["ix_document_chunks_search_vector"]
    assert "hnsw (embedding vector_cosine_ops)" in defs["ix_document_chunks_embedding_hnsw"]
    total, embedded, tsv = (await session.execute(text(
        "SELECT count(*), count(embedding), count(search_vector) FROM document_chunks"))).one()
    assert total == embedded == tsv >= MIN_CHUNKS
    assert await session.scalar(text("SELECT max(vector_dims(embedding)) FROM document_chunks")) == 384


async def test_final_submission_deadline_retrieves_addendum_2(real) -> None:
    _, service = real
    found = await service.retrieve("What is the final submission deadline for Bid1?", bid_id="Bid1", top_k=5)

    hit = [r for r in found.results if "Addendum 2" in r.candidate.file_name]
    assert hit, [r.candidate.file_name for r in found.results]
    assert "July 9, 2024" in hit[0].candidate.content
    assert hit[0].candidate.page_number == 1 and hit[0].candidate.document_type == "addendum"


@pytest.mark.parametrize(
    ("query", "needle"),
    [("JA-207652", "JA-207652"), ("BPM044557", "BPM044557"), ("Dell Latitude 5550", "Latitude 5550"), ("WD22TB4", "WD22TB4")],
)
async def test_exact_terms_rank_first_in_hybrid_and_keyword(real, query, needle) -> None:
    _, service = real
    for mode in (HYBRID, "keyword"):
        top = (await service.retrieve(query, mode=mode, top_k=1)).results[0].candidate
        assert needle.lower() in " ".join(top.content.split()).lower(), (mode, top.file_name)


async def test_html_citation_has_section_not_page(real) -> None:
    _, service = real
    top = (await service.retrieve("BPM044557", top_k=5)).results
    html = [r.candidate for r in top if r.candidate.file_name.endswith(".html")]
    assert html and all(c.page_number is None and "section" in c.metadata for c in html)


async def test_real_evaluation_recall_floor(real) -> None:
    """Regression guard, not a target: fails only if retrieval gets clearly worse."""
    _, service = real
    report = await evaluate(service, load_dataset(), use_filters=True)
    hybrid = report["summary"]["hybrid"]
    assert hybrid["recall@5"] >= 0.9
    assert hybrid["recall@3"] >= 0.8
    assert hybrid["recall@5"] >= report["summary"]["keyword"]["recall@5"]
