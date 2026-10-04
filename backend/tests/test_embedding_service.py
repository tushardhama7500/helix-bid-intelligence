"""Embedding pipeline against real PostgreSQL + pgvector (fake model, no download)."""

import pytest
from sqlalchemy import text

from app.embeddings.model import EmbeddingModelError
from app.embeddings.schema import INDEX_NAME, ensure_embedding_schema
from app.embeddings.service import EmbeddingService
from app.ingestion.chunker import Chunker
from app.ingestion.service import IngestionService
from tests.fakes import DIM, FakeEmbeddingModel, seed_chunks
from tests.helpers import lorem, make_pdf


def _service(session, model=None, batch_size=4) -> EmbeddingService:
    return EmbeddingService(session, model or FakeEmbeddingModel(), batch_size)


async def _scalar(session, sql: str):
    return (await session.execute(text(sql))).scalar_one()


async def test_embed_text_and_texts_handle_blanks(pg_session) -> None:
    service = _service(pg_session)

    vector = await service.embed_text("hello")
    assert len(vector) == DIM
    with pytest.raises(ValueError):
        await service.embed_text("   ")

    vectors = await service.embed_texts(["a", "", "b", "  "])
    assert [v is None for v in vectors] == [False, True, False, True]
    assert all(len(v) == DIM for v in vectors if v)
    assert await service.embed_texts([]) == []


async def test_generate_stores_vectors_with_correct_dimension(pg_session) -> None:
    await seed_chunks(pg_session, [f"chunk number {i}" for i in range(10)])
    model = FakeEmbeddingModel()

    summary = await _service(pg_session, model).generate()

    assert (summary.total_chunks, summary.embedded, summary.skipped, summary.failed) == (10, 10, 0, 0)
    assert await _scalar(pg_session, "SELECT count(embedding) FROM document_chunks") == 10
    assert await _scalar(pg_session, "SELECT min(vector_dims(embedding)) FROM document_chunks") == DIM
    assert await _scalar(pg_session, "SELECT max(vector_dims(embedding)) FROM document_chunks") == DIM
    assert await _scalar(pg_session, "SELECT min(embedding_model) FROM document_chunks") == model.name
    # 10 chunks / batch of 4 -> 3 batched model calls, not 10.
    assert model.calls == 3


async def test_generate_is_idempotent(pg_session) -> None:
    ids = await seed_chunks(pg_session, [f"chunk {i}" for i in range(6)])
    model = FakeEmbeddingModel()
    service = _service(pg_session, model)

    first = await service.generate()
    second = await service.generate()

    assert (first.embedded, first.skipped) == (6, 0)
    assert (second.total_chunks, second.embedded, second.skipped, second.failed) == (6, 0, 6, 0)
    assert len(model.embedded) == 6  # nothing regenerated

    # A newly added chunk is the only thing embedded next time.
    await pg_session.execute(
        text(
            "INSERT INTO document_chunks (document_id, page_id, bid_id, chunk_index, content, metadata) "
            "SELECT document_id, page_id, bid_id, 99, 'brand new chunk', '{}' "
            "FROM document_chunks WHERE id = :id"
        ),
        {"id": ids[0]},
    )
    await pg_session.commit()
    third = await service.generate()
    assert (third.embedded, third.skipped) == (1, 6)
    assert model.embedded[-1] == "brand new chunk"


async def test_failures_are_reported_and_isolated(pg_session) -> None:
    await seed_chunks(pg_session, ["good one", "BAD", "good two", "good three", "good four"])
    model = FakeEmbeddingModel(fail_on={"BAD"})
    service = _service(pg_session, model, batch_size=5)

    summary = await service.generate()

    assert (summary.embedded, summary.failed) == (4, 1)
    assert "simulated embedding failure" in summary.errors[0]
    counts = await service.status()
    assert (counts.embedded, counts.failed, counts.pending) == (4, 1, 0)
    assert await _scalar(pg_session, "SELECT embedding IS NULL FROM document_chunks WHERE content = 'BAD'")

    # The failed chunk is retried on the next run, and succeeds once the fault is gone.
    model.fail_on.clear()
    retry = await service.generate()
    assert (retry.embedded, retry.skipped, retry.failed) == (1, 4, 0)
    counts = await service.status()
    assert (counts.embedded, counts.failed) == (5, 0)


async def test_model_load_failure_is_raised_not_swallowed(pg_session) -> None:
    await seed_chunks(pg_session, ["x"])
    service = _service(pg_session, FakeEmbeddingModel(load_error="cannot download"))

    with pytest.raises(EmbeddingModelError):
        await service.generate()
    assert await _scalar(pg_session, "SELECT count(embedding) FROM document_chunks") == 0


async def test_reingested_document_does_not_keep_stale_embeddings(pg_session, tmp_path) -> None:
    root = tmp_path / "data"
    pdf = make_pdf(root / "Bid1" / "a.pdf", [lorem(20, "OLD")])
    ingestion = IngestionService(pg_session, root, Chunker(300, 50))
    model = FakeEmbeddingModel()
    service = _service(pg_session, model, batch_size=8)

    await ingestion.run()
    first = await service.generate()
    assert first.embedded > 0 and first.failed == 0

    make_pdf(pdf, [lorem(20, "NEW")])
    assert (await ingestion.run()).processed == 1

    counts = await service.status()
    assert counts.embedded == 0 and counts.pending == counts.total > 0
    assert await _scalar(pg_session, "SELECT count(*) FROM document_chunks WHERE content LIKE '%OLD%'") == 0
    second = await service.generate()
    assert second.embedded == counts.total and second.skipped == 0


async def test_vector_index_and_schema_are_idempotent(pg_engine, pg_session) -> None:
    async def index_def() -> str:
        return await _scalar(
            pg_session, f"SELECT indexdef FROM pg_indexes WHERE indexname = '{INDEX_NAME}'"
        )

    definition = await index_def()
    assert "hnsw" in definition and "vector_cosine_ops" in definition

    await pg_session.rollback()  # release read locks so the ALTER TABLE can run
    async with pg_engine.begin() as conn:  # running it again must not fail
        await ensure_embedding_schema(conn, FakeEmbeddingModel().name, DIM)
    assert await index_def() == definition


async def test_model_or_dimension_change_clears_old_vectors(pg_engine, pg_session) -> None:
    await seed_chunks(pg_session, ["a", "b"])
    model = FakeEmbeddingModel()
    await _service(pg_session, model).generate()

    await pg_session.rollback()
    async with pg_engine.begin() as conn:  # different model, same size
        await ensure_embedding_schema(conn, "other/model", DIM)
    assert await _scalar(pg_session, "SELECT count(embedding) FROM document_chunks") == 0

    await _service(pg_session, model).generate()
    await pg_session.rollback()
    async with pg_engine.begin() as conn:  # different dimension
        await ensure_embedding_schema(conn, model.name, 16)
    assert await _scalar(pg_session, "SELECT count(embedding) FROM document_chunks") == 0
    assert await _scalar(
        pg_session,
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding'",
    ) == 16
