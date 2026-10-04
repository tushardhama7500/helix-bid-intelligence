"""End-to-end ingestion against in-memory SQLite (no PostgreSQL needed)."""

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401
from app.core.database import Base
from app.ingestion.chunker import Chunker
from app.ingestion.service import IngestionService
from app.models import Document, DocumentChunk, DocumentPage
from tests.helpers import lorem, make_pdf


@pytest.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as s:
        yield s
    await engine.dispose()


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "data"
    make_pdf(root / "Bid1" / "Addendum 2 RFP X.pdf", [lorem(25, "A"), "", lorem(25, "B")])
    (root / "Bid2").mkdir(parents=True)
    (root / "Bid2" / "notice.html").write_text(
        "<h1>Notice</h1><p>" + lorem(20) + "</p><h2>Scope</h2><p>Scope text.</p>",
        encoding="utf-8",
    )
    (root / "Bid2" / "readme.txt").write_text("ignored")
    return root


def _service(session, data_dir) -> IngestionService:
    return IngestionService(session, data_dir, Chunker(300, 50))


async def _count(session, model) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def test_ingests_and_persists_documents_pages_chunks(session, data_dir) -> None:
    summary = await _service(session, data_dir).run()

    assert (summary.total, summary.processed, summary.skipped, summary.failed) == (2, 2, 0, 0)

    docs = {d.file_name: d for d in (await session.execute(select(Document))).scalars()}
    pdf = docs["Addendum 2 RFP X.pdf"]
    assert (pdf.bid_id, pdf.document_type, pdf.file_type) == ("Bid1", "addendum", "pdf")
    assert pdf.page_count == 3 and pdf.status == "completed" and len(pdf.content_hash) == 64
    assert docs["notice.html"].page_count is None
    assert docs["notice.html"].title == "Notice"

    # Blank PDF page 2 is not stored; pages 1 and 3 keep their real numbers.
    numbers = (
        await session.execute(
            select(DocumentPage.page_number)
            .where(DocumentPage.document_id == pdf.id)
            .order_by(DocumentPage.page_number)
        )
    ).scalars().all()
    assert list(numbers) == [1, 3]

    chunks = (
        await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == pdf.id)
            .order_by(DocumentChunk.chunk_index)
        )
    ).scalars().all()
    assert len(chunks) > 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.bid_id == "Bid1"
        assert c.meta["file_name"] == "Addendum 2 RFP X.pdf"
        assert c.meta["document_type"] == "addendum"
        assert c.meta["page_number"] in (1, 3)
        assert c.meta["chunk_index"] == c.chunk_index

    html_chunk = (
        await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == docs["notice.html"].id)
        )
    ).scalars().first()
    assert html_chunk.meta["page_number"] is None
    assert html_chunk.meta["section"] == "Notice"


async def test_second_run_skips_unchanged_files(session, data_dir) -> None:
    service = _service(session, data_dir)
    await service.run()
    chunks_before = await _count(session, DocumentChunk)

    summary = await service.run()

    assert (summary.processed, summary.skipped, summary.failed) == (0, 2, 0)
    assert await _count(session, Document) == 2
    assert await _count(session, DocumentChunk) == chunks_before


async def test_changed_file_is_reprocessed_without_duplicates(session, data_dir) -> None:
    service = _service(session, data_dir)
    await service.run()
    old_hash = (
        await session.execute(select(Document.content_hash).where(Document.file_type == "html"))
    ).scalar_one()

    (data_dir / "Bid2" / "notice.html").write_text(
        "<h1>Notice</h1><p>Short new text.</p>", encoding="utf-8"
    )
    summary = await service.run()

    assert (summary.processed, summary.skipped) == (1, 1)
    assert await _count(session, Document) == 2
    doc = (await session.execute(select(Document).where(Document.file_type == "html"))).scalar_one()
    assert doc.content_hash != old_hash and doc.status == "completed"
    contents = (
        await session.execute(select(DocumentChunk.content).where(DocumentChunk.document_id == doc.id))
    ).scalars().all()
    assert len(contents) == 1 and "Short new text." in contents[0]


async def test_failed_document_is_tracked_and_retried(session, data_dir) -> None:
    bad = data_dir / "Bid1" / "broken.pdf"
    bad.write_bytes(b"definitely not a pdf")
    service = _service(session, data_dir)

    summary = await service.run()

    assert (summary.processed, summary.failed) == (2, 1)
    doc = (
        await session.execute(select(Document).where(Document.file_name == "broken.pdf"))
    ).scalar_one()
    assert doc.status == "failed" and doc.error_message

    # The same (still broken) file is retried, not skipped; fixing it completes it.
    assert (await service.run()).failed == 1
    make_pdf(bad, ["Now it is valid."])
    summary = await service.run()
    assert (summary.processed, summary.failed) == (1, 0)
    await session.refresh(doc)
    assert doc.status == "completed" and doc.error_message is None


async def test_document_without_text_fails(session, tmp_path) -> None:
    root = tmp_path / "data"
    make_pdf(root / "Bid1" / "blank.pdf", ["", ""])

    summary = await _service(session, root).run()

    assert summary.failed == 1
    assert "No extractable text" in summary.results[0].message
