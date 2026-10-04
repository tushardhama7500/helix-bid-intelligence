"""Database access for ingestion. No parsing or chunking logic lives here."""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunker import ChunkRecord
from app.ingestion.discovery import DiscoveredFile
from app.models import Document, DocumentChunk, DocumentPage, DocumentStatus

_MAX_ERROR_LEN = 2000


async def get_by_path(session: AsyncSession, file_path: str) -> Document | None:
    result = await session.execute(select(Document).where(Document.file_path == file_path))
    return result.scalar_one_or_none()


async def start_document(
    session: AsyncSession,
    existing: Document | None,
    file: DiscoveredFile,
    content_hash: str,
    document_type: str,
) -> int:
    """Create the document row, or reset an existing one for reprocessing.

    Old pages and chunks are removed, so re-ingesting never leaves duplicates.
    Commits so the `processing` state is visible even if later steps fail.
    """
    if existing is not None:
        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == existing.id))
        await session.execute(delete(DocumentPage).where(DocumentPage.document_id == existing.id))
        document = existing
    else:
        document = Document(file_path=file.rel_path)
        session.add(document)

    document.bid_id = file.bid_id
    document.file_name = file.file_name
    document.file_type = file.file_type
    document.document_type = document_type
    document.content_hash = content_hash
    document.status = DocumentStatus.PROCESSING.value
    document.error_message = None
    document.title = None
    document.page_count = None

    await session.flush()
    document_id = document.id
    await session.commit()
    return document_id


async def add_pages(
    session: AsyncSession,
    document_id: int,
    pages: Sequence[tuple[int | None, str, dict[str, Any]]],
) -> list[int]:
    """Insert (page_number, content, metadata) rows; returns ids in input order."""
    rows = [
        DocumentPage(document_id=document_id, page_number=number, content=content, meta=meta)
        for number, content, meta in pages
    ]
    session.add_all(rows)
    await session.flush()
    return [row.id for row in rows]


async def add_chunks(
    session: AsyncSession, document_id: int, bid_id: str, chunks: Sequence[ChunkRecord]
) -> None:
    session.add_all(
        DocumentChunk(
            document_id=document_id,
            page_id=chunk.page_id,
            bid_id=bid_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            meta=chunk.metadata,
        )
        for chunk in chunks
    )
    await session.flush()


async def complete_document(
    session: AsyncSession, document_id: int, title: str | None, page_count: int | None
) -> None:
    await session.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(
            status=DocumentStatus.COMPLETED.value,
            title=title,
            page_count=page_count,
            error_message=None,
        )
    )
    await session.commit()


async def fail_document(session: AsyncSession, document_id: int, message: str) -> None:
    await session.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(status=DocumentStatus.FAILED.value, error_message=message[:_MAX_ERROR_LEN])
    )
    await session.commit()


async def list_documents(session: AsyncSession, bid_id: str | None = None) -> list[Document]:
    query = select(Document).order_by(Document.bid_id, Document.file_name)
    if bid_id:
        query = query.where(Document.bid_id == bid_id)
    return list((await session.execute(query)).scalars())


async def get_document(session: AsyncSession, document_id: int) -> Document | None:
    return await session.get(Document, document_id)


async def get_page_stats(session: AsyncSession, document_id: int) -> list[dict[str, Any]]:
    """Per stored page: number, section, character count and chunk count."""
    pages = await session.execute(
        select(
            DocumentPage.id,
            DocumentPage.page_number,
            DocumentPage.meta,
            func.length(DocumentPage.content),
        )
        .where(DocumentPage.document_id == document_id)
        .order_by(DocumentPage.page_number, DocumentPage.id)
    )
    counts = dict(
        (
            await session.execute(
                select(DocumentChunk.page_id, func.count())
                .where(DocumentChunk.document_id == document_id)
                .group_by(DocumentChunk.page_id)
            )
        ).all()
    )
    return [
        {
            "page_number": number,
            "section": (meta or {}).get("section"),
            "characters": length,
            "chunks": counts.get(page_id, 0),
        }
        for page_id, number, meta, length in pages.all()
    ]
