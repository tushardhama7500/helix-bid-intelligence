"""Orchestrates: discover -> hash -> parse -> normalize -> classify -> chunk -> store."""

import asyncio
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.ingestion import repository as repo
from app.ingestion.chunker import Chunker, ChunkSource
from app.ingestion.discovery import DiscoveredFile, DiscoveryResult, discover_files
from app.ingestion.html_parser import parse_html
from app.ingestion.metadata import build_page_metadata, detect_document_type
from app.ingestion.normalizer import normalize_pages
from app.ingestion.parsed import ParsedDocument
from app.ingestion.pdf_parser import parse_pdf
from app.models import DocumentStatus

logger = logging.getLogger(__name__)

PARSERS: dict[str, Callable[[Path], ParsedDocument]] = {
    "pdf": parse_pdf,
    "html": parse_html,
}

PROCESSED, SKIPPED, FAILED = "processed", "skipped", "failed"


@dataclass
class FileResult:
    file_path: str
    status: str  # processed | skipped | failed
    document_id: int | None = None
    pages: int = 0
    chunks: int = 0
    message: str | None = None


@dataclass
class IngestionSummary:
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    results: list[FileResult] = field(default_factory=list)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionService:
    def __init__(
        self, session: AsyncSession, data_dir: Path, chunker: Chunker, max_file_bytes: int | None = None
    ) -> None:
        self.session = session
        self.data_dir = data_dir
        self.chunker = chunker
        self.max_file_bytes = max_file_bytes

    @classmethod
    def from_settings(cls, session: AsyncSession, settings: Settings) -> "IngestionService":
        chunker = Chunker(settings.chunk_size, settings.chunk_overlap)
        return cls(session, settings.data_path, chunker, settings.max_ingest_file_mb * 1024 * 1024)

    def scan(self) -> DiscoveryResult:
        return discover_files(self.data_dir, self.max_file_bytes)

    async def run(self, force: bool = False) -> IngestionSummary:
        """Ingest everything; `force` reprocesses files even if unchanged."""
        summary = IngestionSummary()
        for file in self.scan().files:
            result = await self.ingest_file(file, force=force)
            summary.results.append(result)
            summary.total += 1
            setattr(summary, result.status, getattr(summary, result.status) + 1)
        logger.info(
            "Ingestion run finished: %d processed, %d skipped, %d failed",
            summary.processed, summary.skipped, summary.failed,
        )  # fmt: skip
        return summary

    async def ingest_file(self, file: DiscoveredFile, force: bool = False) -> FileResult:
        """Ingest one file; never raises for per-document problems."""
        document_type = detect_document_type(file.file_name)

        try:
            content_hash = await asyncio.to_thread(hash_file, file.path)
            existing = await repo.get_by_path(self.session, file.rel_path)
            if (
                not force
                and existing is not None
                and existing.content_hash == content_hash
                and existing.status == DocumentStatus.COMPLETED
            ):
                logger.info("Skipping unchanged document %s", file.rel_path)
                return FileResult(file.rel_path, SKIPPED, existing.id, message="unchanged")
            document_id = await repo.start_document(
                self.session, existing, file, content_hash, document_type
            )
        except Exception as exc:
            # No document row to mark yet (e.g. unreadable file or DB error).
            await self.session.rollback()
            logger.exception("Document ingestion failed: %s", file.rel_path)
            return FileResult(file.rel_path, FAILED, message=str(exc))

        try:
            return await self._process(file, document_id, document_type)
        except Exception as exc:
            await self.session.rollback()
            logger.exception("Document ingestion failed: %s", file.rel_path)
            try:
                await repo.fail_document(self.session, document_id, f"{type(exc).__name__}: {exc}")
            except Exception:
                await self.session.rollback()
                logger.exception("Could not record failure for %s", file.rel_path)
            return FileResult(file.rel_path, FAILED, document_id, message=str(exc))

    async def _process(
        self, file: DiscoveredFile, document_id: int, document_type: str
    ) -> FileResult:
        parsed = await asyncio.to_thread(PARSERS[file.file_type], file.path)
        pages = normalize_pages(parsed.pages, file.file_name)
        if not pages:
            raise ValueError("No extractable text found (scanned or empty document?)")

        page_ids = await repo.add_pages(
            self.session,
            document_id,
            [
                (p.page_number, p.content, build_page_metadata(file, document_type, p))
                for p in pages
            ],
        )
        chunks = self.chunker.chunk_pages(
            [
                ChunkSource(page_id, p.content, build_page_metadata(file, document_type, p))
                for page_id, p in zip(page_ids, pages, strict=True)
            ]
        )
        logger.info("Created %d chunks for %s", len(chunks), file.rel_path)
        await repo.add_chunks(self.session, document_id, file.bid_id, chunks)
        await repo.complete_document(
            self.session,
            document_id,
            title=parsed.title or Path(file.file_name).stem,
            page_count=parsed.page_count,
        )
        logger.info("Document ingestion completed: %s", file.rel_path)
        return FileResult(
            file.rel_path, PROCESSED, document_id, pages=len(pages), chunks=len(chunks)
        )
