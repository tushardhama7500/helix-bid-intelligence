"""Ingestion agent: a thin wrapper over the Phase 2 ingestion and Phase 3 embedding services."""

import logging

from app.agents.schema import DocumentInfo, IngestionResult
from app.agents.tracing import agent_span
from app.embeddings.service import EmbeddingService
from app.ingestion import repository as repo
from app.ingestion.service import IngestionService

logger = logging.getLogger(__name__)


class IngestionAgent:
    """Discovers a bid's files, ingests new/changed ones, embeds what is pending, reports status.

    All real work (parsing, chunking, hashing, persistence, embedding) is done by the
    existing services; unchanged files are skipped by their content hash.
    """

    def __init__(self, ingestion: IngestionService, embeddings: EmbeddingService | None = None) -> None:
        self.ingestion = ingestion
        self.embeddings = embeddings

    def _files(self, bid_id: str):
        try:
            return [f for f in self.ingestion.scan().files if f.bid_id == bid_id]
        except FileNotFoundError:
            return []  # no data directory: fall back to whatever is already stored

    async def bid_exists(self, bid_id: str) -> bool:
        if self._files(bid_id):
            return True
        return bool(await repo.list_documents(self.ingestion.session, bid_id))

    async def run(self, bid_id: str, ingest: bool = True) -> IngestionResult:
        async with agent_span("ingestion", bid_id, "ingest") as span:
            result = IngestionResult(bid_id=bid_id)
            files = self._files(bid_id)
            result.files_found = len(files)
            if not files:
                result.notes.append("No source files found on disk for this bid; using stored documents.")

            if ingest:
                for file in files:
                    outcome = await self.ingestion.ingest_file(file)
                    if outcome.status == "processed":
                        result.ingested += 1
                    elif outcome.status == "skipped":
                        result.skipped += 1
                    else:
                        result.failed_files.append(file.file_name)

            documents = await repo.list_documents(self.ingestion.session, bid_id)
            result.documents = [
                DocumentInfo(document_id=d.id, file_name=d.file_name, document_type=d.document_type, status=d.status)
                for d in documents
            ]  # fmt: skip
            result.failed_files = sorted(
                set(result.failed_files) | {d.file_name for d in documents if d.status == "failed"}
            )

            if ingest and self.embeddings is not None:
                counts = await self.embeddings.status()
                if counts.pending + counts.failed > 0:  # only load the model when there is work
                    summary = await self.embeddings.generate()
                    result.chunks_embedded, result.embedding_failures = summary.embedded, summary.failed
            span["note"] = f"ingested={result.ingested} skipped={result.skipped} failed={len(result.failed_files)}"
            return result
