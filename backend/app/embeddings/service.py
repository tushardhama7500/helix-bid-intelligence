"""Generate embeddings for chunks that do not have one yet."""

import asyncio
import logging
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.embeddings import repository as repo
from app.embeddings.model import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)

_MAX_REPORTED_ERRORS = 20


@dataclass
class EmbeddingSummary:
    total_chunks: int = 0
    embedded: int = 0
    skipped: int = 0  # already embedded, left untouched
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def _error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


class EmbeddingService:
    def __init__(self, session: AsyncSession, model: EmbeddingModel, batch_size: int) -> None:
        self.session = session
        self.model = model
        self.batch_size = batch_size

    @classmethod
    def from_settings(cls, session: AsyncSession, settings: Settings) -> "EmbeddingService":
        model = get_embedding_model(settings.embedding_model, settings.embedding_batch_size)
        return cls(session, model, settings.embedding_batch_size)

    async def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        (vector,) = await asyncio.to_thread(self.model.embed, [text])
        return vector

    async def embed_query(self, query: str) -> list[float]:
        """Embed a search query with the same model as the documents (plus its query prefix)."""
        if not query.strip():
            raise ValueError("Cannot embed empty query")
        return await asyncio.to_thread(self.model.embed_query, query)

    async def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        """Batch-embed; blank texts get `None` instead of a meaningless vector."""
        positions = [i for i, t in enumerate(texts) if t.strip()]
        results: list[list[float] | None] = [None] * len(texts)
        if positions:
            vectors = await asyncio.to_thread(self.model.embed, [texts[i] for i in positions])
            for i, vector in zip(positions, vectors, strict=True):
                results[i] = vector
        return results

    async def status(self) -> repo.ChunkCounts:
        return await repo.get_counts(self.session)

    async def generate(self) -> EmbeddingSummary:
        """Embed every chunk whose embedding is NULL, committing per batch.

        Raises EmbeddingModelError if the model cannot load, and lets database
        errors propagate; batches committed before that stay committed.
        """
        await asyncio.to_thread(self.model.load)

        counts = await repo.get_counts(self.session)
        summary = EmbeddingSummary(
            total_chunks=counts.total, skipped=counts.embedded
        )
        logger.info(
            "Embedding %d chunks (%d already embedded) in batches of %d",
            counts.total - counts.embedded, counts.embedded, self.batch_size,
        )  # fmt: skip

        last_id = 0
        while batch := await repo.next_batch(self.session, last_id, self.batch_size):
            last_id = batch[-1][0]
            try:
                stored, failures = await self._embed_batch(batch)
                await repo.store_embeddings(self.session, self.model.name, stored)
                await repo.store_failures(self.session, failures)
                await self.session.commit()
            except Exception:
                await self.session.rollback()
                logger.exception("Database error while storing embeddings")
                raise

            summary.embedded += len(stored)
            summary.failed += len(failures)
            summary.errors.extend(
                f"chunk {cid}: {msg}" for cid, msg in failures
            )
            summary.errors = summary.errors[:_MAX_REPORTED_ERRORS]
            logger.info("Embedded batch: %d ok, %d failed", len(stored), len(failures))

        logger.info(
            "Embedding finished: %d embedded, %d skipped, %d failed",
            summary.embedded, summary.skipped, summary.failed,
        )  # fmt: skip
        return summary

    async def _embed_batch(
        self, batch: list[tuple[int, str]]
    ) -> tuple[list[tuple[int, list[float]]], list[tuple[int, str]]]:
        """Embed a batch; if it fails as a whole, retry chunk by chunk to isolate the bad ones."""
        try:
            vectors = await self.embed_texts([content for _, content in batch])
            pairs = list(zip((cid for cid, _ in batch), vectors, strict=True))
        except Exception:
            logger.exception("Batch embedding failed; retrying chunks individually")
            pairs = []
            for chunk_id, content in batch:
                try:
                    (vector,) = await self.embed_texts([content])
                except Exception as exc:
                    logger.error("Embedding failed for chunk %d: %s", chunk_id, exc)
                    pairs.append((chunk_id, _error_message(exc)))
                else:
                    pairs.append((chunk_id, vector))

        stored: list[tuple[int, list[float]]] = []
        failures: list[tuple[int, str]] = []
        for chunk_id, value in pairs:
            if isinstance(value, list):
                stored.append((chunk_id, value))
            elif isinstance(value, str):
                failures.append((chunk_id, value))
            else:  # None: blank content
                failures.append((chunk_id, "Empty chunk content"))
        return stored, failures
