"""Test doubles for the embedding model and helpers to seed chunks."""

import hashlib
import math

from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.model import EmbeddingModelError
from app.models import Document, DocumentChunk, DocumentPage

DIM = 384  # matches BAAI/bge-small-en-v1.5, the default model


class FakeEmbeddingModel:
    """Deterministic stand-in with the EmbeddingModel interface (no download)."""

    def __init__(self, name="BAAI/bge-small-en-v1.5", dimension=DIM, fail_on=(), load_error=None):
        self.name = name
        self.dimension = dimension
        self.batch_size = 32
        self.fail_on = set(fail_on)
        self.load_error = load_error
        self.embedded: list[str] = []  # every text that was actually embedded
        self.calls = 0

    def load(self) -> None:
        if self.load_error:
            raise EmbeddingModelError(self.load_error)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if any(t in self.fail_on for t in texts):
            raise RuntimeError("simulated embedding failure")
        self.embedded.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]

    def _vector(self, text: str) -> list[float]:
        raw = [b / 255 - 0.5 for b in hashlib.sha256(text.encode()).digest()]
        vec = [raw[i % len(raw)] + i * 1e-4 for i in range(self.dimension)]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


async def seed_chunks(session: AsyncSession, contents: list[str], bid_id: str = "Bid1") -> list[int]:
    """Insert one document/page and one chunk per string; returns chunk ids."""
    doc = Document(
        bid_id=bid_id, file_name="a.pdf", file_path=f"{bid_id}/a.pdf", file_type="pdf",
        document_type="rfp", content_hash="0" * 64, status="completed",
    )  # fmt: skip
    session.add(doc)
    await session.flush()
    page = DocumentPage(document_id=doc.id, page_number=1, content="\n".join(contents), meta={})
    session.add(page)
    await session.flush()
    chunks = [
        DocumentChunk(
            document_id=doc.id, page_id=page.id, bid_id=bid_id, chunk_index=i,
            content=text, meta={"bid_id": bid_id, "page_number": 1, "chunk_index": i},
        )
        for i, text in enumerate(contents)
    ]  # fmt: skip
    session.add_all(chunks)
    await session.commit()
    return [c.id for c in chunks]


class TopicEmbeddingModel(FakeEmbeddingModel):
    """Deterministic model with real 'semantics': one dimension per topic word."""

    TOPICS = ("deadline", "due", "warranty", "laptop", "dock", "mercury", "closing")

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        vec = [float(lowered.count(t)) for t in self.TOPICS] + [0.0] * (self.dimension - len(self.TOPICS))
        vec[-1] = 0.05  # keeps the vector non-zero so cosine distance is defined
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


# (bid_id, file_name, document_type, page_number, section, content)
CORPUS = [
    ("Bid1", "Addendum 2 RFP JA-207652.pdf", "addendum", 1, None,
     "The new due date for this RFP will be July 9, 2024. The deadline is extended."),
    ("Bid1", "JA-207652 FINAL.pdf", "rfp", 2, None,
     "Solicitation Due 27-JUN-2024. Proposal deadline for JA-207652 devices."),
    ("Bid2", "Dell_Laptop_Specs.pdf", "specification", 1, None,
     "SI# CC7802 Dell Latitude 5550 laptop with Thunderbolt dock WD22TB4."),
    ("Bid2", "PORFP_Dell_Laptop.pdf", "rfp", 3, None,
     "Extended warranty for all machines purchased - 3 Years. Mercury Affidavit required."),
    ("Bid2", "Dell Laptops - BidNet Direct.html", "bid_document", None, "Bid Information",
     "BPM044557 Dell Laptops closing date 07/09/2024."),
]  # fmt: skip


async def seed_corpus(
    session: AsyncSession, model: FakeEmbeddingModel, corpus: list | None = None
) -> dict[str, int]:
    """Insert `corpus` (default CORPUS) as embedded documents/pages/chunks; returns file_name -> chunk id."""
    ids: dict[str, int] = {}
    for index, (bid, file_name, doc_type, page_number, section, content) in enumerate(corpus or CORPUS):
        doc = Document(
            bid_id=bid, file_name=file_name, file_path=f"{bid}/{file_name}",
            file_type="html" if file_name.endswith(".html") else "pdf",
            document_type=doc_type, content_hash=f"{index:064d}", status="completed",
        )  # fmt: skip
        session.add(doc)
        await session.flush()
        meta = {"bid_id": bid, "page_number": page_number, **({"section": section} if section else {})}
        page = DocumentPage(document_id=doc.id, page_number=page_number, content=content, meta=meta)
        session.add(page)
        await session.flush()
        chunk = DocumentChunk(
            document_id=doc.id, page_id=page.id, bid_id=bid, chunk_index=index, content=content,
            meta={**meta, "chunk_index": index}, embedding=model.embed([content])[0],
            embedding_model=model.name,
        )  # fmt: skip
        session.add(chunk)
        await session.flush()
        ids[file_name] = chunk.id
    await session.commit()
    return ids
