import pytest

from app.ingestion.chunker import Chunker, ChunkSource
from tests.helpers import lorem


def test_short_text_is_single_chunk() -> None:
    assert Chunker(1000, 150).split("Hello world.") == ["Hello world."]


def test_chunks_respect_size_and_cover_all_text() -> None:
    text = "\n\n".join(lorem(6, f"P{i}_") for i in range(10))
    chunks = Chunker(300, 50).split(text)

    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    joined = " ".join(chunks)
    for i in range(10):
        assert f"P{i}_0 " in joined


def test_consecutive_chunks_overlap() -> None:
    chunks = Chunker(300, 60).split(lorem(40))

    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:]):
        head = " ".join(nxt.split()[:3])
        assert head in prev[-60:], (prev[-60:], head)


def test_prefers_paragraph_boundaries() -> None:
    paras = [lorem(4, f"P{i}_") for i in range(6)]
    chunks = Chunker(400, 0).split("\n\n".join(paras))
    # With no overlap every chunk is made of whole paragraphs.
    for chunk in chunks:
        assert all(part in paras for part in chunk.split("\n\n"))


def test_oversized_word_is_sliced() -> None:
    chunks = Chunker(50, 10).split("x" * 200)
    assert all(len(c) <= 50 for c in chunks)


def test_metadata_is_preserved_and_index_is_document_wide() -> None:
    meta = {"bid_id": "Bid1", "file_name": "A.pdf", "document_type": "addendum"}
    sources = [
        ChunkSource(11, lorem(30), {**meta, "page_number": 1}),
        ChunkSource(12, lorem(30), {**meta, "page_number": 2}),
    ]
    records = Chunker(300, 50).chunk_pages(sources)

    assert [r.chunk_index for r in records] == list(range(len(records)))
    assert {r.page_id for r in records} == {11, 12}
    for r in records:
        assert r.metadata["bid_id"] == "Bid1"
        assert r.metadata["file_name"] == "A.pdf"
        assert r.metadata["chunk_index"] == r.chunk_index
        assert r.metadata["page_number"] == (1 if r.page_id == 11 else 2)


def test_invalid_config() -> None:
    with pytest.raises(ValueError):
        Chunker(100, 100)
