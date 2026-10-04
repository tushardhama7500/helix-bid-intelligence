"""Deterministic, boundary-aware text chunking.

Text is split into paragraphs, then sentences (only for paragraphs too big to
fit), then words (only for oversized sentences). Those units are packed
greedily up to `chunk_size`. Each new chunk starts with the last
`chunk_overlap` characters of the previous one, snapped to a word boundary.
Chunks never span pages, so every chunk keeps exactly one page's metadata.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

PARA_SEP = "\n\n"
SENT_SEP = " "


@dataclass
class ChunkSource:
    """One page's worth of text plus the metadata its chunks inherit."""

    page_id: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    chunk_index: int  # 0-based, continuous across the whole document
    page_id: int
    content: str
    metadata: dict[str, Any]


def _hard_split(text: str, limit: int) -> list[str]:
    """Split on whitespace to at most `limit` chars; slice words that are longer."""
    parts: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:limit])
            word = word[limit:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= limit:
            current = candidate
        else:
            parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts


def _units(text: str, limit: int) -> list[tuple[str, str]]:
    """Break text into (separator_before, unit) pairs, each unit <= `limit`."""
    units: list[tuple[str, str]] = []
    for paragraph in _PARAGRAPH.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            units.append((PARA_SEP, paragraph))
            continue
        first = True
        for sentence in _SENTENCE_END.split(paragraph):
            for piece in _hard_split(sentence, limit):
                units.append((PARA_SEP if first else SENT_SEP, piece))
                first = False
    return units


def _overlap_tail(chunk: str, overlap: int) -> str:
    if overlap <= 0 or len(chunk) <= overlap:
        return ""
    tail = chunk[-overlap:]
    if not chunk[-overlap - 1].isspace():  # cut landed mid-word: drop the fragment
        tail = "".join(re.split(r"\s", tail, maxsplit=1)[1:])
    return tail.strip()


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Need chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        # Reserve room so overlap + separator + unit always fits in chunk_size.
        limit = max(self.chunk_size - self.chunk_overlap - len(PARA_SEP), 1)
        chunks: list[str] = []
        current = ""
        for sep, unit in _units(text, limit):
            if not current:
                current = unit
            elif len(current) + len(sep) + len(unit) <= self.chunk_size:
                current += sep + unit
            else:
                chunks.append(current)
                tail = _overlap_tail(current, self.chunk_overlap)
                current = f"{tail}{sep}{unit}" if tail else unit
        if current:
            chunks.append(current)
        return chunks

    def chunk_pages(self, sources: Sequence[ChunkSource]) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        for source in sources:
            for content in self.split(source.content):
                index = len(records)
                records.append(
                    ChunkRecord(
                        chunk_index=index,
                        page_id=source.page_id,
                        content=content,
                        metadata={**source.metadata, "chunk_index": index},
                    )
                )
        return records
