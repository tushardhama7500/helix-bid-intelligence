"""Query validation and keyword extraction."""

import re
from dataclasses import dataclass

from app.retrieval.schema import InvalidQueryError

# Words, but keep identifiers intact: JA-207652, WD22TB4, BPM044557, 5550, 3.1, a/b.
_TOKEN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_\-./]*[A-Za-z0-9])?")
_MAX_TERMS = 32


@dataclass(frozen=True)
class ProcessedQuery:
    text: str  # whitespace-normalised original, used for the embedding
    terms: tuple[str, ...]  # distinct keyword terms, in order of appearance


def preprocess_query(raw: str, max_length: int) -> ProcessedQuery:
    """Validate and normalise a user query.

    Stop-word removal and stemming are left to PostgreSQL's `english` text
    search configuration, so the same rules apply at index and query time.
    """
    text = " ".join(raw.split())
    if not text:
        raise InvalidQueryError("Query must not be empty")
    if len(text) > max_length:
        raise InvalidQueryError(f"Query is too long ({len(text)} > {max_length} characters)")

    seen: dict[str, None] = {}
    for token in _TOKEN.findall(text):
        seen.setdefault(token.lower(), None)
    return ProcessedQuery(text=text, terms=tuple(seen)[:_MAX_TERMS])
