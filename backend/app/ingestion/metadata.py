"""Rule-based document typing and citation metadata."""

import re
from typing import Any

from app.ingestion.discovery import DiscoveredFile
from app.ingestion.parsed import ParsedPage

_SEPARATORS = re.compile(r"[_\-.()]+")

# First match wins, so "Addendum 2 RFP ..." is an addendum, not an RFP.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("addendum", re.compile(r"\baddend(um|a)\b")),
    ("affidavit", re.compile(r"\baffidavits?\b")),
    ("specification", re.compile(r"\bspecs?\b|\bspecifications?\b")),
    ("rfp", re.compile(r"\brf[pqi]\b")),
]


DOCUMENT_TYPES = tuple(name for name, _ in _RULES) + ("bid_document",)


def detect_document_type(file_name: str) -> str:
    """Classify by file name only: addendum / affidavit / specification / rfp."""
    name = _SEPARATORS.sub(" ", file_name.lower())
    for doc_type, pattern in _RULES:
        if pattern.search(name):
            return doc_type
    return "bid_document"


def build_page_metadata(
    file: DiscoveredFile, document_type: str, page: ParsedPage
) -> dict[str, Any]:
    """Citation metadata stored on the page and inherited by its chunks."""
    return {
        "bid_id": file.bid_id,
        "file_name": file.file_name,
        "file_path": file.rel_path,
        "file_type": file.file_type,
        "document_type": document_type,
        "page_number": page.page_number,
        **page.metadata,  # e.g. HTML "section"
    }
