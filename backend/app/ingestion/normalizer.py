"""Text cleanup applied to every parsed page before chunking."""

import logging
import re
import unicodedata
from collections.abc import Iterable

from app.ingestion.parsed import ParsedPage

logger = logging.getLogger(__name__)

# Control characters (Postgres also rejects NUL), zero-width and soft hyphens.
_INVISIBLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​‌‍﻿\xad]")
_HYPHEN_WRAP = re.compile(r"(\w)-\n([a-z])")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = _INVISIBLE.sub("", text)
    text = _HYPHEN_WRAP.sub(r"\1\2", text)  # "procure-\nment" -> "procurement"
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def normalize_pages(pages: Iterable[ParsedPage], source: str) -> list[ParsedPage]:
    """Normalize page text and drop pages that end up empty (with a warning)."""
    kept: list[ParsedPage] = []
    for page in pages:
        content = normalize_text(page.content)
        if not content:
            where = f"page {page.page_number}" if page.page_number else "section"
            logger.warning("Empty page: %s of %s has no extractable text", where, source)
            continue
        kept.append(ParsedPage(page.page_number, content, page.metadata))
    return kept
