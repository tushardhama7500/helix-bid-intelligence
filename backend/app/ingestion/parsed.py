"""Plain data structures passed between ingestion stages."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedPage:
    """A PDF page (1-based `page_number`) or an HTML section (`None`)."""

    page_number: int | None
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    title: str | None
    pages: list[ParsedPage]
    # Total pages in the source file (PDF only; includes pages without text).
    page_count: int | None = None
