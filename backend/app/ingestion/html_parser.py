"""HTML -> raw text sections using BeautifulSoup.

HTML has no pages, so the document is split into sections at <h1>/<h2>
headings. Each section becomes a `ParsedPage` with `page_number=None` and the
heading stored as `metadata["section"]`.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import PreformattedString

from app.ingestion.parsed import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

# Elements that never carry document content. <form> is deliberately absent:
# ASP.NET-style pages (e.g. BidNet) wrap the entire page body in one.
_NOISE_TAGS = [
    "script", "style", "noscript", "template", "iframe", "svg", "canvas",
    "nav", "footer", "aside", "button", "select", "head",
]  # fmt: skip
_NOISE_ATTR = re.compile(r"cookie|consent|breadcrumb", re.I)
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)

_HEADINGS = {f"h{n}" for n in range(1, 7)}
_BLOCKS = {
    "html", "body", "p", "div", "section", "article", "main", "ul", "ol",
    "table", "thead", "tbody", "tfoot", "blockquote", "dl", "dt", "dd",
    "address", "figure", "figcaption", "fieldset", "details", "summary", "hr",
}  # fmt: skip


@dataclass
class _Block:
    kind: str  # "heading" | "text"
    text: str
    level: int = 0


class _Collector:
    """Accumulates inline text and flushes it as blocks at block boundaries."""

    def __init__(self) -> None:
        self.blocks: list[_Block] = []
        self._buffer: list[str] = []

    def add_text(self, text: str) -> None:
        self._buffer.append(text)

    def flush(self) -> None:
        text = " ".join("".join(self._buffer).split())
        self._buffer.clear()
        if text:
            self.blocks.append(_Block("text", text))

    def add(self, block: _Block) -> None:
        self.flush()
        if block.text:
            self.blocks.append(block)


def _flat(tag: Tag) -> str:
    return " ".join(tag.get_text(" ").split())


def _walk(node: Tag, out: _Collector) -> None:
    for child in node.children:
        if isinstance(child, PreformattedString):  # comments, doctype, CDATA
            continue
        if isinstance(child, NavigableString):
            out.add_text(str(child))
            continue

        name = child.name
        if name in _HEADINGS:
            out.add(_Block("heading", _flat(child), int(name[1])))
        elif name == "br":
            out.flush()
        elif name == "tr":
            cells = (_flat(c) for c in child.find_all(["td", "th"]))
            out.add(_Block("text", " | ".join(c for c in cells if c)))
        elif name == "pre":
            out.add(_Block("text", child.get_text().strip()))
        elif name == "li":
            out.flush()
            start = len(out.blocks)
            _walk(child, out)
            out.flush()
            if start < len(out.blocks) and out.blocks[start].kind == "text":
                out.blocks[start].text = "- " + out.blocks[start].text
        elif name in _BLOCKS:
            out.flush()
            _walk(child, out)
            out.flush()
        else:  # inline or unknown element: keep flowing into the same block
            _walk(child, out)


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.decomposed or tag.name in ("html", "body"):
            continue
        attrs = tag.attrs
        hidden = (
            "hidden" in attrs
            or attrs.get("aria-hidden") == "true"
            or attrs.get("role") == "navigation"
            or _HIDDEN_STYLE.search(attrs.get("style") or "")
            or _NOISE_ATTR.search(" ".join(_as_list(attrs.get("class"))) + " " + str(attrs.get("id") or ""))
        )
        if hidden:
            tag.decompose()


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [str(value)]


def _to_sections(blocks: list[_Block]) -> list[ParsedPage]:
    sections: list[ParsedPage] = []
    heading: str | None = None
    lines: list[str] = []
    has_body = False

    def close() -> None:
        if lines:
            sections.append(
                ParsedPage(None, "\n\n".join(lines), {"section": heading})
            )

    for block in blocks:
        if block.kind == "heading":
            line = f"{'#' * block.level} {block.text}"
            if block.level <= 2:
                if has_body:  # start a new section; a bare heading just extends
                    close()
                    lines = []
                    has_body = False
                heading = block.text
            lines.append(line)
        else:
            lines.append(block.text)
            has_body = True
    close()
    return sections


def parse_html(path: Path) -> ParsedDocument:
    logger.info("Processing HTML %s", path.name)
    soup = BeautifulSoup(path.read_bytes(), "html.parser")

    title = None
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split()) or None

    _strip_noise(soup)
    collector = _Collector()
    _walk(soup, collector)
    collector.flush()

    sections = _to_sections(collector.blocks)
    if title is None:
        first_h1 = next(
            (b.text for b in collector.blocks if b.kind == "heading" and b.level == 1),
            None,
        )
        title = first_h1

    logger.info("Extracted %d sections from %s", len(sections), path.name)
    return ParsedDocument(title=title, pages=sections, page_count=None)
