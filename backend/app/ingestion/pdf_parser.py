"""PDF -> raw per-page text using PyMuPDF."""

import logging
from pathlib import Path

import pymupdf

from app.ingestion.parsed import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

_TEXT_BLOCK = 0
MAX_PDF_PAGES = 2000  # far above a bid document; guards against page-count bombs


def parse_pdf(path: Path) -> ParsedDocument:
    """Extract text from every page. Pages without text keep empty content.

    Page numbers are 1-based. Cleanup is left to the normalizer.
    """
    logger.info("Processing PDF %s", path.name)
    # Open from bytes so no file handle outlives a failed parse (matters on Windows).
    with pymupdf.open(stream=path.read_bytes(), filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise ValueError("PDF is password protected")
        if pdf.page_count > MAX_PDF_PAGES:
            raise ValueError(f"PDF has too many pages ({pdf.page_count} > {MAX_PDF_PAGES})")

        pages = []
        for number, page in enumerate(pdf, start=1):
            blocks = page.get_text("blocks", sort=True)
            text = "\n\n".join(b[4] for b in blocks if b[6] == _TEXT_BLOCK)
            pages.append(ParsedPage(page_number=number, content=text))

        title = (pdf.metadata or {}).get("title") or None
        page_count = pdf.page_count

    logger.info("Extracted %d pages from %s", len(pages), path.name)
    return ParsedDocument(title=title, pages=pages, page_count=page_count)
