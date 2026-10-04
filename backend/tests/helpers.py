"""Builders for small test documents."""

from pathlib import Path

import pymupdf


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a PDF with one page per string; an empty string gives a blank page."""
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(72, 72, 520, 760), text, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # PyMuPDF cannot overwrite an existing file on Windows
    pdf.save(path)
    pdf.close()
    return path


def lorem(sentences: int, tag: str = "S") -> str:
    return " ".join(
        f"{tag}{i} is a sentence about procurement of student devices."
        for i in range(sentences)
    )
