import pytest

from app.ingestion.pdf_parser import parse_pdf
from tests.helpers import make_pdf


def test_parses_pages_with_one_based_numbers(tmp_path) -> None:
    path = make_pdf(tmp_path / "a.pdf", ["First page text.", "Second page text."])
    doc = parse_pdf(path)

    assert doc.page_count == 2
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert "First page text." in doc.pages[0].content
    assert "Second page text." in doc.pages[1].content


def test_empty_page_does_not_crash(tmp_path) -> None:
    path = make_pdf(tmp_path / "a.pdf", ["Has text.", "", "More text."])
    doc = parse_pdf(path)

    assert doc.page_count == 3
    assert doc.pages[1].page_number == 2
    assert doc.pages[1].content.strip() == ""


def test_corrupt_pdf_raises(tmp_path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(Exception):
        parse_pdf(bad)
