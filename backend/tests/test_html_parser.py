from app.ingestion.html_parser import parse_html

HTML = """<!doctype html><html><head><title> Bid  Notice </title>
<style>.x{color:red}</style><script>track('SECRET_SCRIPT')</script></head>
<body>
<nav><a href="/">NAVLINK</a></nav>
<div class="cookie-banner">COOKIE TEXT</div>
<h1>Bid Notice</h1>
<p>Intro <b>paragraph</b> with <i>inline</i> markup.</p>
<h2>Requirements</h2>
<ul><li>Laptops</li><li>Tablets</li></ul>
<table><tr><th>Item</th><th>Qty</th></tr><tr><td>Laptop</td><td>500</td></tr></table>
<!-- a comment -->
<noscript>NOSCRIPT TEXT</noscript>
<footer>FOOTER TEXT</footer>
</body></html>"""


def _parse(tmp_path, html=HTML):
    path = tmp_path / "notice.html"
    path.write_text(html, encoding="utf-8")
    return parse_html(path)


def test_extracts_visible_text_and_title(tmp_path) -> None:
    doc = _parse(tmp_path)
    text = "\n".join(p.content for p in doc.pages)

    assert doc.title == "Bid Notice"
    assert "Intro paragraph with inline markup." in text
    assert "- Laptops" in text and "- Tablets" in text
    assert "Laptop | 500" in text


def test_removes_noise(tmp_path) -> None:
    text = "\n".join(p.content for p in _parse(tmp_path).pages)
    for noise in ("SECRET_SCRIPT", "color:red", "NAVLINK", "COOKIE", "NOSCRIPT", "FOOTER", "comment"):
        assert noise not in text


def test_no_page_numbers_but_sections(tmp_path) -> None:
    doc = _parse(tmp_path)

    assert doc.page_count is None
    assert all(p.page_number is None for p in doc.pages)
    assert [p.metadata["section"] for p in doc.pages] == ["Bid Notice", "Requirements"]


def test_plain_fragment_without_headings(tmp_path) -> None:
    doc = _parse(tmp_path, "<p>Just one paragraph.</p>")
    assert len(doc.pages) == 1
    assert doc.pages[0].content == "Just one paragraph."
    assert doc.pages[0].metadata["section"] is None
