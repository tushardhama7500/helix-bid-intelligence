import pytest

from app.ingestion.discovery import discover_files


def test_discovers_supported_files_for_any_bid_folder(tmp_path) -> None:
    names = ["Bid1/a.pdf", "Bid2/b.HTML", "Bid3/sub/c.htm", "Bid1/notes.txt", "Bid1/.gitkeep", "stray.pdf"]
    for rel in names:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")

    result = discover_files(tmp_path)

    assert [(f.bid_id, f.file_type, f.rel_path) for f in result.files] == [
        ("Bid1", "pdf", "Bid1/a.pdf"),
        ("Bid2", "html", "Bid2/b.HTML"),
        ("Bid3", "html", "Bid3/sub/c.htm"),
    ]
    assert result.ignored == 3


def test_missing_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_files(tmp_path / "nope")
