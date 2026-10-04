import pytest

from app.ingestion.metadata import detect_document_type


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf", "addendum"),
        ("ADDENDA_1.pdf", "addendum"),
        ("Non-Collusion Affidavit.pdf", "affidavit"),
        ("Technical Specifications.pdf", "specification"),
        ("device_spec.html", "specification"),
        ("RFP JA-207652 Student and Staff Computing Devices.pdf", "rfp"),
        ("Special Terms.pdf", "bid_document"),
        ("notes.html", "bid_document"),
    ],
)
def test_detect_document_type(name: str, expected: str) -> None:
    assert detect_document_type(name) == expected
