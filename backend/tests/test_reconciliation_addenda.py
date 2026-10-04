import pytest

from app.reconciliation.addenda import (
    build_event,
    build_timeline,
    is_addendum_by_content,
    parse_addendum_date,
    parse_addendum_number,
    parse_referenced_number,
)


@pytest.mark.parametrize(
    ("text", "file_name", "expected"),
    [
        ("ADDENDUM No. 2\nRFP JA-207652", None, (2, "content")),
        ("Addendum 1 to the RFP", None, (1, "content")),
        ("ADDENDUM NO. 3", None, (3, "content")),
        ("ADDENDUM #4", None, (4, "content")),
        ("ADDENDUM NUMBER 5", None, (5, "content")),
        ("Addendum Two", None, (2, "content")),
        ("Some text without a header", "Addendum 7 RFP X.pdf", (7, "filename")),
        ("Some text without a header", "Amendment.pdf", (None, None)),  # never invented
        ("ADDENDUM No. 2", "Addendum 9.pdf", (2, "content")),  # content beats file name
    ],
)
def test_addendum_number(text, file_name, expected) -> None:
    assert parse_addendum_number(text, file_name) == expected


def test_dates_are_read_only_when_written() -> None:
    assert parse_addendum_date("ADDENDUM No. 1\nDated: June 20, 2024\n...") == "June 20, 2024"
    assert parse_addendum_date("Date issued 06/20/2024") == "06/20/2024"
    assert parse_addendum_date("Submitter's Signature:\n\nDate:\n\nEND OF ADDENDUM") is None  # blank form line
    assert parse_addendum_date("The new due date for this RFP will be July 9, 2024") is None  # not an issue date


def test_referenced_number_and_content_detection() -> None:
    assert parse_referenced_number("ADDENDUM No. 2\nRFP JA-207652 Student and Staff") == "JA-207652"
    assert is_addendum_by_content("ADDENDUM No. 2\nRFP JA-207652")
    assert not is_addendum_by_content("Request For Proposal 168884\nThis document has addendum notes later")


def event(doc_id, number=None, source="content", date=None, file="a.pdf"):
    return build_event(
        document_id=doc_id, file_name=file, document_type="addendum", bid_id="Bid1",
        first_page=(f"ADDENDUM No. {number}\n" if number else "") + (f"Dated: {date}\n" if date else ""),
        full_text="The new due date is X. For clarification only.",
    ).model_copy(update={"number_source": source if number else None, "addendum_number": number})  # fmt: skip


def test_event_flags_change_and_clarification_language() -> None:
    e = event(1, 1)
    assert e.changes_language and e.clarifies_language and e.identified_by == "document_type"


def test_timeline_orders_by_explicit_number() -> None:
    t = build_timeline("Bid1", [event(20, 2), event(10, 1)], ["rfp.pdf"])
    assert [e.addendum_number for e in t.addenda] == [1, 2]
    assert [e.order for e in t.addenda] == [1, 2]
    assert (t.ordering, t.reliable, t.original_documents) == ("explicit_number", True, ["rfp.pdf"])


def test_timeline_falls_back_to_dates_when_numbers_missing() -> None:
    t = build_timeline("Bid1", [event(1, date="July 1, 2024"), event(2, date="June 25, 2024")])
    assert [e.document_id for e in t.addenda] == [2, 1]
    assert t.ordering == "date" and t.reliable


def test_filename_only_ordering_is_flagged_unreliable() -> None:
    t = build_timeline("Bid1", [event(2, 2, source="filename"), event(1, 1, source="filename")])
    assert [e.addendum_number for e in t.addenda] == [1, 2]
    assert t.ordering == "filename" and t.reliable is False and t.notes


def test_duplicate_or_missing_numbers_keep_the_uncertainty() -> None:
    t = build_timeline("Bid1", [event(1, 2), event(2, 2), event(3)])
    assert t.ordering == "uncertain" and t.reliable is False
    assert all(e.order is None for e in t.addenda)


def test_dates_contradicting_numbers_are_flagged() -> None:
    t = build_timeline("Bid1", [event(1, 1, date="July 1, 2024"), event(2, 2, date="June 1, 2024")])
    assert t.reliable is False and any("disagree" in n for n in t.notes)


def test_single_and_no_addenda() -> None:
    single = build_timeline("Bid1", [event(1, 3)])
    assert single.addenda[0].order == 1 and single.reliable and single.ordering == "explicit_number"
    empty = build_timeline("Bid1", [])
    assert empty.ordering == "none" and empty.addenda == [] and empty.reliable
