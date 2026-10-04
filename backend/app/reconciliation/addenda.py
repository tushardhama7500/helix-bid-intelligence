"""Identify addenda and order them into a per-bid timeline."""

import re
from collections.abc import Sequence
from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentPage
from app.reconciliation.detect import has_change_language, has_clarification_language
from app.reconciliation.schema import AddendumEvent, Timeline

_WORD_NUMBERS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
_NUM = r"(?P<n>\d{1,3}|" + "|".join(w for w in _WORD_NUMBERS if w != "zero") + r")"
_HEADER = re.compile(rf"^\W*addend(?:um|a)\s*(?:no\.?|number|#)?\s*[:\-]?\s*{_NUM}\b", re.I)
_IN_TEXT = re.compile(rf"\baddend(?:um|a)\s*(?:no\.?|number|#)?\s*[:\-]?\s*{_NUM}\b", re.I)
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_DATE_TOKEN = rf"(?:{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|\d{{1,2}}-[A-Za-z]{{3}}-\d{{4}})"
_DATE = re.compile(
    rf"(?:\b(?:dated|date\s+issued|issue\s+date|issued(?:\s+on)?|addendum\s+date)|^\s*date)\s*[:\-]?[ \t]*(?P<d>{_DATE_TOKEN})",
    re.I | re.M,
)
_REF_NUMBER = re.compile(r"\b(?:RFP|IFB|RFQ|PORFP|Solicitation|Bid)\s*(?:No\.?|Number|#)?\s*:?\s*(?P<n>[A-Z]{0,4}-?\d[\w\-]{3,})", re.I)
_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y")


def _to_int(token: str) -> int:
    return int(token) if token.isdigit() else _WORD_NUMBERS[token.lower()]


def parse_addendum_number(text: str, file_name: str | None = None) -> tuple[int | None, str | None]:
    """(number, source): read from the document's own header/text first, the file name only as a fallback."""
    head = text[:500]
    m = _HEADER.search(head) or _IN_TEXT.search(head)
    if m:
        return _to_int(m.group("n")), "content"
    if file_name:
        m = _IN_TEXT.search(file_name.replace("_", " "))
        if m:
            return _to_int(m.group("n")), "filename"
    return None, None


def parse_addendum_date(text: str) -> str | None:
    """A date written next to an issue/date label; blank 'Date:' signature lines yield nothing."""
    m = _DATE.search(text[:3000])
    return m.group("d").strip() if m else None


def parse_date_value(value: str | None) -> date | None:
    if not value:
        return None
    cleaned = value.replace("Sept", "Sep").replace(".", "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def parse_referenced_number(text: str) -> str | None:
    m = _REF_NUMBER.search(text[:600])
    return m.group("n") if m else None


def is_addendum_by_content(first_page: str) -> bool:
    """The page starts with an 'ADDENDUM No. N' header."""
    return bool(_HEADER.search(first_page[:200]))


def build_event(
    *, document_id: int, file_name: str, document_type: str, bid_id: str, first_page: str, full_text: str
) -> AddendumEvent:
    number, source = parse_addendum_number(first_page, file_name)
    return AddendumEvent(
        document_id=document_id,
        file_name=file_name,
        document_type=document_type,
        bid_id=bid_id,
        addendum_number=number,
        number_source=source,
        date=parse_addendum_date(first_page),
        referenced_bid_number=parse_referenced_number(first_page),
        identified_by="document_type" if document_type == "addendum" else "content",
        changes_language=has_change_language(full_text),
        clarifies_language=has_clarification_language(full_text),
    )


def build_timeline(bid_id: str, events: Sequence[AddendumEvent], original_documents: Sequence[str] = ()) -> Timeline:
    """Order addenda by explicit number, then date, then file-name number, else leave unordered.

    Uncertainty is kept, not hidden: `reliable` is False and `order` stays None for
    addenda whose position cannot be established.
    """
    events = [e.model_copy() for e in events]
    notes: list[str] = []
    timeline = Timeline(bid_id=bid_id, original_documents=list(original_documents), addenda=events)
    if not events:
        timeline.ordering, timeline.reliable = "none", True
        return timeline

    if len(events) == 1:
        e = events[0]
        e.order = 1
        timeline.ordering = {"content": "explicit_number", "filename": "filename"}.get(e.number_source or "", "none")
        return timeline

    numbers = [e.addendum_number for e in events]
    dates = [parse_date_value(e.date) for e in events]
    unique_numbers = None not in numbers and len(set(numbers)) == len(numbers)
    all_content = all(e.number_source == "content" for e in events)
    unique_dates = None not in dates and len(set(dates)) == len(dates)

    if unique_numbers and all_content:
        events.sort(key=lambda e: e.addendum_number)  # type: ignore[arg-type,return-value]
        ordering, reliable = "explicit_number", True
        dated = [parse_date_value(e.date) for e in events if e.date]
        if len(dated) == len(events) and dated != sorted(dated):
            notes.append("Addendum dates disagree with addendum numbers; order is not reliable.")
            reliable = False
    elif unique_dates:
        events.sort(key=lambda e: parse_date_value(e.date))  # type: ignore[arg-type,return-value]
        ordering, reliable = "date", True
    elif unique_numbers:
        events.sort(key=lambda e: e.addendum_number)  # type: ignore[arg-type,return-value]
        ordering, reliable = "filename", False
        notes.append("Addendum order was taken from file names only; it could not be confirmed from content.")
    else:
        events.sort(key=lambda e: (e.addendum_number is None, e.addendum_number or 0, e.document_id))
        ordering, reliable = "uncertain", False
        notes.append("Some addenda have no usable number or date, or numbers repeat; their order is uncertain.")

    for position, e in enumerate(events, start=1):
        e.order = position if (reliable or ordering == "filename") else None
    if ordering == "uncertain":
        for e in events:
            e.order = None
    timeline.addenda = events
    timeline.ordering, timeline.reliable, timeline.notes = ordering, reliable, notes  # type: ignore[assignment]
    return timeline


async def load_timeline(session: AsyncSession, bid_id: str) -> Timeline:
    """Build the bid's timeline from stored documents (addenda found by type or by their header)."""
    documents = (await session.execute(select(Document).where(Document.bid_id == bid_id).order_by(Document.id))).scalars().all()
    pages = (
        await session.execute(
            select(DocumentPage.document_id, DocumentPage.content)
            .where(DocumentPage.document_id.in_([d.id for d in documents]))
            .order_by(DocumentPage.document_id, DocumentPage.page_number.nulls_first(), DocumentPage.id)
        )
    ).all()
    by_doc: dict[int, list[str]] = {}
    for doc_id, content in pages:
        by_doc.setdefault(doc_id, []).append(content)

    events: list[AddendumEvent] = []
    originals: list[str] = []
    for d in documents:
        texts = by_doc.get(d.id, [])
        first = texts[0] if texts else ""
        if d.document_type == "addendum" or is_addendum_by_content(first):
            events.append(
                build_event(
                    document_id=d.id, file_name=d.file_name, document_type=d.document_type, bid_id=bid_id,
                    first_page=first, full_text="\n".join(texts),
                )  # fmt: skip
            )
        else:
            originals.append(d.file_name)
    return build_timeline(bid_id, events, originals)
