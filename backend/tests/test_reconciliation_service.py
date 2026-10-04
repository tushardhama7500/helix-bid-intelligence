import json
import re

import pytest

from app.core.config import Settings
from app.extraction.service import LLMProviderError, UnknownBidError
from app.reconciliation.addenda import build_event, build_timeline
from app.reconciliation.service import ReconciliationService
from app.retrieval.schema import Candidate, FusedResult
from tests.extraction_fakes import FakeLLM, Found

SETTINGS = Settings(_env_file=None, database_url="postgresql+psycopg://x/x")


def chunk(chunk_id, doc_id, bid, file, text, doc_type="bid_document", page=1, index=0) -> Candidate:
    return Candidate(
        chunk_id=chunk_id, content=text, document_id=doc_id, bid_id=bid, file_name=file, document_type=doc_type,
        page_number=page, chunk_index=index, metadata={}, score=0.1,
    )  # fmt: skip


RFP = chunk(1, 1, "Bid1", "JA-207652 FINAL.pdf", "Solicitation Due 27-JUN-2024 14:00:00 Pre-Proposal Meeting 10-JUN-2024 14:00:00", page=2)
ADD1 = chunk(11, 11, "Bid1", "Addendum 1.pdf", "Answer: Dallas ISD does not anticipate extending the submission deadline.", "addendum")
ADD2 = chunk(12, 12, "Bid1", "Addendum 2.pdf", "ADDENDUM No. 2 The new due date for this RFP will be July 9, 2024 at 2:00 PM CST.", "addendum")
OTHER_BID = [
    chunk(21, 21, "Bid2", "PORFP.pdf", "Solicitation Due 01-MAY-2024 10:00:00"),
    chunk(22, 22, "Bid2", "Addendum 1 PORFP.pdf", "The new due date for this RFP will be August 1, 2024 at 9:00 AM EST.", "addendum"),
]  # fmt: skip


class Retrieval:
    """Honours bid / document_type / document_id filters like the real service."""

    def __init__(self, chunks, error=None):
        self.chunks, self.error, self.calls = chunks, error, []

    async def available_filters(self):
        return {"bids": sorted({c.bid_id for c in self.chunks}), "document_types": [], "file_names": []}

    def matching(self, bid_id, document_type, document_id):
        return [
            c for c in self.chunks
            if c.bid_id == bid_id and document_type in (None, c.document_type) and document_id in (None, c.document_id)
        ]  # fmt: skip

    async def retrieve(self, query, bid_id=None, document_type=None, document_id=None, top_k=5, **_):
        if self.error:
            raise self.error
        self.calls.append({"bid_id": bid_id, "document_id": document_id})
        found = self.matching(bid_id, document_type, document_id)[:top_k]
        return Found([FusedResult(candidate=c, rrf_score=0.01) for c in found])


class LeakyRetrieval(Retrieval):
    """A buggy retrieval that ignores the bid filter: the service must still not use other bids' chunks."""

    def matching(self, bid_id, document_type, document_id):
        return [c for c in self.chunks if document_type in (None, c.document_type)]


def loader(bid_addenda):
    async def load(bid_id):
        events = [
            build_event(document_id=c.document_id, file_name=c.file_name, document_type="addendum", bid_id=bid_id,
                        first_page=c.content, full_text=c.content)
            for c in bid_addenda if c.bid_id == bid_id
        ]  # fmt: skip
        return build_timeline(bid_id, events, ["JA-207652 FINAL.pdf"])

    return load


def make(chunks, llm=None, retrieval_cls=Retrieval, **kw):
    retrieval = retrieval_cls(chunks, kw.pop("error", None))
    addenda = [c for c in chunks if c.document_type == "addendum"]

    async def no_sleep(_):
        return None

    return ReconciliationService(retrieval, SETTINGS, llm, loader(addenda), sleep=no_sleep), retrieval


def by_key(result, key):
    return next(f for f in result.fields if f.key == key)


# ------------------------------------------------------------ deterministic path
async def test_explicit_replacement_resolves_without_groq() -> None:
    llm = FakeLLM()
    service, _ = make([RFP, ADD1, ADD2], llm)

    result = await service.reconcile_bid("Bid1")

    due = by_key(result, "due_date")
    assert due.original_value == "27-JUN-2024 14:00:00"
    assert due.final_value == "July 9, 2024 at 2:00 PM CST"
    assert (due.status, due.resolution_method) == ("resolved", "explicit_addendum_change")
    assert [(c.addendum_number, c.change_type, c.previous_value, c.new_value) for c in due.changes] == [
        (2, "replacement", "27-JUN-2024 14:00:00", "July 9, 2024 at 2:00 PM CST"),
    ]
    assert {s.file_name for s in due.original_sources} == {"JA-207652 FINAL.pdf"}
    assert {s.file_name for s in due.sources} == {"Addendum 2.pdf"}
    assert llm.calls == [] and result.used_llm is False and result.complete
    assert [e.addendum_number for e in result.timeline.addenda] == [1, 2]
    assert by_key(result, "pre_bid_meeting").status == "unchanged"  # untouched fields stay as they were
    assert "Bid Summary" in result.skipped_fields
    assert result.reconciliation_required == []


async def test_unknown_bid() -> None:
    service, _ = make([RFP, ADD2])
    with pytest.raises(UnknownBidError, match="Bid9"):
        await service.reconcile_bid("Bid9")


# --------------------------------------------------------------- cross-bid isolation
async def test_other_bids_addenda_never_leak_in() -> None:
    service, retrieval = make([RFP, ADD2, *OTHER_BID])
    result = await service.reconcile_bid("Bid1")

    text = json.dumps(result.model_dump(), default=str)
    assert "August 1" not in text and "PORFP" not in text and "01-MAY-2024" not in text
    assert {c["bid_id"] for c in retrieval.calls} == {"Bid1"}
    assert by_key(result, "due_date").final_value == "July 9, 2024 at 2:00 PM CST"
    assert [e.file_name for e in result.timeline.addenda] == ["Addendum 2.pdf"]

    other = await service.reconcile_bid("Bid2")
    assert by_key(other, "due_date").final_value == "August 1, 2024 at 9:00 AM EST"


async def test_service_ignores_chunks_from_other_bids_even_if_retrieval_returns_them() -> None:
    service, _ = make([RFP, ADD2, *OTHER_BID], retrieval_cls=LeakyRetrieval)
    result = await service.reconcile_bid("Bid1")

    text = json.dumps(result.model_dump(), default=str)
    assert "August 1" not in text and "01-MAY-2024" not in text
    assert by_key(result, "due_date").final_value == "July 9, 2024 at 2:00 PM CST"


# ------------------------------------------------------------------ Groq fallback
BOND = chunk(
    31, 31, "Bid1", "Addendum 3.pdf",
    "ADDENDUM No. 3 The bid bond requirement is amended as described below. A bid bond of 5% of the bid amount is required.",
    "addendum",
)  # fmt: skip


def addendum_chunk_ids(prompt: str) -> list[int]:
    section = prompt.split("ADDENDUM EVIDENCE", 1)[1]
    return [int(i) for i in re.findall(r"\[chunk (\d+) \|", section)]


def reply(change_type="replacement", new_value="5% of the bid amount", ids=None, confidence=0.9):
    def build(prompt: str) -> str:
        return json.dumps({"changes": [{
            "change_type": change_type, "previous_value": None, "new_value": new_value,
            "explanation": "The addendum sets the requirement.", "chunk_ids": ids if ids is not None else addendum_chunk_ids(prompt),
            "confidence": confidence,
        }]})  # fmt: skip

    return build


async def test_ambiguous_wording_is_sent_to_groq_and_validated() -> None:
    llm = FakeLLM(script=[reply()])
    service, _ = make([RFP, BOND], llm)

    result = await service.reconcile_bid("Bid1")

    bond = by_key(result, "bid_bond")
    assert result.used_llm and len(llm.calls) == 1
    prompt = llm.calls[0][1]["content"]
    assert "5% of the bid amount" in prompt and "ADDENDUM 3" in prompt
    assert "Request For Proposal" not in prompt  # only relevant chunks, not whole documents
    assert (bond.status, bond.resolution_method, bond.final_value) == ("resolved", "llm_reconciliation", "5% of the bid amount")
    assert bond.confidence <= 0.8 and bond.sources[0].chunk_id == 31 and bond.changes[0].method == "llm"
    assert by_key(result, "due_date").status == "unchanged"  # deterministic fields never hit Groq


@pytest.mark.parametrize(
    ("script", "why"),
    [
        (reply(ids=[999]), "invented chunk id"),
        (reply(ids=[]), "no citation"),
        (reply(new_value="7% of the bid amount"), "value not contained in the cited text"),
        (reply(change_type="replacement", new_value=None), "replacement without a value"),
    ],
)
async def test_invalid_groq_claims_are_rejected_and_ambiguity_stays_visible(script, why) -> None:
    service, _ = make([RFP, BOND], FakeLLM(script=[script]))
    bond = by_key(await service.reconcile_bid("Bid1"), "bid_bond")

    assert bond.final_value is None and bond.status == "insufficient_evidence", why
    assert any(c.change_type == "unknown" for c in bond.changes), why


async def test_groq_saying_none_does_not_clear_explicit_amendment_wording() -> None:
    # "is amended as described below" is explicit change wording the rules could not parse, so a
    # model answer of "none" cannot make the field look unchanged (see the regression tests below).
    service, _ = make([RFP, BOND], FakeLLM(script=[reply(change_type="none", new_value=None)]))
    bond = by_key(await service.reconcile_bid("Bid1"), "bid_bond")
    assert bond.status == "insufficient_evidence" and any(c.change_type == "unknown" for c in bond.changes)


async def test_malformed_groq_json_is_retried_once() -> None:
    llm = FakeLLM(script=["not json", reply()])
    service, _ = make([RFP, BOND], llm)
    result = await service.reconcile_bid("Bid1")

    assert len(llm.calls) == 2 and by_key(result, "bid_bond").final_value == "5% of the bid amount"
    assert "could not be used" in llm.calls[1][-1]["content"]


async def test_groq_invalid_twice_reports_error_and_leaves_field_unresolved() -> None:
    llm = FakeLLM(script=['{"changes": "x"}', '{"changes": "x"}'])
    service, _ = make([RFP, BOND], llm)
    result = await service.reconcile_bid("Bid1")

    assert result.complete is False and "invalid after 2 attempts" in result.errors[0]
    bond = by_key(result, "bid_bond")
    assert bond.final_value is None and bond.status == "insufficient_evidence"
    assert "Bid Bond Requirement" in result.reconciliation_required


async def test_groq_provider_failure_propagates() -> None:
    import groq
    import httpx

    down = groq.APIConnectionError(request=httpx.Request("POST", "http://groq"))
    service, _ = make([RFP, BOND], FakeLLM(script=[down]))
    with pytest.raises(LLMProviderError):
        await service.reconcile_bid("Bid1")


async def test_missing_groq_configuration_degrades_to_unresolved_not_a_crash() -> None:
    service, _ = make([RFP, ADD2, BOND], llm=None)
    result = await service.reconcile_bid("Bid1")

    assert result.used_llm is False and result.complete is False
    assert "Groq is not configured" in result.errors[0]
    assert by_key(result, "bid_bond").status == "insufficient_evidence"
    assert by_key(result, "due_date").status == "resolved"  # the deterministic part still works


async def test_retrieval_failure_propagates() -> None:
    service, _ = make([RFP, ADD2], error=RuntimeError("db down"))
    with pytest.raises(RuntimeError):
        await service.reconcile_bid("Bid1")


async def test_extraction_supplies_originals_for_fields_without_locators() -> None:
    from app.extraction.schema import BidExtraction, ExtractedField, Source

    term = ExtractedField(
        value="3 years", status="found", confidence=0.9,
        sources=[Source(chunk_id=1, file_name="JA-207652 FINAL.pdf", document_type="bid_document", page_number=2, chunk_index=1)],
    )  # fmt: skip
    extraction = BidExtraction(bid_id="Bid1", fields={"Term of Bid": term})
    service, _ = make([RFP, ADD2])

    result = await service.reconcile_bid("Bid1", extraction)

    t = by_key(result, "term_of_bid")
    assert t.original_value == "3 years" and t.status == "unchanged" and t.confidence == 0.85


# ------------------------------------------- regression: real-model "none" must not hide an explicit change
STRONG = chunk(41, 41, "Bid1", "Addendum 4.pdf", "ADDENDUM No. 4 The bid bond requirement has been extended. A new amount will be announced.", "addendum")
VAGUE = chunk(42, 42, "Bid1", "Addendum 5.pdf", "ADDENDUM No. 5 A new bid bond requirement form is attached for reference.", "addendum")


def says_none(prompt: str) -> str:
    ids = addendum_chunk_ids(prompt)
    return json.dumps({"changes": [{"change_type": "none", "new_value": None, "explanation": "unaffected", "chunk_ids": ids, "confidence": 0.9}]})


async def test_model_saying_none_does_not_clear_an_explicit_change_verb() -> None:
    """Real Groq answered 'none' for 'the deadline has been extended; new date TBA', which made a stale value look final."""
    service, _ = make([RFP, STRONG], FakeLLM(script=[says_none]))
    bond = by_key(await service.reconcile_bid("Bid1"), "bid_bond")
    assert bond.status == "insufficient_evidence" and bond.final_value is None
    assert any(c.change_type == "unknown" for c in bond.changes)  # the ambiguity stays visible


async def test_model_saying_none_still_clears_a_vague_mention() -> None:
    service, _ = make([RFP, VAGUE], FakeLLM(script=[says_none]))
    bond = by_key(await service.reconcile_bid("Bid1"), "bid_bond")
    assert not any(c.change_type == "unknown" for c in bond.changes)
