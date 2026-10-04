import json
import re

import groq
import httpx
import pytest

from app.agents.report_qa_agent import QAOutputError, ReportQAAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.schema import REFUSAL
from app.agents.validation_agent import ChunkRecord
from app.extraction.schema import BidExtraction, ExtractedField, Source
from app.extraction.service import LLMProviderError
from app.reconciliation.schema import BidReconciliation, Timeline
from app.reconciliation.service import ReconciliationService
from tests.extraction_fakes import FakeLLM
from tests.test_reconciliation_service import ADD1, ADD2, OTHER_BID, RFP, SETTINGS, LeakyRetrieval, Retrieval, loader


def records(chunks):
    return {c.chunk_id: ChunkRecord(c.chunk_id, c.bid_id, c.file_name, c.document_type, c.page_number, c.chunk_index, c.content) for c in chunks}


def build(chunks, llm, retrieval_cls=Retrieval, lookup_override=None, with_recon=True):
    retrieval = retrieval_cls(chunks)
    store = lookup_override or records(chunks)

    async def lookup(ids):
        return {i: store[i] for i in ids if i in store}

    addenda = [c for c in chunks if c.document_type == "addendum"]
    recon = ReconciliationService(retrieval, SETTINGS, None, loader(addenda)) if with_recon else None
    return ReportQAAgent(RetrievalAgent(retrieval, SETTINGS), lookup, SETTINGS, llm, recon), retrieval


def evidence_ids(prompt: str) -> list[int]:
    return [int(i) for i in re.findall(r"\[chunk (\d+) \|", prompt)]


def say(answer, ids=None, answerable=True, confidence=0.9):
    def reply(prompt: str) -> str:
        return json.dumps({"answerable": answerable, "answer": answer, "chunk_ids": ids if ids is not None else evidence_ids(prompt), "confidence": confidence})

    return reply


DEADLINE = "The final submission deadline is July 9, 2024 at 2:00 PM CST (originally 27-JUN-2024 14:00:00)."


async def test_grounded_answer_uses_the_validated_reconciliation_and_cites_it() -> None:
    llm = FakeLLM(script=[say(DEADLINE, ids=[12, 1])])
    agent, _ = build([RFP, ADD1, ADD2], llm)

    out = await agent.answer("Bid1", "What is the final submission deadline?")

    assert out.answerable and out.answer == DEADLINE and out.used_reconciliation
    assert {(s.file, s.page, s.chunk_id) for s in out.sources} == {("Addendum 2.pdf", 1, 12), ("JA-207652 FINAL.pdf", 2, 1)}
    assert 0 < out.confidence <= 0.95
    prompt = llm.calls[0][1]["content"]
    assert "ADDENDUM-RECONCILED VALUE for Due Date (status: resolved)" in prompt
    assert "final: July 9, 2024 at 2:00 PM CST" in prompt and "original: 27-JUN-2024 14:00:00" in prompt


async def test_insufficient_evidence_is_refused() -> None:
    agent, _ = build([RFP, ADD2], FakeLLM(script=[say(None, ids=[], answerable=False)]))
    out = await agent.answer("Bid1", "Who won the contract?")
    assert (out.answer, out.answerable, out.sources, out.confidence) == (REFUSAL, False, [], 0.0)


async def test_no_evidence_at_all_is_refused_without_calling_the_model() -> None:
    llm = FakeLLM()
    agent, _ = build([RFP], llm)
    out = await agent.answer("Bid3", "anything?")
    assert out.answer == REFUSAL and llm.calls == []


async def test_answer_citing_unknown_chunks_is_rejected() -> None:
    agent, _ = build([RFP, ADD2], FakeLLM(script=[say("It is July 9, 2024 at 2:00 PM CST.", ids=[999])]))
    out = await agent.answer("Bid1", "What is the final submission deadline?")
    assert out.answer == REFUSAL and out.sources == [] and "no retrieved evidence" in out.notes


async def test_answer_with_figures_not_in_the_cited_text_is_rejected() -> None:
    agent, _ = build([RFP, ADD2], FakeLLM(script=[say("The deadline is July 10, 2024 at 3:00 PM CST.", ids=[12])]))
    out = await agent.answer("Bid1", "What is the final submission deadline?")
    assert out.answer == REFUSAL and "figures" in out.notes


async def test_other_bids_evidence_never_reaches_the_prompt_or_the_citations() -> None:
    llm = FakeLLM(script=[say("Quotes are due August 1, 2024.", ids=[22])])  # the model cites Bid2's chunk
    agent, _ = build([RFP, ADD2, *OTHER_BID], llm, retrieval_cls=LeakyRetrieval)

    out = await agent.answer("Bid1", "What is the deadline?")

    prompt = llm.calls[0][1]["content"]
    assert "August 1" not in prompt and "PORFP" not in prompt and "[chunk 22" not in prompt
    assert out.answer == REFUSAL and out.sources == []


async def test_reconciliation_chunks_from_another_bid_are_dropped() -> None:
    chunks = [RFP, ADD2]
    wrong = records(chunks)
    wrong[1] = ChunkRecord(1, "Bid2", "JA-207652 FINAL.pdf", "bid_document", 2, 1, RFP.content)  # the store says chunk 1 is Bid2's
    llm = FakeLLM(script=[say("The deadline is July 9, 2024 at 2:00 PM CST.", ids=[12])])
    agent, retrieval = build(chunks, llm, lookup_override=wrong)

    await agent.answer("Bid1", "What is the final submission deadline?")

    prompt = llm.calls[0][1]["content"]
    assert prompt.count("[chunk 1 |") == 1  # only the Bid1 retrieval copy of chunk 1 is present; lookup's Bid2 record was refused


async def test_groq_failure_propagates() -> None:
    down = groq.APIConnectionError(request=httpx.Request("POST", "http://groq"))
    agent, _ = build([RFP, ADD2], FakeLLM(script=[down]))
    with pytest.raises(LLMProviderError):
        await agent.answer("Bid1", "What is the deadline?")


async def test_malformed_model_output_is_retried_once_then_reported() -> None:
    llm = FakeLLM(script=["nope", say("Due July 9, 2024 at 2:00 PM CST.", ids=[12])])
    agent, _ = build([RFP, ADD2], llm)
    out = await agent.answer("Bid1", "What is the deadline?")
    assert out.answerable and len(llm.calls) == 2 and "could not be used" in llm.calls[1][-1]["content"]

    bad = FakeLLM(script=['{"answerable": "maybe"}', '{"x": 1}'])
    agent, _ = build([RFP, ADD2], bad)
    with pytest.raises(QAOutputError, match="invalid after 2 attempts"):
        await agent.answer("Bid1", "What is the deadline?")


async def test_questions_not_about_a_reconcilable_field_skip_reconciliation() -> None:
    llm = FakeLLM(script=[say("Quotes are submitted with Form 1295.", ids=[1])])
    agent, retrieval = build([RFP, ADD2], llm)
    out = await agent.answer("Bid1", "Who is the buyer?")
    assert not out.used_reconciliation and "ADDENDUM-RECONCILED" not in llm.calls[0][1]["content"]


async def test_preconditions() -> None:
    agent, _ = build([RFP], None)
    with pytest.raises(RuntimeError, match="Groq"):
        await agent.answer("Bid1", "x?")
    agent, _ = build([RFP], FakeLLM())
    with pytest.raises(ValueError):
        await agent.answer("Bid1", "   ")


# ------------------------------------------------------------------------- report
def src(chunk_id, file="a.pdf", doc_type="bid_document", page=1) -> Source:
    return Source(chunk_id=chunk_id, file_name=file, document_type=doc_type, page_number=page, chunk_index=0)


async def test_report_prefers_reconciled_values_and_never_invents() -> None:
    extraction = BidExtraction(bid_id="Bid1", fields={
        "Due Date": ExtractedField(value=None, sources=[src(1), src(2)], status="conflict", notes="conflict",
                                   conflicting_values=[__import__("app.extraction.schema", fromlist=["x"]).ConflictingValue(value="a", sources=[src(1)]),
                                                       __import__("app.extraction.schema", fromlist=["x"]).ConflictingValue(value="b", sources=[src(2)])]),
        "Title": ExtractedField(value="Devices", sources=[src(3)], confidence=0.9, status="found"),
        "Bid Bond Requirement": ExtractedField.missing(),
    })  # fmt: skip
    agent, _ = build([RFP, ADD2], FakeLLM())
    recon = await agent.reconciliation.reconcile_bid("Bid1", None)

    report = ReportQAAgent.build_report("Bid1", extraction, recon)
    by = {f.field: f for f in report.fields}

    assert by["Due Date"].value == "July 9, 2024 at 2:00 PM CST" and by["Due Date"].changed_by_addendum
    assert by["Due Date"].original_value == "27-JUN-2024 14:00:00" and by["Due Date"].origin == "reconciliation"
    assert by["Title"].value == "Devices" and by["Title"].origin == "extraction"
    assert by["Bid Bond Requirement"].value is None and by["Bid Bond Requirement"].status == "not_found"
    assert len(report.fields) == 20


async def test_report_without_extraction_marks_fields_not_extracted_and_conflicts_unresolved() -> None:
    from app.reconciliation.reconcile import OriginalValue, reconcile_field

    conflict = reconcile_field("due_date", "Due Date", [OriginalValue("A", [src(1)]), OriginalValue("B", [src(2)])], [], Timeline(bid_id="Bid1"))
    recon = BidReconciliation(bid_id="Bid1", timeline=Timeline(bid_id="Bid1"), fields=[conflict])

    report = ReportQAAgent.build_report("Bid1", None, recon)
    by = {f.field: f for f in report.fields}
    assert by["Due Date"].status == "unresolved" and by["Due Date"].value is None
    assert by["Title"].status == "not_extracted" and report.unresolved == ["Due Date"]
