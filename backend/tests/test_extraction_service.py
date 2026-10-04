import re

import groq
import httpx
import pytest

from app.core.config import Settings
from app.extraction.fields import FIELDS
from app.extraction.service import ExtractionService, LLMProviderError, UnknownBidError
from tests.extraction_fakes import BID1, BID2, FakeLLM, StubRetrieval, llm_json

SETTINGS = Settings(_env_file=None, database_url="postgresql+psycopg://x/x")


def service(llm=None, retrieval=None, settings=SETTINGS, sleeps=None):
    async def fake_sleep(seconds):
        if sleeps is not None:
            sleeps.append(seconds)

    return ExtractionService(retrieval or StubRetrieval(), llm or FakeLLM(), settings, sleep=fake_sleep)


DUE_DATE_CONFLICT = {
    "due_date": ([{"value": "27-JUN-2024 14:00:00", "chunk_ids": [1]},
                  {"value": "July 9, 2024 at 2:00 PM CST", "chunk_ids": [2]}], 0.9),
}  # fmt: skip


# ------------------------------------------------------------- retrieval integration
async def test_extraction_reuses_hybrid_retrieval_with_field_queries() -> None:
    retrieval = StubRetrieval()
    await service(retrieval=retrieval).extract_bid("Bid1")

    assert {c["bid_id"] for c in retrieval.calls} == {"Bid1"}  # always scoped to the requested bid
    queries = {c["query"] for c in retrieval.calls}
    for spec in FIELDS:
        assert spec.queries[0] in queries  # one retrieval per field query, not one generic query
    assert len(queries) > 20
    # amendable fields also ask for addendum text, so amendments are visible
    addendum_calls = [c for c in retrieval.calls if c["document_type"] == "addendum"]
    assert any("due date" in c["query"] for c in addendum_calls)
    assert not any("bid number" in c["query"] for c in addendum_calls)


async def test_evidence_is_sent_to_the_llm_once_per_chunk() -> None:
    llm = FakeLLM()
    await service(llm=llm).extract_bid("Bid1")

    assert len(llm.calls) == 5  # one call per field group, not per field
    schedule = next(c for c in llm.calls if 'key="due_date"' in c[1]["content"])[1]["content"]
    for chunk_id in (1, 2):
        assert len(re.findall(rf"\[chunk {chunk_id} \|", schedule)) == 1  # duplicates removed
    assert "Solicitation Due 27-JUN-2024" in schedule and "July 9, 2024" in schedule
    assert "Never use outside knowledge" in llm.calls[0][0]["content"]
    assert not any("Bid2" in m["content"] or "E20P4600040" in m["content"] for call in llm.calls for m in call)


async def test_chunks_per_call_are_limited() -> None:
    small = Settings(_env_file=None, database_url="postgresql+psycopg://x/x", extraction_max_chunks_per_call=2)
    llm = FakeLLM()
    await service(llm=llm, settings=small).extract_bid("Bid1")

    for call in llm.calls:
        assert len(re.findall(r"\[chunk \d+ \|", call[1]["content"])) <= 2


# ---------------------------------------------------------------- full extraction
async def test_bid1_extraction_flags_due_date_conflict_without_choosing() -> None:
    answers = {
        **DUE_DATE_CONFLICT,
        "bid_number": ([{"value": "JA-207652", "chunk_ids": [1]}], 0.95),
        "pre_bid_meeting": ([{"value": "10-JUN-2024 14:00:00", "chunk_ids": [1]}], 0.9),
    }
    result = await service(llm=FakeLLM(answers=answers)).extract_bid("Bid1")

    assert result.bid_id == "Bid1" and result.complete and result.model == "test-model"
    assert list(result.fields) == [f.name for f in FIELDS]
    due = result.fields["Due Date"]
    assert due.status == "conflict" and due.value is None and due.confidence == 0.0
    assert {s.file_name for s in due.sources} == {"JA-207652 FINAL.pdf", "Addendum 2 RFP JA-207652.pdf"}
    assert result.reconciliation_required == ["Due Date"]
    assert result.fields["Bid Number"].value == ["JA-207652"] and result.fields["Bid Number"].confidence > 0
    assert result.fields["Bid Bond Requirement"].status == "not_found"


async def test_bid2_extraction() -> None:
    answers = {
        "bid_number": ([{"value": "E20P4600040", "chunk_ids": [11]}], 0.9),
        "delivery_date": ([{"value": "Delivery within 45 days of Award", "chunk_ids": [11]}], 0.9),
        "product_specification": ([{"value": {"product": "Dell Latitude 5550", "specifications": {
            "processor": "Intel Core Ultra 5 125U", "memory": "16 GB DDR5", "storage": "256 GB"}}, "chunk_ids": [12]}], 0.9),
    }  # fmt: skip
    result = await service(llm=FakeLLM(answers=answers)).extract_bid("Bid2")

    assert result.fields["Bid Number"].value == ["E20P4600040"]
    assert result.fields["Delivery Date"].sources[0].file_name == "PORFP_Dell_Laptop.pdf"
    spec = result.fields["Product Specification"]
    assert spec.status == "found" and spec.value["specifications"]["processor"] == "Intel Core Ultra 5 125U"
    assert result.reconciliation_required == []


async def test_invented_chunk_ids_and_values_never_reach_the_result() -> None:
    answers = {
        "bid_number": ([{"value": "JA-207652", "chunk_ids": [999]}], 0.99),  # unknown chunk
        "title": ([{"value": "Student and Staff Computing Devices", "chunk_ids": [1, 424242]}], 0.9),
        "bid_bond": ([{"value": "5% bid bond", "chunk_ids": [1]}], 0.9),  # not in the cited text
    }
    result = await service(llm=FakeLLM(answers=answers)).extract_bid("Bid1")

    assert result.fields["Bid Number"].status == "unsupported"
    assert result.fields["Bid Number"].value is None
    assert result.fields["Title"].status == "found"
    assert [s.chunk_id for s in result.fields["Title"].sources] == [1]
    assert result.fields["Bid Bond Requirement"].status == "unsupported"
    assert result.fields["Bid Bond Requirement"].confidence == 0.0


# ----------------------------------------------------------------- Groq behaviour
async def test_malformed_json_is_retried_once_with_the_error() -> None:
    llm = FakeLLM(script=["this is not json"])  # first reply bad, everything afterwards well-formed
    result = await service(llm=llm).extract_bid("Bid1")

    assert result.complete and result.errors == []
    assert len(llm.calls) == 6  # 5 groups + 1 retry
    retry = llm.calls[1]
    assert retry[-2]["role"] == "assistant" and retry[-2]["content"] == "this is not json"
    assert "could not be used" in retry[-1]["content"]


async def test_invalid_schema_twice_reports_an_error_instead_of_partial_data() -> None:
    bad = '{"fields": [{"key": "bid_number", "candidates": [], "confidence": 7}]}'  # confidence out of range
    llm = FakeLLM(script=[bad, bad])
    result = await service(llm=llm).extract_bid("Bid1")

    assert result.complete is False and len(result.errors) == 1
    assert "invalid after 2 attempts" in result.errors[0]
    first_group = [result.fields[n] for n in ("Bid Number", "Title", "company_name", "contact_info", "Contract or Cooperative to use")]
    assert all(f.status == "error" and f.value is None and f.confidence == 0.0 for f in first_group)
    assert len(llm.calls) == 6  # 2 attempts for the failed group, then 4 normal groups
    assert result.fields["Due Date"].status != "error"  # other groups are unaffected


async def test_rate_limit_is_waited_out_once() -> None:
    response = httpx.Response(429, headers={"retry-after": "3"}, request=httpx.Request("POST", "http://groq"))
    rate_limited = groq.RateLimitError("slow down", response=response, body=None)
    sleeps: list[float] = []
    llm = FakeLLM(script=[rate_limited])

    result = await service(llm=llm, sleeps=sleeps).extract_bid("Bid1")

    assert sleeps == [3.0] and result.complete


async def test_persistent_rate_limit_and_provider_errors_are_fatal() -> None:
    response = httpx.Response(429, request=httpx.Request("POST", "http://groq"))
    limited = groq.RateLimitError("slow down", response=response, body=None)
    with pytest.raises(LLMProviderError, match="rate limit"):
        await service(llm=FakeLLM(script=[limited, limited])).extract_bid("Bid1")

    down = groq.APIConnectionError(request=httpx.Request("POST", "http://groq"))
    with pytest.raises(LLMProviderError, match="APIConnectionError"):
        await service(llm=FakeLLM(script=[down])).extract_bid("Bid1")


async def test_unknown_bid_fails_before_any_llm_call() -> None:
    llm = FakeLLM()
    with pytest.raises(UnknownBidError, match="Bid9"):
        await service(llm=llm).extract_bid("Bid9")
    assert llm.calls == []


async def test_group_without_evidence_skips_the_llm() -> None:
    llm = FakeLLM()
    result = await service(llm=llm, retrieval=StubRetrieval({"Empty": []})).extract_bid("Empty")

    assert llm.calls == []
    assert all(f.status == "not_found" and f.value is None for f in result.fields.values())
