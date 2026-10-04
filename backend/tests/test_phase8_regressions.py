"""Regression tests for problems found by running the system against real Groq and real documents.

They check behaviour and grounding, never one model's exact wording.
"""

import json
import re

import groq
import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.extraction_agent import ReconciliationAgent
from app.agents.report_qa_agent import QA_VARIANT_CHUNKS, ReportQAAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.validation_agent import ChunkRecord
from app.core.config import Settings, get_settings
from app.embeddings.model import EmbeddingModelError, get_embedding_model
from app.extraction.fields import FIELDS_BY_KEY
from app.extraction.service import ExtractionService, LLMProviderError, call_llm, generate_validated
from app.retrieval.expansion import MAX_VARIANTS, is_weak, query_variants
from app.retrieval.schema import FusedResult
from tests.extraction_fakes import BID1, FakeLLM, Found, StubRetrieval, cand, llm_json
from tests.test_agents_qa import records, say
from tests.test_reconciliation_service import ADD2, BOND, RFP, SETTINGS, Retrieval, loader

SCOPE_QUERY = "what goods or services does this request for proposal solicit"
SCOPE = cand(77, "Student and Staff Computing Devices - BidNet.html", "This RFP is for student and staff computing devices, including laptops, desktops, tablets and monitors.", page=None, bid="Bid1")


# ============================ 1. generic product questions (retrieval expansion)
@pytest.mark.parametrize("question", [
    "What product is being procured?", "Which equipment is being purchased?", "What items are being bought?", "What goods are requested?",
])
def test_generic_procurement_questions_get_a_solicitation_phrased_variant(question) -> None:
    variants = query_variants(question)
    assert variants == [SCOPE_QUERY] and len(variants) <= MAX_VARIANTS
    assert variants[0].lower() != question.lower()


@pytest.mark.parametrize("question", [
    "What is the final submission deadline?", "Is a Mercury Affidavit required?", "Who is the contact person?", "WD22TB4",
])
def test_specific_questions_are_not_expanded(question) -> None:
    assert query_variants(question) == []


def test_weakness_means_no_result_confirmed_by_both_channels() -> None:
    assert is_weak([["vector"], ["keyword"], ["vector"]]) and is_weak([])
    assert not is_weak([["vector"], ["vector", "keyword"]])


class ScopeRetrieval(Retrieval):
    """The scope chunk is only reachable through the solicitation-phrased query (as in the real corpus)."""

    def __init__(self, chunks, methods):
        super().__init__(chunks)
        self.methods, self.queries = methods, []

    async def retrieve(self, query, bid_id=None, document_type=None, document_id=None, top_k=5, **_):
        self.queries.append({"query": query, "bid_id": bid_id, "top_k": top_k, "document_type": document_type})
        pool = [c for c in self.chunks if c.bid_id == bid_id and document_type in (None, c.document_type)]
        if SCOPE_QUERY not in query:
            pool = [c for c in pool if c.chunk_id != SCOPE.chunk_id]
        return Found([FusedResult(candidate=c, rrf_score=0.01, methods=list(self.methods)) for c in pool[:top_k]])


def qa(chunks, methods, llm=None):
    retrieval = ScopeRetrieval(chunks, methods)

    async def lookup(ids):
        store = records(chunks)
        return {i: store[i] for i in ids if i in store}

    return ReportQAAgent(RetrievalAgent(retrieval, SETTINGS), lookup, SETTINGS, llm or FakeLLM()), retrieval


async def test_weak_retrieval_triggers_a_bounded_bid_scoped_variant_and_finds_the_chunk() -> None:
    agent, retrieval = qa([RFP, ADD2, SCOPE], methods=["vector"])
    evidence = await agent._evidence("Bid1", "What product is being procured?")

    assert SCOPE.chunk_id in evidence  # found only through the variant
    variant_calls = [q for q in retrieval.queries if q["query"] == SCOPE_QUERY]
    assert len(variant_calls) == 1 and variant_calls[0]["bid_id"] == "Bid1" and variant_calls[0]["top_k"] == QA_VARIANT_CHUNKS
    assert {q["bid_id"] for q in retrieval.queries} == {"Bid1"}  # generated queries never escape the bid filter


async def test_confirmed_retrieval_does_not_expand() -> None:
    agent, retrieval = qa([RFP, ADD2, SCOPE], methods=["vector", "keyword"])
    evidence = await agent._evidence("Bid1", "What product is being procured?")
    assert SCOPE.chunk_id not in evidence and all(q["query"] != SCOPE_QUERY for q in retrieval.queries)


async def test_expansion_cannot_leak_another_bids_chunks() -> None:
    foreign = cand(99, "PORFP.pdf", "Dell Latitude 5550 laptops and docks.", bid="Bid2")
    agent, _ = qa([RFP, SCOPE, foreign], methods=["vector"])
    evidence = await agent._evidence("Bid1", "What product is being procured?")
    assert 99 not in evidence


@pytest.fixture
async def real():
    settings = get_settings().model_copy(update={"groq_api_key": None})
    engine = create_async_engine(settings.async_database_url)
    try:
        async with engine.connect() as conn:
            embedded = await conn.scalar(text("SELECT count(embedding) FROM document_chunks"))
            bids = (await conn.execute(text("SELECT DISTINCT bid_id FROM documents"))).scalars().all()
    except Exception:
        await engine.dispose()
        pytest.skip("real database not available")
    if embedded < 200 or {"Bid1", "Bid2"} - set(bids):
        await engine.dispose()
        pytest.skip("real documents are not ingested and embedded")
    try:
        get_embedding_model(settings.embedding_model, settings.embedding_batch_size).load()
    except EmbeddingModelError as exc:
        await engine.dispose()
        pytest.skip(f"embedding model unavailable: {exc}")
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session, settings
    await engine.dispose()


async def test_real_bid1_generic_product_question_now_retrieves_the_scope_text(real) -> None:
    """Phase 7 miss: the answering chunks ranked 31st/45th for this wording; no Groq needed to check."""
    from app.agents.factory import build_qa_agent
    from app.agents.qa_eval import evidence_for

    session, settings = real
    agent = build_qa_agent(session, settings, None)
    chunks = await evidence_for(agent, "Bid1", "What product is being procured?")
    text_ = " ".join(" ".join(c.content.split()) for c in chunks)

    assert re.search(r"computing devices, including laptops", text_, re.I)
    assert {c.bid_id for c in chunks} == {"Bid1"}
    primary = await agent.retrieval.retrieve("Bid1", "What product is being procured?", top_k=6)
    assert is_weak([r.methods for r in primary])  # the trigger really is the cause


# ====================== 2. extraction: wrong citations, scope, follow-up, list fields
def headers(prompt: str) -> dict[str, int]:
    return {file.strip(): int(i) for i, file in re.findall(r"\[chunk (\d+) \| ([^|]+) \|", prompt)}


def evidence_for_identity(**per_field):
    base = {key: [] for key in FIELDS_BY_KEY}
    base.update(per_field)
    return base


def identity_reply(cited_file: str, value="Procurement Services 972-925-3700"):
    def reply(prompt: str) -> str:
        return llm_json(prompt, {"contact_info": ([{"value": [value], "chunk_ids": [headers(prompt)[cited_file]]}], 0.9)})

    return reply


CONTACT = cand(50, "BidNet.html", "Contact Information: Procurement Services 972-925-3700 ProcurementCS@dallasisd.org", page=None)
NOISE = cand(51, "RFP.pdf", "Request for Proposal 168884. General information and schedule.", page=2)


async def run_extraction(llm, evidence):
    service = ExtractionService(StubRetrieval(), llm, SETTINGS, sleep=lambda s: __import__("asyncio").sleep(0))
    return service, await service.extract_bid("Bid1", evidence)


async def test_citing_a_chunk_from_another_fields_evidence_is_accepted_when_it_was_sent_in_the_call() -> None:
    """Real run: the contact details were only in a chunk retrieved for another field of the same call."""
    evidence = evidence_for_identity(bid_number=[CONTACT], contact_info=[NOISE])
    _, result = await run_extraction(FakeLLM(script=[identity_reply("BidNet.html")]), evidence)
    field = result.fields["contact_info"]
    assert field.status == "found" and [s.chunk_id for s in field.sources] == [50]


async def test_a_value_the_cited_chunk_does_not_contain_is_still_rejected() -> None:
    evidence = evidence_for_identity(bid_number=[CONTACT], contact_info=[NOISE])
    llm = FakeLLM(script=[identity_reply("RFP.pdf"), identity_reply("RFP.pdf")])  # wrong citation, then wrong again
    service, result = await run_extraction(llm, evidence)

    assert result.fields["contact_info"].status == "unsupported" and result.fields["contact_info"].value is None
    assert service._citation_retries == 1 and result.citation_retries == 1


async def test_citation_follow_up_fixes_a_wrong_citation_without_repairing_anything_itself() -> None:
    evidence = evidence_for_identity(bid_number=[CONTACT], contact_info=[NOISE])
    llm = FakeLLM(script=[identity_reply("RFP.pdf"), identity_reply("BidNet.html")])
    _, result = await run_extraction(llm, evidence)

    assert result.fields["contact_info"].status == "found" and result.citation_retries == 1
    follow_up = llm.calls[1]
    assert follow_up[-1]["role"] == "user" and "could not be verified" in follow_up[-1]["content"]
    assert "contact_info" in follow_up[-1]["content"] and follow_up[-2]["role"] == "assistant"
    assert len(llm.calls) == 2  # the identity call plus exactly one follow-up (other groups have no evidence)


async def test_a_hallucinated_value_is_not_rescued_by_the_follow_up() -> None:
    evidence = evidence_for_identity(bid_number=[CONTACT], contact_info=[NOISE])
    invented = identity_reply("BidNet.html", value="Jane Doe 555-000-1234")
    _, result = await run_extraction(FakeLLM(script=[invented, invented]), evidence)
    assert result.fields["contact_info"].status == "unsupported" and result.fields["contact_info"].value is None


async def test_malformed_follow_up_is_tolerated() -> None:
    evidence = evidence_for_identity(bid_number=[CONTACT], contact_info=[NOISE])
    _, result = await run_extraction(FakeLLM(script=[identity_reply("RFP.pdf"), "not json"]), evidence)
    assert result.fields["contact_info"].status == "unsupported" and result.complete


async def test_no_follow_up_when_nothing_was_rejected() -> None:
    evidence = evidence_for_identity(bid_number=[CONTACT], contact_info=[CONTACT])
    llm = FakeLLM(script=[identity_reply("BidNet.html")])
    _, result = await run_extraction(llm, evidence)
    assert result.citation_retries == 0 and len(llm.calls) == 1


@pytest.mark.parametrize("key", ["bid_number", "model_no", "part_no", "product", "installation", "bid_submission_type", "additional_documentation", "contact_info"])
def test_fields_that_legitimately_have_several_values_are_lists(key) -> None:
    """Real runs reported a PORFP number + project number, a laptop + dock, ... as false 'conflicts'."""
    assert FIELDS_BY_KEY[key].kind == "list"


def test_dates_stay_scalar_so_real_conflicts_are_still_surfaced() -> None:
    assert FIELDS_BY_KEY["due_date"].kind == "scalar" and FIELDS_BY_KEY["pre_bid_meeting"].kind == "scalar"


def test_delivery_date_excludes_repair_and_response_times() -> None:
    assert "warranty repair" in FIELDS_BY_KEY["delivery_date"].description.lower()


# ============================================ 3. bounded failure behaviour
async def test_timeout_is_a_structured_error_with_a_single_attempt() -> None:
    timeout = groq.APITimeoutError(request=httpx.Request("POST", "http://x"))

    class Slow:
        model, n = "m", 0

        async def generate_json(self, messages, **_):
            self.n += 1
            raise timeout

    llm = Slow()
    with pytest.raises(LLMProviderError, match="APITimeoutError"):
        await call_llm(llm, [{"role": "user", "content": "x"}], max_tokens=10)
    assert llm.n == 1  # the SDK already retries transient errors; this layer never loops


async def test_rate_limit_retries_once_then_stops() -> None:
    limited = groq.RateLimitError("slow", response=httpx.Response(429, headers={"retry-after": "1"}, request=httpx.Request("POST", "http://x")), body=None)
    waits: list[float] = []

    class Always:
        model, n = "m", 0

        async def generate_json(self, messages, **_):
            self.n += 1
            raise limited

    async def record(seconds):
        waits.append(seconds)

    llm = Always()
    with pytest.raises(LLMProviderError, match="rate limit"):
        await call_llm(llm, [{"role": "user", "content": "x"}], max_tokens=10, sleep=record)
    assert llm.n == 2 and waits == [1.0]


async def test_invalid_json_is_retried_exactly_once_then_reported() -> None:
    llm = FakeLLM(script=["nope", "still nope", "never reached"])
    parsed, error = await generate_validated(llm, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], json.loads, max_tokens=10)
    assert parsed is None and "invalid after 2 attempts" in error and len(llm.calls) == 2


async def test_reconciliation_agent_keeps_the_deterministic_result_when_groq_fails() -> None:
    from app.reconciliation.service import ReconciliationService

    down = groq.APIConnectionError(request=httpx.Request("POST", "http://x"))
    retrieval = Retrieval([RFP, ADD2, BOND])
    addenda = [ADD2, BOND]
    service = ReconciliationService(retrieval, SETTINGS, FakeLLM(script=[down]), loader(addenda))

    result, warnings = await ReconciliationAgent(service).run("Bid1", None)

    due = next(f for f in result.fields if f.key == "due_date")
    assert due.final_value == "July 9, 2024 at 2:00 PM CST" and due.status == "resolved"  # not lost
    assert any("Groq was unavailable" in w for w in warnings)
    assert result.used_llm is False


# ============================== 4. grounding cannot be bypassed by model output
async def test_qa_answers_with_ungrounded_figures_are_refused_whatever_the_model_claims() -> None:
    agent, _ = qa([RFP, ADD2], ["vector", "keyword"], FakeLLM(script=[say("Quotes are due July 10, 2024 at 5:00 PM CST.", ids=[12], confidence=1.0)]))
    out = await agent.answer("Bid1", "What is the deadline?")
    assert out.answerable is False and out.sources == []


async def test_qa_cannot_cite_chunks_outside_the_supplied_evidence() -> None:
    agent, _ = qa([RFP, ADD2, SCOPE], ["vector", "keyword"], FakeLLM(script=[say("Anything at all.", ids=[12345])]))
    assert (await agent.answer("Bid1", "What is the deadline?")).answerable is False


# ========== 5. Groq JSON mode answers HTTP 400 `json_validate_failed` for invalid JSON (seen with a real model)
def json_failure(code="json_validate_failed") -> groq.BadRequestError:
    return groq.BadRequestError(
        "Failed to validate JSON", response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
        body={"error": {"code": code, "type": "invalid_request_error", "message": "Failed to validate JSON."}},
    )


async def test_provider_json_failure_is_malformed_output_not_an_outage() -> None:
    from app.extraction.service import ProviderJSONError

    class Bad:
        model = "m"

        async def generate_json(self, messages, **_):
            raise json_failure()

    with pytest.raises(ProviderJSONError):
        await call_llm(Bad(), [{"role": "user", "content": "x"}], max_tokens=10)
    assert issubclass(ProviderJSONError, ValueError) and not issubclass(ProviderJSONError, LLMProviderError)


async def test_other_bad_requests_stay_fatal_provider_errors() -> None:
    class Bad:
        model = "m"

        async def generate_json(self, messages, **_):
            raise json_failure("context_length_exceeded")

    with pytest.raises(LLMProviderError, match="BadRequestError"):
        await call_llm(Bad(), [{"role": "user", "content": "x"}], max_tokens=10)


async def test_json_failure_is_retried_once_then_reported() -> None:
    llm = FakeLLM(script=[json_failure(), json.dumps({"ok": True})])
    parsed, error = await generate_validated(llm, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], json.loads, max_tokens=10)
    assert parsed == {"ok": True} and error is None and len(llm.calls) == 2

    llm = FakeLLM(script=[json_failure(), json_failure(), json.dumps({"never": "reached"})])
    parsed, error = await generate_validated(llm, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], json.loads, max_tokens=10)
    assert parsed is None and "invalid after 2 attempts" in error and len(llm.calls) == 2


async def test_extraction_group_survives_one_provider_json_failure() -> None:
    evidence = evidence_for_identity(bid_number=[CONTACT], contact_info=[CONTACT])
    llm = FakeLLM(script=[json_failure(), identity_reply("BidNet.html")])
    _, result = await run_extraction(llm, evidence)
    assert result.fields["contact_info"].status == "found" and result.schema_retries == 1 and result.complete


async def test_qa_answer_is_refused_cleanly_when_the_provider_cannot_produce_json() -> None:
    from app.agents.report_qa_agent import QAOutputError

    agent, _ = qa([RFP, ADD2], ["vector", "keyword"], FakeLLM(script=[json_failure(), json_failure()]))
    with pytest.raises(QAOutputError):
        await agent.answer("Bid1", "What is the deadline?")


def test_evaluator_ignores_typographic_dashes() -> None:
    from app.agents.qa_eval import _squash

    assert re.search("10-JUN-2024", _squash("meeting on 10‑JUN‑2024 at 2 PM"), re.I)
