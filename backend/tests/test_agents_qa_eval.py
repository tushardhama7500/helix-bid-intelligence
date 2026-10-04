import json
from pathlib import Path

from app.agents.qa_eval import DEFAULT_CASES, evaluate, has_expected_evidence, load_cases, summarize
from tests.extraction_fakes import FakeLLM
from tests.test_agents_qa import DEADLINE, build, say
from tests.test_reconciliation_service import ADD1, ADD2, RFP

CASES = [
    {"id": "deadline", "bid": "Bid1", "question": "What is the final submission deadline?", "answer": "July 9, 2024",
     "evidence": [{"file": "Addendum 2", "pattern": "July 9, 2024"}]},
    {"id": "award", "bid": "Bid1", "question": "Which vendor was awarded the contract?", "refuse": True},
]  # fmt: skip


def test_the_shipped_case_file_is_well_formed() -> None:
    cases = load_cases(DEFAULT_CASES)
    assert len(cases) >= 12 and {c["bid"] for c in cases} == {"Bid1", "Bid2"}
    assert all(("refuse" in c) or c.get("evidence") for c in cases) and len({c["id"] for c in cases}) == len(cases)
    assert sum(1 for c in cases if c.get("refuse")) >= 2


def test_expected_evidence_matching() -> None:
    assert has_expected_evidence([ADD2], [{"file": "addendum 2", "pattern": "July\\s+9"}])
    assert not has_expected_evidence([RFP], [{"file": "Addendum 2", "pattern": "July 9"}])


async def test_evidence_only_mode_needs_no_model() -> None:
    agent, _ = build([RFP, ADD1, ADD2], None)
    rows = await evaluate(agent, CASES, evidence_only=True)
    assert rows[0]["checks"] == {"evidence_available": True, "bid_isolation": True}
    assert rows[1]["checks"] == {}  # refusal cannot be judged without the model
    assert summarize(rows) == {"bid_isolation": (1, 1), "evidence_available": (1, 1)}


async def test_full_mode_scores_answers_citations_grounding_and_refusals() -> None:
    llm = FakeLLM(script=[say(DEADLINE, ids=[12, 1]), say(None, ids=[], answerable=False)])
    agent, _ = build([RFP, ADD1, ADD2], llm)
    rows = await evaluate(agent, CASES, evidence_only=False)

    assert rows[0]["checks"] == {"answer_correct": True, "isolation": True, "citation_correct": True, "grounded": True}
    assert rows[1]["checks"] == {"refusal": True, "isolation": True}
    assert rows[0]["seconds"] >= 0 and rows[0]["sources"]  # timing and cited sources are recorded per question


async def test_full_mode_catches_a_wrong_answer_and_a_failed_refusal() -> None:
    llm = FakeLLM(script=[say("The deadline is 27-JUN-2024 14:00:00.", ids=[1]), say("It was awarded in July 9, 2024 at 2:00 PM CST.", ids=[12])])
    agent, _ = build([RFP, ADD1, ADD2], llm)
    rows = await evaluate(agent, CASES, evidence_only=False)

    assert rows[0]["checks"]["answer_correct"] is False and rows[0]["checks"]["citation_correct"] is False
    assert rows[1]["checks"]["refusal"] is False
