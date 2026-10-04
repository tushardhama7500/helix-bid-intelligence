"""Q&A evaluation: `python -m app.agents.qa_eval [--evidence-only] [-v]`.

Full mode (needs Groq) measures, per case:
  answer correctness   - the answer matches the expected pattern
  citation correctness - every cited chunk is from the asked bid, and one is the expected evidence
  grounding            - the answer's figures occur in the cited text (re-checked here)
  refusal              - cases with no support in the bid's documents must be refused
`--evidence-only` needs no Groq: it checks whether the expected evidence is among what the Q&A
agent would hand to the model (retrieval + validated reconciliation).
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

from app.agents.factory import build_qa_agent, optional_groq
from app.agents.report_qa_agent import QAOutputError, ReportQAAgent
from app.core.config import get_settings
from app.core.database import dispose_engine, get_sessionmaker
from app.extraction.service import LLMProviderError
from app.extraction.validation import grounding
from app.retrieval.schema import Candidate

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "eval" / "qa_eval.json"


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})


def _squash(text: str) -> str:
    return " ".join(text.translate(_DASHES).split())


def has_expected_evidence(chunks: list[Candidate], expected: list[dict[str, str]]) -> bool:
    return any(
        e["file"].lower() in c.file_name.lower() and re.search(e["pattern"], _squash(c.content), re.I)
        for c in chunks for e in expected
    )  # fmt: skip


async def evidence_for(agent: ReportQAAgent, bid: str, question: str) -> list[Candidate]:
    """Exactly what the agent would give the model: retrieved chunks plus reconciliation chunks."""
    evidence = await agent._evidence(bid, question)
    fields = await agent._reconciled_fields(bid, question, None)
    for c in await agent._reconciliation_chunks(bid, fields):
        evidence.setdefault(c.chunk_id, c)
    return list(evidence.values())


async def _answer(agent: ReportQAAgent, case: dict[str, Any], patient: bool):
    """Ask one question. With `patient` (evaluation only), wait out provider rate limits (e.g. Groq's
    per-day token cap) instead of failing; production code keeps its single bounded wait."""
    for attempt in range(1, 121 if patient else 2):
        try:
            return await agent.answer(case["bid"], case["question"])
        except LLMProviderError as exc:
            if not patient or "rate limit" not in str(exc).lower() or attempt == 120:
                raise
            await asyncio.sleep(60)


async def evaluate(
    agent: ReportQAAgent, cases: list[dict[str, Any]], evidence_only: bool, patient: bool = False
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        row: dict[str, Any] = {"id": case["id"], "checks": {}, "answer": None}
        if evidence_only:
            if not case.get("refuse"):
                chunks = await evidence_for(agent, case["bid"], case["question"])
                row["checks"]["evidence_available"] = has_expected_evidence(chunks, case["evidence"])
                row["checks"]["bid_isolation"] = all(c.bid_id == case["bid"] for c in chunks)
            rows.append(row)
            continue

        calls_before = len(getattr(agent.llm, "calls", []))
        started = time.perf_counter()
        try:
            result = await _answer(agent, case, patient)
        except (LLMProviderError, QAOutputError) as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            rows.append(row)  # a provider failure is not an answer: reported, excluded from the metrics
            continue
        mine = [c for c in list(getattr(agent.llm, "calls", []))[calls_before:] if hasattr(c, "seconds")]  # real client records only
        row["seconds"] = round(time.perf_counter() - started, 1)
        row["groq"] = {"calls": len(mine), "tokens_in": sum(c.prompt_tokens or 0 for c in mine if c.ok),
                       "tokens_out": sum(c.completion_tokens or 0 for c in mine if c.ok), "prompt_chars": [c.prompt_chars for c in mine]}  # fmt: skip
        row["answer"] = result.answer
        row["answerable"] = result.answerable
        row["sources"] = [f"{s.file} p.{s.page}" for s in result.sources]
        row["notes"] = result.notes
        if case.get("refuse"):
            row["checks"]["refusal"] = not result.answerable and result.sources == []
            cited = await agent.lookup([s.chunk_id for s in result.sources])
            row["checks"]["isolation"] = all(r.bid_id == case["bid"] for r in cited.values())
        else:
            if not case.get("unscored_answer"):
                row["checks"]["answer_correct"] = result.answerable and bool(re.search(case["answer"], _squash(result.answer), re.I))
            cited = await agent.lookup([s.chunk_id for s in result.sources])
            records = list(cited.values())
            row["checks"]["isolation"] = all(r.bid_id == case["bid"] for r in records)
            row["checks"]["citation_correct"] = (
                result.answerable and bool(records) and all(r.bid_id == case["bid"] for r in records)
                and has_expected_evidence([r.as_candidate() for r in records], case["evidence"])
            )  # fmt: skip
            if result.answerable:
                padded = [Candidate(**{**r.as_candidate().__dict__, "content": f"{r.content} {r.file_name} {case['question']}"}) for r in records]
                row["checks"]["grounded"] = bool(padded) and grounding(result.answer, padded, "text") is not None
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    names = sorted({k for r in rows for k in r["checks"]})
    return {n: (sum(r["checks"][n] for r in rows if n in r["checks"]), sum(1 for r in rows if n in r["checks"])) for n in names}


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    logging.basicConfig(level="WARNING")
    llm = None if args.evidence_only else optional_groq(settings)
    if not args.evidence_only and llm is None:
        print("Groq is not configured (GROQ_API_KEY / GROQ_MODEL): use --evidence-only, or configure Groq.")
        return 2
    cases = load_cases()
    try:
        async with get_sessionmaker()() as session:
            agent = build_qa_agent(session, settings, llm)
            rows = await evaluate(agent, cases, args.evidence_only, args.patient)
    finally:
        await dispose_engine()

    for r in rows:
        flags = " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r["checks"].items()) or "(not evaluable without Groq)"
        print(f"{r['id']:<24} {flags}")
        if r.get("error"):
            print(f"    ERROR: {r['error']}")
        if args.verbose and r["answer"]:
            print(f"    -> {r['answer'][:200]}  | sources: {r.get('sources')} | {r.get('seconds')}s {r.get('groq')}")
    print("\nMetrics (passed/total):")
    for name, (ok, total) in summarize(rows).items():
        print(f"  {name:<20} {ok}/{total}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    errors = [r["id"] for r in rows if r.get("error")]
    if errors:
        print(f"  provider/model errors (excluded from metrics): {errors}")
    timed = [r for r in rows if r.get("seconds") is not None]
    if timed:
        secs = [r["seconds"] for r in timed]
        print(f"  Q&A latency avg/min/max={sum(secs)/len(secs):.1f}/{min(secs):.1f}/{max(secs):.1f}s over {len(secs)} questions; "
              f"Groq calls={sum(r['groq']['calls'] for r in timed)}, tokens in/out="
              f"{sum(r['groq']['tokens_in'] for r in timed)}/{sum(r['groq']['tokens_out'] for r in timed)}")
    if args.evidence_only:
        print("  (answer correctness, grounding and refusal are NOT measured in --evidence-only mode)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--patient", action="store_true", help="wait out provider rate limits (evaluation only)")
    parser.add_argument("--json", help="write per-question results to this file")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # document text can contain non-cp1252 characters
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main(parser.parse_args())))
