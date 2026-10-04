"""Behavioural regression cases (eval/regression/*.json): `python -m app.agents.regression`.

Every case documents WHAT must hold, not how a model words it, and is checked deterministically
against the real ingested documents (no Groq needed):

  evidence_contains   the evidence the Q&A agent would send to the model contains the expected text
  field_evidence      the evidence retrieved for an extraction field contains every pattern
                      (so a real model can surface the conflict / value)
  reconciliation      deterministic reconciliation yields the expected final / original value
  no_timeline         the bid has no addenda and nothing is changed by one
  no_cross_bid        every chunk retrieved for the question belongs to the asked bid
  absent_from_bid     the text does not occur anywhere in the bid, so the only correct answer is
                      null / a refusal (a missing value must never be invented)
  validated_grounding the reconciled output passes the Validation agent (citations + grounding)

Case schema: {id, bid_id, question_or_field, expected_behavior, expected_sources[], notes, check, params}.
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.factory import build_qa_agent
from app.agents.qa_eval import evidence_for, has_expected_evidence
from app.agents.validation_agent import ValidationAgent, db_chunk_lookup
from app.core.config import Settings, get_settings
from app.core.database import dispose_engine, get_sessionmaker
from app.extraction.evidence import gather_evidence
from app.reconciliation.service import ReconciliationService
from app.retrieval.service import RetrievalService

REGRESSION_DIR = Path(__file__).resolve().parents[2] / "eval" / "regression"
REQUIRED_KEYS = {"id", "bid_id", "question_or_field", "expected_behavior", "expected_sources", "notes", "check", "params"}
CHECKS = {"evidence_contains", "field_evidence", "reconciliation", "no_timeline", "no_cross_bid", "absent_from_bid", "validated_grounding"}


def load_cases(directory: Path = REGRESSION_DIR) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(directory.glob("*.json")):
        for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
            case["_file"] = path.name
            cases.append(case)
    return cases


def validate_case(case: dict[str, Any]) -> list[str]:
    problems = [f"missing key {k}" for k in REQUIRED_KEYS - case.keys()]
    if case.get("check") not in CHECKS:
        problems.append(f"unknown check {case.get('check')!r}")
    if not isinstance(case.get("expected_sources"), list):
        problems.append("expected_sources must be a list")
    return problems


def _squash(value: str) -> str:
    return " ".join(value.split())


async def run_case(case: dict[str, Any], session: AsyncSession, settings: Settings) -> tuple[bool, str]:
    bid, check, params = case["bid_id"], case["check"], case["params"]
    retrieval = RetrievalService.from_settings(session, settings)

    if check == "evidence_contains":
        chunks = await evidence_for(build_qa_agent(session, settings, None), bid, case["question_or_field"])
        ok = has_expected_evidence(chunks, case["expected_sources"])
        return ok, f"{len(chunks)} evidence chunks; expected source present: {ok}"

    if check == "field_evidence":
        evidence = await gather_evidence(retrieval, bid, settings)
        joined = _squash(" ".join(c.content for c in evidence[params["field"]]))
        missing = [p for p in params["patterns"] if not re.search(p, joined, re.I)]
        return not missing, f"patterns missing from the {params['field']} evidence: {missing}"

    if check in ("reconciliation", "no_timeline"):
        result = await ReconciliationService(retrieval, settings, None).reconcile_bid(bid)
        if check == "no_timeline":
            clean = result.timeline.addenda == [] and not any(f.changes or f.status == "resolved" for f in result.fields)
            return clean, f"addenda={len(result.timeline.addenda)}"
        field = next(f for f in result.fields if f.key == params["field"])
        final, original = _squash(str(field.final_value or "")), _squash(str(field.original_value or ""))
        ok = field.status == params["status"] and re.search(params["final_pattern"], final, re.I) is not None
        if "original_pattern" in params:
            ok = ok and re.search(params["original_pattern"], original, re.I) is not None
        return ok, f"status={field.status} final={final!r} original={original!r}"

    if check == "no_cross_bid":
        chunks = await evidence_for(build_qa_agent(session, settings, None), bid, case["question_or_field"])
        foreign = {c.bid_id for c in chunks if c.bid_id != bid}
        leaked = [c.file_name for c in chunks if re.search(params.get("forbidden_text", "$^"), c.content, re.I)] if "forbidden_text" in params else []
        return not foreign and not leaked, f"{len(chunks)} chunks; foreign bids={sorted(foreign)}; forbidden text in {leaked}"

    if check == "absent_from_bid":
        found = (await session.execute(
            text("SELECT count(*) FROM document_chunks WHERE bid_id = :b AND content ~* :p"), {"b": bid, "p": params["pattern"]}
        )).scalar_one()  # fmt: skip
        return found == 0, f"{found} chunk(s) of {bid} match {params['pattern']!r}"

    if check == "validated_grounding":
        result = await ReconciliationService(retrieval, settings, None).reconcile_bid(bid)
        report = await ValidationAgent(db_chunk_lookup(session)).validate(bid, None, result)
        return report.valid and report.checked["citations"] > 0, f"valid={report.valid} citations={report.checked['citations']} errors={[i.code for i in report.errors]}"

    return False, f"unknown check {check}"


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    cases = load_cases()
    bad = {c["id"]: validate_case(c) for c in cases if validate_case(c)}
    if bad:
        print(f"Malformed cases: {bad}")
        return 2
    failures = 0
    try:
        async with get_sessionmaker()() as session:
            for case in cases:
                ok, detail = await run_case(case, session, settings)
                failures += not ok
                print(f"{'PASS' if ok else 'FAIL'}  [{case['_file'][:-5]:<15}] {case['id']:<34} {detail}")
    finally:
        await dispose_engine()
    print(f"\n{len(cases) - failures}/{len(cases)} regression cases hold")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main(parser.parse_args())))
