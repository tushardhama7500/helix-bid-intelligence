"""Extraction evaluation against hand-verified ground truth (eval/extraction_eval.json).

Two independent measurements:
  * evidence coverage  - does the retrieved evidence for the field contain the expected
    text? Needs no LLM; it isolates retrieval quality from model quality.
  * extraction scoring - does the validated extraction match the ground truth?
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.extraction.evidence import gather_evidence
from app.extraction.fields import FIELDS
from app.extraction.schema import BidExtraction
from app.extraction.validation import flatten
from app.retrieval.service import RetrievalService

DEFAULT_GROUND_TRUTH = Path(__file__).resolve().parents[2] / "eval" / "extraction_eval.json"
_KEY_BY_NAME = {f.name: f.key for f in FIELDS}


def load_ground_truth(path: Path = DEFAULT_GROUND_TRUTH) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _squash(text: str) -> str:
    return " ".join(text.split())


def _all_match(patterns: list[str], text: str) -> bool:
    return all(re.search(p, text, re.IGNORECASE) for p in patterns)


@dataclass
class CheckResult:
    id: str
    field: str
    expect: str
    scored: bool
    passed: bool
    status: str | None  # extraction status (None for evidence-only runs)
    detail: str


def score_extraction(extraction: BidExtraction, checks: list[dict[str, Any]]) -> list[CheckResult]:
    results = []
    for check in checks:
        field = extraction.fields[check["field"]]
        patterns = check.get("patterns", [])
        expect = check["expect"]
        if expect == "not_found":
            passed = field.value is None and field.status != "error"
            detail = f"status={field.status}, value={'null' if field.value is None else _squash(flatten(field.value))[:80]}"
        elif expect == "conflict":
            text = _squash(" ".join(flatten(c.value) for c in field.conflicting_values))
            passed = field.status == "conflict" and _all_match(patterns, text)
            detail = f"status={field.status}, values={[_squash(flatten(c.value))[:40] for c in field.conflicting_values]}"
        else:
            text = _squash(flatten(field.value))
            passed = field.status == "found" and bool(field.sources) and _all_match(patterns, text)
            detail = f"status={field.status}, value={text[:90] or 'null'}"
        results.append(CheckResult(check["id"], check["field"], expect, check.get("kind") != "info", passed, field.status, detail))
    return results


async def evidence_coverage(
    retrieval: RetrievalService, settings: Settings, bid_id: str, checks: list[dict[str, Any]]
) -> list[CheckResult]:
    """Is the expected text present in the evidence retrieved for each field? (no LLM involved)"""
    evidence = await gather_evidence(retrieval, bid_id, settings)
    results = []
    for check in checks:
        if check["expect"] == "not_found":
            continue
        chunks = evidence[_KEY_BY_NAME[check["field"]]]
        text = _squash(" ".join(c.content for c in chunks))
        found = [bool(re.search(p, text, re.IGNORECASE)) for p in check["patterns"]]
        results.append(
            CheckResult(
                check["id"], check["field"], check["expect"], check.get("kind") != "info", all(found), None,
                f"{len(chunks)} chunks; patterns found: {found}",
            )
        )
    return results


def summarize(results: list[CheckResult]) -> dict[str, Any]:
    scored = [r for r in results if r.scored]
    passed = sum(r.passed for r in scored)
    return {"scored": len(scored), "passed": passed, "rate": passed / len(scored) if scored else 0.0}
