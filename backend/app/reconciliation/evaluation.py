"""Reconciliation evaluation against hand-checked cases (eval/reconciliation_eval.json).

Four things are measured separately:
  resolution accuracy      - expected-resolved cases that resolve to the expected value by the expected method
  conflict preservation    - expected-conflict cases that stay unresolved (n/a when the file has none)
  citation correctness     - cited chunks exist, come from the expected files, and contain the value
  original-value preservation - the original value (and its citation) survives in the history
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.reconciliation.schema import BidReconciliation, ReconciledField

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "eval" / "reconciliation_eval.json"


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


@dataclass
class CaseResult:
    id: str
    checks: dict[str, bool | None] = field(default_factory=dict)  # None = not applicable
    detail: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(v is not False for v in self.checks.values())


def _squash(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


async def _cited_text(session: AsyncSession, chunk_ids: list[int]) -> list[str | None]:
    out = []
    for chunk_id in chunk_ids:
        out.append(await session.scalar(text("SELECT content FROM document_chunks WHERE id = :i"), {"i": chunk_id}))
    return out


async def check_case(case: dict[str, Any], result: BidReconciliation, session: AsyncSession) -> CaseResult:
    expect, res = case["expect"], CaseResult(case["id"])

    if case["field"] == "*":  # timeline / whole-bid expectations
        numbers = [e.addendum_number for e in result.timeline.addenda]
        res.checks["timeline"] = numbers == expect["addendum_numbers_in_order"] and result.timeline.ordering == expect["ordering"]
        res.detail.append(f"addenda={numbers}, ordering={result.timeline.ordering}")
        if expect.get("no_field_changed"):
            res.checks["no_field_changed"] = not any(f.changes or f.status == "resolved" for f in result.fields)
        return res

    f: ReconciledField = next(x for x in result.fields if x.key == case["field"])
    res.detail.append(f"status={f.status}, method={f.resolution_method}, final={_squash(f.final_value)!r}")

    status_ok = f.status == expect["status"] and (expect.get("method") is None or f.resolution_method == expect["method"])
    if expect["status"] == "resolved":
        res.checks["resolution"] = status_ok and bool(re.search(re.escape(expect["final_pattern"]), _squash(f.final_value)))
        res.checks["addenda"] = [c.addendum_number for c in f.changes if c.change_type == "replacement"] == expect["addendum_numbers"]
        res.checks["conflict_preservation"] = None
    elif expect["status"] == "conflict":
        res.checks["conflict_preservation"] = status_ok and f.final_value is None
        res.checks["resolution"] = None
    else:  # unchanged
        res.checks["unchanged"] = status_ok and bool(re.search(expect["final_pattern"], _squash(f.final_value)))

    if "original_pattern" in expect:
        kept = bool(re.search(expect["original_pattern"], _squash(f.original_value)))
        if f.changes:  # the history must start from the original value
            kept = kept and any(_squash(c.previous_value) == _squash(f.original_value) for c in f.changes)
        res.checks["original_preserved"] = kept
        res.checks["original_cited"] = (
            bool(f.original_sources) and all(s.file_name in expect.get("original_files", [s.file_name]) for s in f.original_sources)
        )  # fmt: skip

    # Citation correctness: chunks exist, are from the expected files, and contain the value.
    if f.final_value is not None:
        texts = await _cited_text(session, [s.chunk_id for s in f.sources])
        value_in_text = all(t is not None for t in texts) and any(_squash(f.final_value) in _squash(t) for t in texts if t)
        files_ok = all(s.file_name in expect.get("final_files", [s.file_name]) for s in f.sources) and bool(f.sources)
        res.checks["citation_correct"] = value_in_text and files_ok
    return res


def summarize(results: list[CaseResult]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    names = {k for r in results for k in r.checks}
    for name in sorted(names):
        applicable = [r.checks[name] for r in results if r.checks.get(name) is not None]
        out[name] = {"passed": sum(applicable), "total": len(applicable)}
    return out
