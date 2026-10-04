"""Real-Groq evaluation of the reconciliation fallback on controlled ambiguous cases.

`python -m app.reconciliation --llm-eval` (needs GROQ_API_KEY / GROQ_MODEL). Each case supplies an
original chunk and an addendum chunk, runs the deterministic detector, and - when it finds the
wording ambiguous - the real Groq fallback, then the normal reconciliation core. It reports whether
the outcome is the expected one and whether the model ever produced a value that is not allowed.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.extraction.fields import FIELDS_BY_KEY
from app.extraction.schema import Source
from app.extraction.service import LLMClient
from app.extraction.validation import build_source, flatten, grounding
from app.reconciliation.addenda import build_event, build_timeline
from app.reconciliation.detect import detect_changes
from app.reconciliation.fields import RECONCILABLE_BY_KEY
from app.reconciliation.llm import interpret_with_llm
from app.reconciliation.reconcile import ChangeRecord, OriginalValue, reconcile_field
from app.retrieval.schema import Candidate

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "eval" / "reconciliation_llm_cases.json"


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def _chunk(chunk_id: int, doc_id: int, file: str, doc_type: str, text: str) -> Candidate:
    return Candidate(
        chunk_id=chunk_id, content=text, document_id=doc_id, bid_id="Eval", file_name=file, document_type=doc_type,
        page_number=1, chunk_index=0, metadata={}, score=0.0,
    )  # fmt: skip


async def run_case(case: dict[str, Any], llm: LLMClient, settings: Settings) -> dict[str, Any]:
    key = case["field"]
    rf, spec = RECONCILABLE_BY_KEY[key], FIELDS_BY_KEY[key]
    original = _chunk(1, 1, "RFP.pdf", "rfp", case["original"])
    addendum = _chunk(2, 2, "Addendum 3.pdf", "addendum", case["addendum"])
    event = build_event(document_id=2, file_name="Addendum 3.pdf", document_type="addendum", bid_id="Eval",
                        first_page=case["addendum"], full_text=case["addendum"])  # fmt: skip
    timeline = build_timeline("Eval", [event], ["RFP.pdf"])

    detected = detect_changes(addendum.content, rf.terms)
    records: list[ChangeRecord] = [
        ChangeRecord(
            change_type=d.change_type, explanation=d.sentence, sources=[build_source(addendum)], addendum_number=3,
            addendum_file="Addendum 3.pdf", document_id=2, order=event.order, previous_value=d.previous_value,
            new_value=d.new_value,
        )
        for d in detected
    ]  # fmt: skip
    llm_used = False
    detail: dict[str, Any] = {"deterministic": [(d.change_type, d.new_value) for d in detected]}
    if not detected or any(r.change_type == "unknown" for r in records):
        llm_used = True
        before = len(llm.calls) if hasattr(llm, "calls") else 0
        llm_changes, decided, error = await interpret_with_llm(
            llm, field_name=spec.name, description=spec.description, kind=rf.kind, original=[original],
            addendum_chunks={2: (event, [addendum])}, char_limit=settings.extraction_chunk_char_limit,
            max_tokens=settings.extraction_max_output_tokens, sleep=asyncio.sleep,
            strong_docs={2} if any(r.change_type == "unknown" for r in records) else set(),
        )  # fmt: skip
        detail["llm"] = [(c.change_type, c.new_value, c.llm_confidence) for c in llm_changes]
        detail["llm_error"] = error
        detail["llm_calls"] = (len(llm.calls) - before) if hasattr(llm, "calls") else None
        if not error:
            records = [r for r in records if not (r.change_type == "unknown" and r.document_id in decided)] + llm_changes

    result = reconcile_field(key, spec.name, [OriginalValue(case["original"], [build_source(original)])], records, timeline, rf.kind)
    final = flatten(result.final_value)
    expect, problems = case["expect"], []
    if "status" in expect and result.status != expect["status"]:
        problems.append(f"status {result.status} != {expect['status']}")
    if "final_pattern" in expect and not re.search(re.escape(expect["final_pattern"]), final, re.I):
        problems.append(f"final {final!r} does not contain {expect['final_pattern']!r}")
    if expect.get("no_invented_value") and result.status == "resolved":
        problems.append(f"resolved to {final!r} although the addendum gives no value")
    for bad in case.get("forbid_final", []):
        if bad.lower() in final.lower():
            problems.append(f"forbidden value {bad!r} in final")
    if result.final_value is not None and result.status == "resolved":
        cited = [addendum] if result.sources else []
        if not result.sources or grounding(result.final_value, cited, "scalar") is None:
            problems.append("final value is not grounded in its citation")
    return {"id": case["id"], "ok": not problems, "problems": problems, "status": result.status, "method": result.resolution_method,
            "final": final or None, "llm_used": llm_used, "detail": detail}  # fmt: skip
