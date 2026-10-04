"""Export the submission artifacts for a bid, with no Groq call.

  python -m app.agents.export_submission --extraction Bid1=PATH --extraction Bid2=PATH \
      --meta Bid1=PATH --out ../outputs --trace ../docs/agent_trace.md --trace-bid Bid1

For each bid this runs the real pipeline (ingestion check, retrieval, reconciliation, validation, report) against
the real database. The extraction stage *replays* a stored extraction (the JSON written by
`python -m app.extraction BID --json FILE` after a live Groq run) instead of calling Groq, so nothing is invented
and no quota is used. Every output file records that provenance. To produce fully live output instead, run
`python -m app.agents BID --json FILE` with Groq configured.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from app.agents.extraction_agent import ReconciliationAgent
from app.agents.factory import build_orchestrator
from app.agents.schema import BidPipelineState
from app.agents.tracing import agent_span
from app.core.config import get_settings
from app.core.database import dispose_engine, get_sessionmaker
from app.extraction.schema import BidExtraction


class ReplayExtractionAgent:
    """Stands in for `ExtractionAgent`: returns a stored extraction instead of calling Groq."""

    available = True
    service = None

    def __init__(self, extraction: BidExtraction) -> None:
        self.extraction = extraction

    async def run(self, bid_id: str, evidence=None) -> BidExtraction:
        async with agent_span("extraction", bid_id, "extract") as span:
            span["note"] = "replayed from a stored live run; no Groq call"
            return self.extraction


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(f"{record.name}: {record.getMessage()}")


def _src(s: Any) -> dict[str, Any]:
    return {"file": s.file_name, "document_type": s.document_type, "page": s.page_number, "section": s.section,
            "chunk_id": s.chunk_id, "chunk_index": s.chunk_index}  # fmt: skip


def submission_json(bid: str, state: BidPipelineState, provenance: dict[str, Any]) -> dict[str, Any]:
    """One bid's deliverable: values, confidence, citations, reconciliation and validation."""
    assert state.report and state.validation and state.extraction and state.reconciliation
    fields = []
    for f in state.report.fields:
        item: dict[str, Any] = {
            "field": f.field, "value": f.value, "status": f.status, "confidence": f.confidence,
            "citations": [_src(s) for s in f.sources], "notes": f.notes,
        }  # fmt: skip
        if f.changed_by_addendum:
            item["reconciliation"] = {"original_value": f.original_value, "original_citations": [_src(s) for s in f.original_sources]}
        fields.append(item)
    changes = [
        {"field": r.field, "original_value": r.original_value, "final_value": r.final_value, "status": r.status,
         "resolution_method": r.resolution_method, "confidence": r.confidence,
         "changes": [{"addendum": c.addendum_file, "addendum_number": c.addendum_number, "type": c.change_type,
                      "previous_value": c.previous_value, "new_value": c.new_value, "explanation": c.explanation,
                      "method": c.method, "citations": [_src(s) for s in c.sources]} for c in r.changes]}
        for r in state.reconciliation.fields if r.changes
    ]  # fmt: skip
    ext = state.extraction
    return {
        "bid_id": bid,
        "provenance": provenance,
        "pipeline_status": state.status,
        "stages": {n: s.state for n, s in state.stages.items()},
        "fields": fields,
        "unresolved_fields": state.report.unresolved,
        "extraction_conflicts_before_reconciliation": ext.reconciliation_required,
        "addendum_timeline": [
            {"order": a.order, "number": a.addendum_number, "file": a.file_name, "identified_by": a.identified_by}
            for a in state.reconciliation.timeline.addenda
        ],
        "reconciled_changes": changes,
        "validation": {"valid": state.validation.valid, "checked": state.validation.checked,
                       "errors": [i.code for i in state.validation.errors],
                       "warnings": [i.code for i in state.validation.warnings]},  # fmt: skip
    }


def _short(value: Any, n: int = 90) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= n else text[: n - 3] + "..."


def _where(sources) -> str:
    return "; ".join(f"{s.file_name} {('p.' + str(s.page_number)) if s.page_number is not None else 'html'} (chunk {s.chunk_id})" for s in sources)


def render_trace(state: BidPipelineState, logs: list[str], live_meta: list[str], extraction_file: str) -> str:
    st, ext, rec, val = state.stages, state.extraction, state.reconciliation, state.validation
    ing, ret = state.ingestion, state.retrieval
    out = [f"# Agent trace: `{state.bid_id}`, run `{state.run_id}`", ""]
    out += [
        "This trace is generated by `python -m app.agents.export_submission` from the pipeline state of one real run against the real "
        "database. **Which parts were live and which were replayed:**", "",
        "- Orchestrator, ingestion check, retrieval, reconciliation, validation and report: **executed live** in this run (no Groq; reconciliation used deterministic rules only).",
        f"- Extraction: **replayed** from `{extraction_file}`, the output of an earlier live Groq run (`{ext.model}`). No Groq call was made in this run, so this run's own log has no model calls. The live run's call metadata is quoted verbatim below.",
        "- That earlier extraction run predates the latest Bid Number and multi-product Product Specification changes, so its Bid Number and Product Specification values reflect the older behaviour.",
        "",
        "Timings are this run's measured stage durations. Nothing else (IDs, counts) is estimated.", "",
    ]
    out += ["## 1. Orchestrator", f"- input: `bid_id={state.bid_id}`, ingestion enabled", f"- run id: `{state.run_id}`",
            f"- final status: **{state.status}**; stages: " + ", ".join(f"{n}={s.state}" for n, s in st.items()),
            f"- total: {state.timings_ms.get('total')} ms", ""]  # fmt: skip
    out += ["## 2. Ingestion Agent", f"- status: **{st['ingestion'].state}** ({st['ingestion'].duration_ms} ms)",
            "- action: scan the bid folder, skip unchanged files (SHA-256), embed any chunk without an embedding",
            f"- output: files_found={ing.files_found}, ingested={ing.ingested}, skipped={ing.skipped}, chunks_embedded={ing.chunks_embedded}", "",
            "| document | type | status |", "| --- | --- | --- |"]  # fmt: skip
    out += [f"| {d.file_name} | {d.document_type} | {d.status} |" for d in ing.documents] + [""]
    out += ["## 3. Retrieval Agent: evidence selection",
            f"- status: **{st['retrieval'].state}** ({st['retrieval'].duration_ms} ms)",
            "- action: for each of the 20 fields run its hybrid (vector + keyword, RRF) queries, filtered to this bid; amendable fields also fetch addendum chunks",
            f"- output: {ret.total_chunks} evidence chunks across {ret.fields} fields; empty fields: {ret.empty_fields or 'none'}; chunks from other bids discarded: {ret.dropped_foreign_chunks}", "",
            "| field | evidence chunks |", "| --- | --- |"]  # fmt: skip
    out += [f"| {k} | {v} |" for k, v in ret.evidence_counts.items()] + [""]
    out += ["## 4. Extraction Agent (replayed)", f"- status: **{st['extraction'].state}**",
            f"- model recorded in the stored run: `{ext.model}`; complete={ext.complete}; schema_retries={ext.schema_retries}; citation_retries={ext.citation_retries}; errors={ext.errors or 'none'}",
            "- live-run metadata, quoted verbatim from the stored run's output:"]  # fmt: skip
    out += [f"  - `{line.strip()}`" for line in live_meta] or ["  - (not available)"]
    out += ["", "| field | status | confidence | value (shortened) | citations |", "| --- | --- | --- | --- | --- |"]
    for name, f in ext.fields.items():
        value = "null" if f.value is None else _short(f.value).replace("|", "/")
        out.append(f"| {name} | {f.status} | {f.confidence} | {value} | {len(f.sources)} |")
    conflicts = [(n, f) for n, f in ext.fields.items() if f.status == "conflict"]
    for name, f in conflicts:
        out += ["", f"Extraction does not choose between conflicting values. **{name}** candidates:"]
        out += [f"- `{_short(c.value)}` <- {_where(c.sources)}" for c in f.conflicting_values]
    out.append("")
    out += ["## 5. Reconciliation", f"- status: **{st['reconciliation'].state}** ({st['reconciliation'].duration_ms} ms)",
            f"- addenda found: {len(rec.timeline.addenda)}; ordering: {rec.timeline.ordering}; used LLM: {rec.used_llm}"]  # fmt: skip
    out += [f"  - order {a.order}: addendum {a.addendum_number} `{a.file_name}` (identified by {a.identified_by})" for a in rec.timeline.addenda]
    for r in rec.fields:
        if not r.changes:
            continue
        out += ["", f"**{r.field}**: status `{r.status}` via `{r.resolution_method}`, confidence {r.confidence}",
                f"- original: `{_short(r.original_value)}` <- {_where(r.original_sources)}"]  # fmt: skip
        for c in r.changes:
            out += [f"- change by addendum {c.addendum_number} (`{c.change_type}`, method `{c.method}`): {_short(c.explanation, 160)}",
                    f"  - citations: {_where(c.sources)}"]  # fmt: skip
        out.append(f"- final: `{_short(r.final_value)}` <- {_where(r.sources)}")
    unresolved = rec.reconciliation_required
    out += ["", f"- unresolved after reconciliation: {unresolved or 'none'}", ""]
    out += ["## 6. Validation Agent", f"- status: **{st['validation'].state}** ({st['validation'].duration_ms} ms)",
            "- action: re-read every cited chunk from the database; check it exists, belongs to this bid, matches its file/page, and contains the value",
            f"- output: valid={val.valid}; checked={val.checked}; errors={[i.code for i in val.errors] or 'none'}; warnings={[i.code for i in val.warnings] or 'none'}", ""]  # fmt: skip
    changed = [f for f in state.report.fields if f.changed_by_addendum]
    out += ["## 7. Final report", f"- {len(state.report.fields)} fields; {sum(f.value is not None for f in state.report.fields)} with a value; changed by an addendum: {[f.field for f in changed] or 'none'}; unresolved: {state.report.unresolved or 'none'}", ""]
    for f in changed:
        out += [f"- **{f.field}** = `{_short(f.value)}` (confidence {f.confidence}); original `{_short(f.original_value)}`", f"  - final cited to {_where(f.sources)}", f"  - original cited to {_where(f.original_sources)}"]
    out += ["", "## 8. Raw log lines captured during this run (verbatim)", "", "```"] + logs + ["```", ""]
    return "\n".join(out)


def _pairs(items: list[str]) -> dict[str, Path]:
    return {k: Path(v) for k, v in (i.split("=", 1) for i in items)}


async def main(args: argparse.Namespace) -> int:
    settings = get_settings().model_copy(update={"groq_api_key": None})  # never call Groq here
    extractions, metas = _pairs(args.extraction), _pairs(args.meta or [])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = _Capture()
    for name in ("helix.agents", "app.agents.orchestrator"):
        logging.getLogger(name).addHandler(capture)
        logging.getLogger(name).setLevel(logging.INFO)
    try:
        async with get_sessionmaker()() as session:
            for bid, path in extractions.items():
                capture.lines.clear()
                stored = json.loads(path.read_text(encoding="utf-8"))
                if bid in stored:  # `python -m app.extraction --json` wraps the result per bid
                    stored = stored[bid]["extraction"]
                extraction = BidExtraction.model_validate(stored)
                orchestrator = build_orchestrator(session, settings, None)
                orchestrator.extraction = ReplayExtractionAgent(extraction)
                state = await orchestrator.run(bid, ingest=True)
                if state.status == "failed" or not state.report:
                    print(f"{bid}: pipeline failed: {state.errors}")
                    return 1
                provenance = {
                    "extraction": f"replayed from a stored live Groq run ({extraction.model}); {path.name}",
                    "other_stages": "executed live against the database by this export; reconciliation used deterministic rules only (no Groq)",
                    "note": "the stored extraction predates the latest Bid Number and multi-product Product Specification changes",
                }
                (out_dir / f"{bid}.json").write_text(
                    json.dumps(submission_json(bid, state, provenance), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
                )
                print(f"{bid}: {state.status}, run {state.run_id} -> {out_dir / (bid + '.json')}")
                if args.trace and bid == args.trace_bid:
                    meta_lines = []
                    if bid in metas:
                        meta_lines = [ln for ln in metas[bid].read_text(encoding="utf-8").splitlines()
                                      if re.match(r"(Groq:|Retries:|Database validation:)", ln)]  # fmt: skip
                    Path(args.trace).write_text(render_trace(state, list(capture.lines), meta_lines, path.name), encoding="utf-8")
                    print(f"trace -> {args.trace}")
    finally:
        await dispose_engine()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extraction", action="append", required=True, metavar="BID=FILE", help="stored extraction JSON per bid")
    parser.add_argument("--meta", action="append", metavar="BID=FILE", help="stored run's text output (for the trace)")
    parser.add_argument("--out", default="../outputs", help="directory for <Bid>.json")
    parser.add_argument("--trace", help="write a Markdown agent trace here")
    parser.add_argument("--trace-bid", default="Bid1")
    ns = parser.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main(ns)))
