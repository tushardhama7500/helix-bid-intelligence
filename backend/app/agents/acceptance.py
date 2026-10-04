"""End-to-end acceptance checks for a finished pipeline run (Bid1 / Bid2).

Used by the tests (deterministic run, no Groq) and for real-model runs:

  python -m app.agents.acceptance --from-json eval/runs/pipeline_Bid1.json --bid Bid1 --require-extraction

`--require-extraction` additionally demands that real extraction ran and fed reconciliation.
The checks verify behaviour (values, provenance, grounding, isolation, stage outcomes), never prose.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from app.agents.schema import BidPipelineState

FINAL_DUE = "July 9, 2024 at 2:00 PM CST"
ORIGINAL_DUE = "27-JUN-2024 14:00:00"
FOREIGN_MARKERS = {"Bid1": ("PORFP", "Dell_Laptop", "Mercury_Affidavit", "Contract_Affidavit"), "Bid2": ("JA-207652", "Addendum")}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _sources(state: BidPipelineState) -> list:
    out = []
    if state.reconciliation:
        for f in state.reconciliation.fields:
            out += [*f.original_sources, *f.sources, *(x for c in f.changes for x in c.sources)]
    if state.extraction:
        for f in state.extraction.fields.values():
            out += [*f.sources, *(x for c in f.conflicting_values for x in c.sources)]
    return out


def _common(bid_id: str, state: BidPipelineState, require_extraction: bool) -> list[Check]:
    names = {d.file_name for d in (state.ingestion.documents if state.ingestion else [])}
    cited = _sources(state)
    foreign = [s.file_name for s in cited if s.file_name not in names or any(m in s.file_name for m in FOREIGN_MARKERS[bid_id])]
    stage = {n: s.state for n, s in state.stages.items()}
    checks = [
        Check("pipeline did not fail", not state.fatal and state.status != "failed", f"status={state.status} stages={stage}"),
        Check("validation passed", bool(state.validation and state.validation.valid), str([i.code for i in (state.validation.errors if state.validation else [])])),
        Check("every citation belongs to this bid", not foreign and (bool(cited) or not require_extraction), f"{len(cited)} citations, foreign={sorted(set(foreign))}"),
        Check("report was produced", state.report is not None and len(state.report.fields) == 20, ""),
    ]
    if require_extraction:
        extraction_ok = state.stages["extraction"].state == "complete" and state.extraction is not None and state.extraction.complete
        checks += [
            Check("real extraction completed", extraction_ok, f"extraction={stage['extraction']}"),
            Check("reconciliation ran after extraction", state.stages["reconciliation"].state in ("complete", "partial") and state.reconciliation is not None, f"reconciliation={stage['reconciliation']}"),
            Check("no required stage failed or was skipped", all(v in ("complete", "partial") for v in stage.values()), str(stage)),
        ]
    return checks


def check_bid1(state: BidPipelineState, require_extraction: bool = False) -> list[Check]:
    checks = _common("Bid1", state, require_extraction)
    due = next((f for f in state.reconciliation.fields if f.key == "due_date"), None) if state.reconciliation else None
    if due is None:
        return checks + [Check("due date reconciled", False, "no reconciliation")]
    final_src = due.sources[0] if due.sources else None
    orig_src = due.original_sources[0] if due.original_sources else None
    change = due.changes[0] if due.changes else None
    checks += [
        Check("final due date is Addendum 2's", due.final_value == FINAL_DUE and due.status == "resolved", f"final={due.final_value!r} status={due.status} method={due.resolution_method}"),
        Check("original due date preserved", due.original_value == ORIGINAL_DUE and change is not None and change.previous_value == ORIGINAL_DUE, f"original={due.original_value!r}"),
        Check("original cited to the RFP, page 2", bool(orig_src) and orig_src.file_name.endswith("FINAL.pdf") and orig_src.page_number == 2, str(orig_src and (orig_src.file_name[:30], orig_src.page_number))),
        Check("final cited to Addendum 2, page 1", bool(final_src) and final_src.file_name.startswith("Addendum 2") and final_src.page_number == 1 and final_src.document_type == "addendum", str(final_src and (final_src.file_name[:30], final_src.page_number))),
        Check("report shows the revised value with its original", _report_due(state) == (FINAL_DUE, ORIGINAL_DUE), str(_report_due(state))),
    ]
    if require_extraction and state.extraction is not None:
        ex_due = state.extraction.fields["Due Date"]
        checks.append(Check("extraction did not silently pick a date", ex_due.value is None and ex_due.status == "conflict", f"extraction status={ex_due.status}"))
    return checks


def _report_due(state: BidPipelineState):
    f = next((x for x in state.report.fields if x.field == "Due Date"), None) if state.report else None
    return (f.value, f.original_value) if f else None


def check_bid2(state: BidPipelineState, require_extraction: bool = False) -> list[Check]:
    checks = _common("Bid2", state, require_extraction)
    recon = state.reconciliation
    fields = recon.fields if recon else []
    checks += [
        Check("no addendum timeline fabricated", bool(recon) and recon.timeline.addenda == [] and recon.timeline.ordering == "none", f"addenda={len(recon.timeline.addenda) if recon else '?'}"),
        Check("no field changed by an addendum", not any(f.changes or f.status == "resolved" for f in fields), ""),
    ]
    values = [f for f in (state.report.fields if state.report else []) if f.value is not None]
    checks.append(Check("every reported value is cited", all(f.sources for f in values), f"{len(values)} values"))
    if require_extraction and state.extraction is not None:
        docs = state.extraction.fields["Any Additional Documentation Required"]
        text = json.dumps(docs.value).lower() if docs.value else ""
        files = {s.file_name for s in docs.sources}
        checks.append(Check("affidavits are Bid2's own", all(f.startswith(("PORFP", "Mercury_Affidavit", "Contract_Affidavit", "Dell")) for f in files) and "ja-207652" not in text, f"files={sorted(files)}"))
        checks.append(Check("missing information stays null", state.extraction.fields["Bid Bond Requirement"].value is None and state.extraction.fields["Pre Bid Meeting"].value is None, ""))
    return checks


def run_checks(bid_id: str, state: BidPipelineState, require_extraction: bool = False) -> list[Check]:
    return (check_bid1 if bid_id == "Bid1" else check_bid2)(state, require_extraction)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-json", required=True, help="a pipeline_<bid>.json written by app.agents.real_run")
    parser.add_argument("--bid", required=True, choices=["Bid1", "Bid2"])
    parser.add_argument("--require-extraction", action="store_true", help="also require a real extraction -> reconciliation run")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    state = BidPipelineState.model_validate(json.loads(Path(args.from_json).read_text(encoding="utf-8"))["state"])
    checks = run_checks(args.bid, state, args.require_extraction)
    for c in checks:
        print(f"{'PASS' if c.ok else 'FAIL'}  {c.name}  {c.detail}")
    failed = sum(not c.ok for c in checks)
    print(f"\n{len(checks) - failed}/{len(checks)} acceptance checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
