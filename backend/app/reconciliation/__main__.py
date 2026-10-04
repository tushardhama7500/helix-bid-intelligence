"""CLI: reconcile addenda for one or more bids.

  python -m app.reconciliation Bid1
  python -m app.reconciliation Bid1 Bid2 --json out.json
  python -m app.reconciliation Bid1 --use-extraction   # also run Phase 5 extraction first (needs Groq)

Deterministic rules run without any LLM. If GROQ_API_KEY and GROQ_MODEL are set,
Groq is used for ambiguous addendum wording; otherwise such fields are left unresolved.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from app.core.config import ConfigurationError, get_settings
from app.core.database import dispose_engine, get_sessionmaker
from app.extraction.service import ExtractionService, LLMProviderError
from app.llm.groq_client import GroqClient
from app.reconciliation.schema import BidReconciliation, ReconciledField
from app.reconciliation.service import ReconciliationService
from app.retrieval.service import RetrievalService


def _fmt(value) -> str:
    if value is None:
        return "(none)"
    return "; ".join(map(str, value)) if isinstance(value, list) else str(value)


def _where(f_sources) -> str:
    return "; ".join(
        f"{s.file_name[:48]} {('p.' + str(s.page_number)) if s.page_number is not None else 'html'}" for s in f_sources[:2]
    )


def print_field(f: ReconciledField) -> None:
    print(f"\nField: {f.key}")
    if f.status == "insufficient_evidence" and not f.changes and f.original_value is None:
        print(f"  Status: insufficient_evidence  ({f.notes})")
        return
    print(f"  Original:\n    {_fmt(f.original_value)}   <- {_where(f.original_sources)}")
    for c in f.changes:
        who = f"Addendum {c.addendum_number}" if c.addendum_number is not None else "Addendum (number unknown)"
        value = _fmt(c.new_value) if c.new_value is not None else f"({c.change_type})"
        print(f"  {who} [{c.change_type}, {c.method}]:\n    {value}   <- {_where(c.sources)}")
    if f.status == "conflict":
        print("  Final:\n    UNRESOLVED CONFLICT - authoritative value cannot be determined")
        for cv in f.conflicting_values:
            print(f"    * {_fmt(cv.value)}   <- {_where(cv.sources)}")
    elif f.status == "insufficient_evidence":
        print("  Final:\n    UNRESOLVED - authoritative value cannot be determined")
    else:
        print(f"  Final:\n    {_fmt(f.final_value)}   <- {_where(f.sources)}")
    print(f"  Status: {f.status}\n  Method: {f.resolution_method}   Confidence: {f.confidence}")
    if f.notes:
        print(f"  Notes: {f.notes}")


def print_bid(result: BidReconciliation) -> None:
    t = result.timeline
    print(f"\n{'=' * 70}\nBid: {result.bid_id}   (LLM used: {result.used_llm})")
    print(f"Timeline ({t.ordering}, reliable={t.reliable}): original documents -> " + (" -> ".join(
        f"Addendum {e.addendum_number if e.addendum_number is not None else '?'} [{e.file_name[:40]}]" for e in t.addenda) or "no addenda"))
    for note in t.notes:
        print(f"  note: {note}")
    shown = [f for f in result.fields if f.changes or f.original_value is not None or f.status == "conflict"]
    for f in shown:
        print_field(f)
    quiet = [f.key for f in result.fields if f not in shown]
    if quiet:
        print(f"\nNo original value or addendum change found for: {', '.join(quiet)}")
    if result.reconciliation_required:
        print(f"\nUNRESOLVED: {', '.join(result.reconciliation_required)}")
    for err in result.errors:
        print(f"ERROR: {err}")


async def run_eval(args: argparse.Namespace) -> int:
    from app.reconciliation.evaluation import check_case, load_cases, summarize

    settings, cases = get_settings(), load_cases()
    try:
        async with get_sessionmaker()() as session:
            retrieval = RetrievalService.from_settings(session, settings)
            service = ReconciliationService(retrieval, settings, None)  # deterministic path only
            results = []
            for bid in dict.fromkeys(c["bid"] for c in cases):
                reconciled = await service.reconcile_bid(bid)
                for case in [c for c in cases if c["bid"] == bid]:
                    results.append(await check_case(case, reconciled, session))
    finally:
        await dispose_engine()
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'} {r.id}  {r.detail[0]}")
        for name, ok in r.checks.items():
            print(f"      {name}: {'n/a' if ok is None else ok}")
    print("\nMetrics (passed/total):")
    for name, m in summarize(results).items():
        print(f"  {name:<22} {m['passed']}/{m['total']}")
    return 0 if all(r.passed for r in results) else 1


async def run_llm_eval(args: argparse.Namespace) -> int:
    from app.reconciliation.llm_eval import load_cases, run_case

    settings = get_settings()
    try:
        llm = GroqClient.from_settings(settings)
    except ConfigurationError as exc:
        print(f"Cannot run: {exc}")
        return 2
    rows = []
    for case in load_cases():
        try:
            row = await run_case(case, llm, settings)
        except LLMProviderError as exc:  # e.g. a Groq rate limit after the one bounded wait
            row = {"id": case["id"], "ok": False, "problems": [f"provider error: {exc}"], "status": "error", "method": "-", "final": None, "llm_used": True, "detail": {}}
        rows.append(row)
        print(f"{'PASS' if row['ok'] else 'FAIL'} {row['id']:<26} status={row['status']:<22} method={row['method']:<26} llm_used={row['llm_used']} final={str(row['final'])[:50]!r}")
        print(f"       detail={row['detail']}")
        for problem in row["problems"]:
            print(f"       PROBLEM: {problem}")
    ok = sum(r["ok"] for r in rows)
    secs = [c.seconds for c in llm.calls]
    print(f"\n{ok}/{len(rows)} cases as expected; Groq calls={len(llm.calls)} (failed {sum(not c.ok for c in llm.calls)}), "
          f"latency avg/min/max={sum(secs)/max(len(secs),1):.1f}/{min(secs, default=0):.1f}/{max(secs, default=0):.1f}s, "
          f"tokens in/out={sum(c.prompt_tokens or 0 for c in llm.calls)}/{sum(c.completion_tokens or 0 for c in llm.calls)}")
    return 0 if ok == len(rows) else 1


async def main(args: argparse.Namespace) -> int:
    if args.llm_eval:
        return await run_llm_eval(args)
    if args.eval:
        return await run_eval(args)
    if not args.bids:
        print("Give at least one bid id, or use --eval")
        return 2
    settings = get_settings()
    logging.basicConfig(level="WARNING")
    llm = None
    try:
        llm = GroqClient.from_settings(settings)
    except ConfigurationError:
        print("(Groq is not configured: deterministic reconciliation only)")
    out = {}
    try:
        async with get_sessionmaker()() as session:
            retrieval = RetrievalService.from_settings(session, settings)
            for bid in args.bids:
                extraction = None
                if args.use_extraction:
                    if llm is None:
                        print("--use-extraction needs GROQ_API_KEY and GROQ_MODEL")
                        return 2
                    extraction = await ExtractionService(retrieval, llm, settings).extract_bid(bid)
                result = await ReconciliationService(retrieval, settings, llm).reconcile_bid(bid, extraction)
                print_bid(result)
                out[bid] = result.model_dump()
    finally:
        await dispose_engine()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bids", nargs="*")
    parser.add_argument("--json")
    parser.add_argument("--use-extraction", action="store_true")
    parser.add_argument("--llm-eval", action="store_true", help="real-Groq fallback on controlled ambiguous cases")
    parser.add_argument("--eval", action="store_true", help="score against eval/reconciliation_eval.json (ignores bids)")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # document text can contain non-cp1252 characters
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main(parser.parse_args())))
