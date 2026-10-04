"""CLI: extract bids and evaluate against ground truth.

  python -m app.extraction Bid1 Bid2            # extract with Groq and score against ground truth
  python -m app.extraction --evidence-only      # no LLM: is the needed text in the retrieved evidence?
  python -m app.extraction Bid1 --json out.json # also write the full extraction

Needs GROQ_API_KEY and GROQ_MODEL (except with --evidence-only).
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from app.core.config import ConfigurationError, get_settings
from app.core.database import dispose_engine, get_sessionmaker
from app.extraction.evaluation import evidence_coverage, load_ground_truth, score_extraction, summarize
from app.agents.validation_agent import ValidationAgent, db_chunk_lookup
from app.extraction.service import ExtractionService
from app.llm.groq_client import GroqClient
from app.retrieval.service import RetrievalService


def _sources(field) -> str:
    srcs = field.sources
    return "; ".join(
        f"{s.file_name[:34]} {('p.' + str(s.page_number)) if s.page_number is not None else 'html'}" for s in srcs[:3]
    )


def print_extraction(extraction) -> None:
    print(f"\n=== {extraction.bid_id}  (model: {extraction.model}, complete: {extraction.complete})")
    for name, f in extraction.fields.items():
        if f.status == "conflict":
            print(f"- {name} [CONFLICT, conf 0.0] reconciliation required")
            for c in f.conflicting_values:
                print(f"    * {str(c.value)[:80]}  <- {_sources(c)}")
        else:
            value = str(f.value)[:110] if f.value is not None else None
            print(f"- {name} [{f.status}, conf {f.confidence}] {value}  <- {_sources(f)}")
            if f.status in ("unsupported", "error") and f.notes:
                print(f"    note: {f.notes[:140]}")
    for err in extraction.errors:
        print(f"  ERROR: {err}")


def print_llm_stats(llm, elapsed: float) -> None:
    calls = llm.calls
    ok = [c for c in calls if c.ok]
    tin = sum(c.prompt_tokens or 0 for c in ok)
    tout = sum(c.completion_tokens or 0 for c in ok)
    secs = [c.seconds for c in calls]
    print(f"\nGroq: {len(calls)} calls ({len(calls) - len(ok)} failed), tokens in/out={tin}/{tout}, "
          f"call latency avg/min/max={sum(secs)/max(len(secs),1):.1f}/{min(secs, default=0):.1f}/{max(secs, default=0):.1f}s, "
          f"total {elapsed:.1f}s, prompt chars={[c.prompt_chars for c in calls]}, "
          f"finish={[c.finish_reason for c in ok]}")


def print_results(title: str, results) -> None:
    print(f"\n--- {title}")
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        info = "" if r.scored else " (info, not scored)"
        print(f"{mark} {r.id:<24} {r.field:<38} {r.detail}{info}")
    s = summarize(results)
    print(f"=> {s['passed']}/{s['scored']} scored checks passed ({s['rate']:.2f})")


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    logging.basicConfig(level="WARNING")
    truth = load_ground_truth()
    bids = args.bids or list(truth)
    out: dict[str, object] = {}
    try:
        llm = None
        if not args.evidence_only:
            try:
                llm = GroqClient.from_settings(settings)
            except ConfigurationError as exc:
                print(f"Cannot run extraction: {exc} Set it in backend/.env, or use --evidence-only.")
                return 2
        async with get_sessionmaker()() as session:
            retrieval = RetrievalService.from_settings(session, settings)
            for bid in bids:
                checks = truth.get(bid, [])
                coverage = await evidence_coverage(retrieval, settings, bid, checks)
                print_results(f"{bid}: evidence coverage (retrieval only, no LLM)", coverage)
                out[bid] = {"evidence_coverage": [r.__dict__ for r in coverage]}
                if llm is None:
                    continue
                llm.reset_stats()
                started = time.perf_counter()
                extraction = await ExtractionService(retrieval, llm, settings).extract_bid(bid)
                elapsed = time.perf_counter() - started
                print_extraction(extraction)
                print_llm_stats(llm, elapsed)
                print(f"Retries: malformed-output={extraction.schema_retries}, citation-follow-ups={extraction.citation_retries}")
                validation = await ValidationAgent(db_chunk_lookup(session)).validate(bid, extraction, None)
                found = sum(1 for f in extraction.fields.values() if f.status == "found")
                print(f"\nDatabase validation: valid={validation.valid}, citations checked={validation.checked['citations']}, "
                      f"found fields={found}, errors={[i.code + ':' + str(i.field) for i in validation.errors]}")
                scored = score_extraction(extraction, checks)
                print_results(f"{bid}: extraction vs ground truth", scored)
                out[bid].update(extraction=extraction.model_dump(), scores=[r.__dict__ for r in scored])
    finally:
        await dispose_engine()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bids", nargs="*", help="bid ids (default: all in the ground truth)")
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--json")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # document text can contain non-cp1252 characters
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main(parser.parse_args())))
