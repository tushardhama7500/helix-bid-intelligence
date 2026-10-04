"""CLI: run the bid pipeline, and optionally ask a question.

  python -m app.agents Bid1
  python -m app.agents Bid1 --ask "What is the final submission deadline?"
  python -m app.agents Bid1 --json out.json --no-ingest

Works without Groq (extraction is skipped and the status is `partial`); `--ask` needs Groq.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from app.agents.factory import build_orchestrator, build_qa_agent, optional_groq
from app.agents.orchestrator import Orchestrator
from app.agents.report_qa_agent import QAOutputError
from app.core.config import get_settings
from app.core.database import dispose_engine, get_sessionmaker
from app.extraction.service import LLMProviderError, UnknownBidError


def _where(sources) -> str:
    return "; ".join(f"{s.file_name[:46]} {('p.' + str(s.page_number)) if s.page_number is not None else 'html'}" for s in sources[:2])


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    logging.basicConfig(level="WARNING")
    llm = optional_groq(settings)
    try:
        async with get_sessionmaker()() as session:
            try:
                state = await build_orchestrator(session, settings, llm).run(args.bid, ingest=not args.no_ingest)
            except UnknownBidError as exc:
                print(exc)
                return 2
            print(f"Bid: {state.bid_id}   status: {state.status.upper()}   (Groq: {'configured' if llm else 'not configured'})")
            for name, stage in state.stages.items():
                mark = {"complete": "OK ", "partial": "~  ", "failed": "ERR", "skipped": "-  "}[stage.state]
                print(f"  [{mark}] {name:<15} {stage.state:<8} {stage.duration_ms:>6} ms  {stage.detail if stage.detail else ''}")
            for e in state.errors:
                print(f"  ERROR:   {e}")
            for w in state.warnings:
                print(f"  WARNING: {w}")
            if state.report:
                print("\nFields with a value:")
                for f in state.report.fields:
                    if f.value is not None:
                        extra = f"  (original: {f.original_value})" if f.changed_by_addendum else ""
                        print(f"  {f.field:<40} {str(f.value)[:70]}{extra}  <- {_where(f.sources)}")
                if state.report.unresolved:
                    print(f"\nUnresolved: {', '.join(state.report.unresolved)}")
            elif state.validation and not state.validation.valid:
                print("\nValidation failed; report withheld:")
                for i in state.validation.errors:
                    print(f"  {i.code}: {i.field}: {i.message}")
            if args.json:
                Path(args.json).write_text(json.dumps(state.model_dump(), indent=2, default=str), encoding="utf-8")

            if args.ask:
                if llm is None:
                    print("\n--ask needs GROQ_API_KEY and GROQ_MODEL")
                    return 2
                try:
                    answer = await build_qa_agent(session, settings, llm).answer(args.bid, args.ask)
                except (LLMProviderError, QAOutputError) as exc:
                    print(f"\nCould not answer: {exc}")  # a structured message, not a traceback
                    return 1
                print(f"\nQ: {answer.question}\nA: {answer.answer}")
                for s in answer.sources:
                    print(f"   source: {s.file} {('p.' + str(s.page)) if s.page is not None else 'html'} (chunk {s.chunk_id})")
                print(f"   confidence: {answer.confidence}" + (f"   note: {answer.notes}" if answer.notes else ""))
    finally:
        await dispose_engine()
    return 0 if state.status != "failed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bid")
    parser.add_argument("--ask")
    parser.add_argument("--json")
    parser.add_argument("--no-ingest", action="store_true")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # document text can contain non-cp1252 characters
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main(parser.parse_args())))
