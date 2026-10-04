"""Real-model end-to-end validation: ingestion -> retrieval -> extraction -> reconciliation -> validation.

  python -m app.agents.real_run --patient --bids Bid1 Bid2 --qa

Uses the configured Groq model (GROQ_MODEL). Writes one JSON file per step to --out so partial
progress survives an interruption. `--patient` waits for the Groq token quota to refill instead of
failing (evaluation only). `--qa` then asks the key Q&A regression questions (a subset of
eval/qa_eval.json, unchanged) with the same model.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from app.agents.factory import build_orchestrator, build_qa_agent
from app.agents.qa_eval import evaluate, load_cases, summarize
from app.core.config import ConfigurationError, get_settings
from app.core.database import dispose_engine, get_sessionmaker
from app.llm.groq_client import GroqClient
from app.llm.patient import PatientGroq

# Questions the Phase 9 plan treats as the important regression set (all exist in qa_eval.json).
QA_SUBSET = [
    "b1-final-deadline", "b1-original-deadline", "b1-addendum2-change", "b1-product",
    "b1-mercury", "b2-model", "b2-warranty", "b1-award",
]  # fmt: skip
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "eval" / "runs"


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    logging.basicConfig(level="WARNING")
    try:
        real = GroqClient.from_settings(settings)
    except ConfigurationError as exc:
        print(f"Cannot run: {exc}")
        return 2
    llm = PatientGroq(real, max_wait=args.max_wait) if args.patient else real
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _say(f"model={real.model} patient={args.patient} bids={args.bids} qa={args.qa}")
    try:
        async with get_sessionmaker()() as session:
            for bid in args.bids:
                real.reset_stats()
                started = time.perf_counter()
                state = await build_orchestrator(session, settings, llm).run(bid)
                seconds = round(time.perf_counter() - started, 1)
                (out / f"pipeline_{bid}.json").write_text(
                    json.dumps({"model": real.model, "seconds": seconds, "usage": real.usage_since(0),
                                "waited_for_rate_limit_s": getattr(llm, "waited", 0.0),
                                "state": state.model_dump(mode="json")}, indent=2, default=str),
                    encoding="utf-8",
                )  # fmt: skip
                stages = {n: s.state for n, s in state.stages.items()}
                _say(f"{bid}: status={state.status} stages={stages} total={seconds}s usage={real.usage_since(0)}")

            if args.qa:
                real.reset_stats()
                cases = [c for c in load_cases() if c["id"] in QA_SUBSET]
                agent = build_qa_agent(session, settings, llm)
                rows = await evaluate(agent, cases, evidence_only=False, patient=args.patient)
                (out / f"qa_{real.model.replace('/', '_')}.json").write_text(
                    json.dumps({"model": real.model, "rows": rows, "summary": summarize(rows)}, indent=2, default=str),
                    encoding="utf-8",
                )
                _say(f"Q&A ({len(rows)} questions): {summarize(rows)} usage={real.usage_since(0)}")
    finally:
        await dispose_engine()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bids", nargs="+", default=["Bid1", "Bid2"], help="bids to run through the full pipeline")
    parser.add_argument("--qa", action="store_true", help="also run the Q&A regression subset")
    parser.add_argument("--patient", action="store_true", help="wait out Groq rate limits (evaluation only)")
    parser.add_argument("--max-wait", type=float, default=12 * 3600, help="max total seconds to wait for quota")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="directory for the JSON results")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main(parser.parse_args())))
