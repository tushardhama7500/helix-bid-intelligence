"""CLI: `python -m app.retrieval [--no-filters] [--json FILE] [--dataset FILE]`.

Runs the retrieval evaluation against the real database and prints Recall@K for
vector-only, keyword-only and hybrid search. No LLM is involved.
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.database import dispose_engine, get_sessionmaker
from app.retrieval.evaluation import DEFAULT_DATASET, KS, MODES, evaluate, load_dataset
from app.retrieval.service import RetrievalService


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    logging.basicConfig(level="WARNING")
    questions = load_dataset(Path(args.dataset))
    try:
        async with get_sessionmaker()() as session:
            report = await evaluate(RetrievalService.from_settings(session, settings), questions, not args.no_filters)
    finally:
        await dispose_engine()

    print(f"Questions: {report['questions']}   metadata filters: {'on' if report['filters'] else 'off'}\n")
    print(f"{'mode':<9}" + "".join(f"Recall@{k:<4}" for k in KS) + "MRR@5")
    for mode in MODES:
        s = report["summary"][mode]
        print(f"{mode:<9}" + "".join(f"{s[f'recall@{k}']:<10.3f}" for k in KS) + f"{s['mrr@5']:.3f}")

    print("\nPer question (rank of first relevant chunk, '-' = miss):")
    print(f"{'id':<24}" + "".join(f"{m:<9}" for m in MODES) + "query")
    by_mode = report["outcomes"]
    for i, q in enumerate(questions):
        ranks = [by_mode[m][i].hit_rank for m in MODES]
        print(f"{q['id']:<24}" + "".join(f"{(r or '-')!s:<9}" for r in ranks) + q["query"])

    if args.verbose:
        for i, q in enumerate(questions):
            print(f"\n[{q['id']}] {q['query']}")
            for m in MODES:
                print(f"  {m}:")
                for line in by_mode[m][i].top:
                    print(f"    {line}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, default=lambda o: dataclasses.asdict(o)), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-filters", action="store_true", help="ignore bid/type filters in the dataset")
    parser.add_argument("--json", help="write the full report to this file")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("-v", "--verbose", action="store_true", help="show the top-5 results per question")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main(parser.parse_args())))
