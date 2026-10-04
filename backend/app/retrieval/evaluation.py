"""Retrieval evaluation: Recall@K for vector-only, keyword-only and hybrid search.

Ground truth lives in `eval/retrieval_eval.json`. A retrieved chunk is relevant
when it comes from a file whose name contains `file_contains` AND its text
matches `content_regex` (whitespace-normalised, case-insensitive). A question is
a hit at K when any of its top-K chunks is relevant, so Recall@K here is the
fraction of questions answered by at least one relevant chunk in the top K.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.retrieval.schema import FusedResult
from app.retrieval.service import HYBRID, KEYWORD, VECTOR, RetrievalService

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "eval" / "retrieval_eval.json"
KS = (1, 3, 5)
MODES = (VECTOR, KEYWORD, HYBRID)


def load_dataset(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_relevant(result: FusedResult, relevant: list[dict[str, str]]) -> bool:
    candidate = result.candidate
    content = " ".join(candidate.content.split())
    return any(
        item["file_contains"].lower() in candidate.file_name.lower()
        and re.search(item["content_regex"], content, re.IGNORECASE)
        for item in relevant
    )


def label(result: FusedResult) -> str:
    c = result.candidate
    where = f"p.{c.page_number}" if c.page_number is not None else f"sec:{(c.metadata.get('section') or '-')[:20]}"
    return f"{c.file_name[:38]} {where} #{c.chunk_index}"


@dataclass
class QuestionOutcome:
    id: str
    query: str
    mode: str
    hit_rank: int | None  # 1-based rank of first relevant chunk in the top max(KS), else None
    top: list[str]  # labels of the top-5 results, with a "*" on relevant ones


async def evaluate(
    service: RetrievalService,
    questions: list[dict[str, Any]],
    use_filters: bool = True,
    modes: tuple[str, ...] = MODES,
) -> dict[str, Any]:
    outcomes: dict[str, list[QuestionOutcome]] = {m: [] for m in modes}
    for q in questions:
        filters = (
            {k: q[k] for k in ("bid_id", "document_type", "file_name") if k in q} if use_filters else {}
        )
        for mode in modes:
            found = await service.retrieve(q["query"], top_k=max(KS), mode=mode, **filters)
            flags = [is_relevant(r, q["relevant"]) for r in found.results]
            hit = next((i for i, ok in enumerate(flags, start=1) if ok), None)
            top = [("*" if ok else " ") + label(r) for r, ok in zip(found.results, flags, strict=True)]
            outcomes[mode].append(QuestionOutcome(q["id"], q["query"], mode, hit, top))

    summary: dict[str, Any] = {}
    for mode, items in outcomes.items():
        n = len(items)
        summary[mode] = {
            **{f"recall@{k}": sum(1 for o in items if o.hit_rank and o.hit_rank <= k) / n for k in KS},
            "mrr@5": sum(1 / o.hit_rank for o in items if o.hit_rank) / n,
        }
    return {"questions": len(questions), "filters": use_filters, "summary": summary, "outcomes": outcomes}
