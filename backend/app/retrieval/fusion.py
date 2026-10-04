"""Reciprocal Rank Fusion."""

from collections.abc import Mapping, Sequence

from app.retrieval.schema import Candidate, FusedResult


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Candidate]], k: int = 60
) -> list[FusedResult]:
    """Fuse ranked lists: ``RRF(d) = sum over lists of 1 / (k + rank_in_list(d))``.

    `rankings` maps a method name to its candidates, best first (rank = position,
    starting at 1). Only ranks matter, so scores of different scales (cosine
    distance, ts_rank) never need normalising. A chunk found by several methods
    sums their contributions; a chunk missing from a list simply gets nothing
    from it. Each chunk appears once in the output.

    The result is a *ranking score*: only its order is meaningful, and it is not
    a probability or a confidence. With k=60 a chunk ranked first in both of two
    lists scores 2/61 ~ 0.0328; first in one list only, 1/61 ~ 0.0164.

    Order is deterministic: higher score, then found by more methods, then better
    best-rank, then lower chunk id.
    """
    fused: dict[int, FusedResult] = {}
    for method, candidates in rankings.items():
        for position, candidate in enumerate(candidates, start=1):
            entry = fused.get(candidate.chunk_id)
            if entry is None:
                entry = fused[candidate.chunk_id] = FusedResult(candidate=candidate, rrf_score=0.0)
            if method in entry.ranks:  # duplicate within one list: keep the best rank
                continue
            entry.rrf_score += 1.0 / (k + position)
            entry.methods.append(method)
            entry.ranks[method] = position
            entry.scores[method] = candidate.score

    return sorted(
        fused.values(),
        key=lambda r: (-r.rrf_score, -len(r.methods), min(r.ranks.values()), r.candidate.chunk_id),
    )
