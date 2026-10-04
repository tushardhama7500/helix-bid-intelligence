import pytest

from app.retrieval.filters import SearchFilters
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.query import preprocess_query
from app.retrieval.schema import Candidate, InvalidQueryError


def cand(chunk_id: int, score: float = 0.0) -> Candidate:
    return Candidate(
        chunk_id=chunk_id, content=f"c{chunk_id}", document_id=1, bid_id="Bid1", file_name="f.pdf",
        document_type="rfp", page_number=1, chunk_index=chunk_id, metadata={}, score=score,
    )  # fmt: skip


def test_rrf_combines_two_lists() -> None:
    fused = reciprocal_rank_fusion({"vector": [cand(1), cand(2), cand(3)], "keyword": [cand(3), cand(1)]}, k=60)

    scores = {r.candidate.chunk_id: r.rrf_score for r in fused}
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)  # rank 1 + rank 2
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)  # rank 3 + rank 1
    assert scores[2] == pytest.approx(1 / 62)  # vector only
    assert [r.candidate.chunk_id for r in fused] == [1, 3, 2]


def test_chunk_in_both_lists_appears_once_with_both_methods() -> None:
    fused = reciprocal_rank_fusion({"vector": [cand(7, 0.2)], "keyword": [cand(7, 0.9)]})

    assert len(fused) == 1
    assert fused[0].methods == ["vector", "keyword"]
    assert fused[0].ranks == {"vector": 1, "keyword": 1}
    assert fused[0].scores == {"vector": 0.2, "keyword": 0.9}  # raw scores stay separate


def test_missing_or_empty_list_is_handled() -> None:
    only_vector = reciprocal_rank_fusion({"vector": [cand(1), cand(2)], "keyword": []})
    assert [r.candidate.chunk_id for r in only_vector] == [1, 2]
    assert all(r.methods == ["vector"] for r in only_vector)
    assert reciprocal_rank_fusion({}) == []


def test_duplicate_inside_one_list_counts_once() -> None:
    fused = reciprocal_rank_fusion({"vector": [cand(1), cand(1), cand(2)]})
    assert [r.candidate.chunk_id for r in fused] == [1, 2]
    assert fused[0].rrf_score == pytest.approx(1 / 61)


def test_k_is_configurable_and_ties_are_deterministic() -> None:
    fused = reciprocal_rank_fusion({"vector": [cand(5)], "keyword": [cand(4)]}, k=10)
    assert fused[0].rrf_score == pytest.approx(1 / 11)
    assert [r.candidate.chunk_id for r in fused] == [4, 5]  # equal score -> lower id first


def test_preprocess_keeps_identifiers_and_dedups() -> None:
    q = preprocess_query("  What is   JA-207652 and WD22TB4, ja-207652?  ", 500)
    assert q.text == "What is JA-207652 and WD22TB4, ja-207652?"
    assert q.terms == ("what", "is", "ja-207652", "and", "wd22tb4")


def test_preprocess_rejects_empty_and_long() -> None:
    with pytest.raises(InvalidQueryError):
        preprocess_query("   ", 500)
    with pytest.raises(InvalidQueryError):
        preprocess_query("x" * 501, 500)


def test_filters_sql_uses_bind_parameters_only() -> None:
    clause, params = SearchFilters(bid_id="Bid1'; DROP TABLE x;--", document_type="addendum").to_sql()
    assert clause == " AND c.bid_id = :f_bid_id AND d.document_type = :f_document_type"
    assert "DROP" not in clause and params["f_bid_id"].startswith("Bid1")
    assert SearchFilters().to_sql() == ("", {}) and not SearchFilters().active
