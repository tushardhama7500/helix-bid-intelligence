from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    # Hard caps reject absurd payloads before any work is done; the service applies the configured limits.
    query: str = Field(max_length=2000, description="Natural-language question or exact terms")
    bid_id: str | None = Field(default=None, max_length=64)
    document_type: str | None = Field(default=None, max_length=32)
    file_name: str | None = Field(default=None, max_length=300, description="Exact file name")
    document_id: int | None = Field(default=None, ge=1, le=2**31 - 1)
    page_number: int | None = Field(default=None, ge=1, le=100_000)
    top_k: int | None = Field(default=None, ge=-1000, le=1000, description="Defaults to DEFAULT_TOP_K; max MAX_TOP_K")


class Source(BaseModel):
    """Everything needed to cite a chunk. `page_number` is null for HTML."""

    document_id: int
    bid_id: str
    file_name: str
    document_type: str
    page_number: int | None
    chunk_index: int
    metadata: dict[str, Any]  # e.g. {"section": "..."} for HTML


class VectorScore(BaseModel):
    rank: int
    cosine_distance: float = Field(description="0 = identical direction; smaller is closer")
    cosine_similarity: float = Field(description="1 - cosine_distance")


class KeywordScore(BaseModel):
    rank: int
    ts_rank: float = Field(description="PostgreSQL ts_rank_cd; only comparable within one query")


class Scores(BaseModel):
    rrf: float = Field(description="Reciprocal Rank Fusion score: a ranking score, not a probability")
    vector: VectorScore | None = None
    keyword: KeywordScore | None = None


class SearchResult(BaseModel):
    chunk_id: int
    content: str
    score: float = Field(description="The RRF score (same as scores.rrf). Not a confidence.")
    retrieval_methods: list[Literal["vector", "keyword"]]
    scores: Scores
    source: Source


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResult]


class FiltersResponse(BaseModel):
    bids: list[str]
    document_types: list[str]
    file_names: list[str]
