"""Vector / keyword / hybrid retrieval on PostgreSQL + pgvector with a small seeded corpus."""

import pytest
from sqlalchemy import text

from app.core.config import Settings
from app.embeddings.service import EmbeddingService
from app.retrieval.filters import SearchFilters
from app.retrieval.keyword_search import keyword_search
from app.retrieval.query import preprocess_query
from app.retrieval.schema import InvalidFilterError, InvalidQueryError
from app.retrieval.service import RetrievalService
from app.retrieval.vector_search import vector_search
from tests.fakes import TopicEmbeddingModel, seed_corpus

NO_FILTERS = SearchFilters()


@pytest.fixture
async def corpus(pg_session):
    model = TopicEmbeddingModel()
    ids = await seed_corpus(pg_session, model)
    settings = Settings(_env_file=None, database_url="postgresql+psycopg://x/x", max_top_k=10)
    service = RetrievalService(pg_session, EmbeddingService(pg_session, model, 8), settings)
    return {"ids": ids, "model": model, "service": service, "session": pg_session}


async def vsearch(c, query: str, filters=NO_FILTERS, limit=10):
    vector = c["model"].embed_query(query)
    return await vector_search(c["session"], vector, c["model"].name, filters, limit)


async def ksearch(c, query: str, filters=NO_FILTERS, limit=10, coverage=0.6):
    terms = preprocess_query(query, 500).terms
    return await keyword_search(c["session"], terms, filters, limit, coverage)


def files(candidates) -> list[str]:
    return [x.file_name for x in candidates]


# ---------------------------------------------------------------- vector search
async def test_vector_search_ranks_by_cosine_and_returns_citation_fields(corpus) -> None:
    hits = await vsearch(corpus, "Which warranty applies?")

    assert hits[0].file_name == "PORFP_Dell_Laptop.pdf"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert hits[0].score <= hits[-1].score  # cosine distance, ascending
    top = hits[0]
    assert (top.bid_id, top.document_type, top.page_number, top.chunk_index) == ("Bid2", "rfp", 3, 3)
    assert top.chunk_id and top.document_id and top.content and top.metadata["bid_id"] == "Bid2"


async def test_vector_search_applies_filters_in_sql(corpus) -> None:
    assert {h.bid_id for h in await vsearch(corpus, "deadline", SearchFilters(bid_id="Bid1"))} == {"Bid1"}
    only_spec = await vsearch(corpus, "deadline", SearchFilters(document_type="specification"))
    assert files(only_spec) == ["Dell_Laptop_Specs.pdf"]
    by_name = await vsearch(corpus, "deadline", SearchFilters(file_name="JA-207652 FINAL.pdf"))
    assert files(by_name) == ["JA-207652 FINAL.pdf"]
    assert await vsearch(corpus, "deadline", SearchFilters(bid_id="Bid1", document_type="specification")) == []


async def test_vector_search_ignores_chunks_from_other_models(corpus) -> None:
    other = corpus["model"].embed_query("deadline")
    assert await vector_search(corpus["session"], other, "some/other-model", NO_FILTERS, 10) == []


# ---------------------------------------------------------------- keyword search
async def test_keyword_exact_bid_number(corpus) -> None:
    hits = await ksearch(corpus, "JA-207652")
    # Matching is on chunk text, so only the chunk that actually contains the number qualifies.
    assert files(hits) == ["JA-207652 FINAL.pdf"]
    assert files(await ksearch(corpus, "BPM044557")) == ["Dell Laptops - BidNet Direct.html"]


async def test_keyword_exact_product_and_model_numbers(corpus) -> None:
    assert files(await ksearch(corpus, "WD22TB4")) == ["Dell_Laptop_Specs.pdf"]
    assert files(await ksearch(corpus, "Dell Latitude 5550")) == ["Dell_Laptop_Specs.pdf"]
    assert files(await ksearch(corpus, "July 9 2024"))[0] in {
        "Addendum 2 RFP JA-207652.pdf",
        "Dell Laptops - BidNet Direct.html",
    }


async def test_keyword_search_applies_filters(corpus) -> None:
    assert await ksearch(corpus, "WD22TB4", SearchFilters(bid_id="Bid1")) == []
    assert len(await ksearch(corpus, "deadline")) == 2  # addendum and RFP chunks both mention it
    assert files(await ksearch(corpus, "deadline", SearchFilters(document_type="addendum"))) == [
        "Addendum 2 RFP JA-207652.pdf"
    ]


async def test_keyword_search_requires_enough_term_coverage(corpus) -> None:
    # "warranty" and "dock" occur in different chunks; both are needed at 0.6 coverage...
    assert await ksearch(corpus, "warranty dock") == []
    # ...but one term is enough when the threshold is lowered.
    assert len(await ksearch(corpus, "warranty dock", coverage=0.5)) == 2
    # Stop words ("what is the for") never count as terms.
    assert files(await ksearch(corpus, "What is the WD22TB4 for?")) == ["Dell_Laptop_Specs.pdf"]
    assert await ksearch(corpus, "what is the") == []


# ---------------------------------------------------------------- hybrid service
async def test_hybrid_combines_methods_and_preserves_metadata(corpus) -> None:
    found = await corpus["service"].retrieve("What is the deadline for JA-207652?", top_k=3)

    assert found.top_k == 3 and len(found.results) <= 3
    first = found.results[0]
    assert first.candidate.file_name == "JA-207652 FINAL.pdf"
    assert first.methods == ["vector", "keyword"]  # evidence from both rankings
    assert first.rrf_score > found.results[1].rrf_score
    c = first.candidate
    assert (c.bid_id, c.document_type, c.page_number, c.chunk_index) == ("Bid1", "rfp", 2, 1)
    ids = [r.candidate.chunk_id for r in found.results]
    assert len(ids) == len(set(ids))  # no duplicate chunks


async def test_hybrid_html_result_keeps_section_and_no_page_number(corpus) -> None:
    found = await corpus["service"].retrieve("BPM044557", mode="hybrid")
    html = found.results[0].candidate
    assert html.file_name.endswith(".html")
    assert html.page_number is None
    assert html.metadata["section"] == "Bid Information"


async def test_modes_run_one_method_only(corpus) -> None:
    svc = corpus["service"]
    assert all(r.methods == ["vector"] for r in (await svc.retrieve("deadline", mode="vector")).results)
    kw = await svc.retrieve("WD22TB4", mode="keyword")
    assert [r.methods for r in kw.results] == [["keyword"]]


async def test_top_k_is_enforced(corpus) -> None:
    svc = corpus["service"]
    assert len((await svc.retrieve("deadline laptop warranty", top_k=2)).results) == 2
    assert len((await svc.retrieve("deadline laptop warranty")).results) == 5  # default_top_k
    for bad in (0, -1, 11):
        with pytest.raises(InvalidQueryError):
            await svc.retrieve("deadline", top_k=bad)


async def test_filters_and_validation(corpus) -> None:
    svc = corpus["service"]
    found = await svc.retrieve("deadline", bid_id="Bid2", document_type="rfp", top_k=10)
    assert [r.candidate.file_name for r in found.results] == ["PORFP_Dell_Laptop.pdf"]

    with pytest.raises(InvalidFilterError, match="Bid9"):
        await svc.retrieve("deadline", bid_id="Bid9")
    with pytest.raises(InvalidFilterError, match="document_type"):
        await svc.retrieve("deadline", document_type="memo")
    with pytest.raises(InvalidQueryError):
        await svc.retrieve("   ")
    with pytest.raises(InvalidQueryError):
        await svc.retrieve("x" * 501)


async def test_available_filters(corpus) -> None:
    f = await corpus["service"].available_filters()
    assert f["bids"] == ["Bid1", "Bid2"]
    assert "addendum" in f["document_types"] and len(f["file_names"]) == 5


async def test_gin_and_hnsw_indexes_exist(pg_session) -> None:
    rows = (await pg_session.execute(
        text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'document_chunks'")
    )).all()
    defs = {name: d for name, d in rows}
    assert "USING gin (search_vector)" in defs["ix_document_chunks_search_vector"]
    assert "hnsw" in defs["ix_document_chunks_embedding_hnsw"] and "vector_cosine_ops" in defs["ix_document_chunks_embedding_hnsw"]
