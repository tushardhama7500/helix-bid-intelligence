"""Keyword search: PostgreSQL full-text search over the generated `search_vector`."""

import math

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.filters import SearchFilters
from app.retrieval.schema import FTS_CONFIG, Candidate
from app.retrieval.vector_search import row_to_candidate

_MEANINGFUL_TERMS_SQL = f"""
SELECT t.term FROM unnest(CAST(:terms AS text[])) WITH ORDINALITY AS t(term, pos)
WHERE numnode(websearch_to_tsquery('{FTS_CONFIG}', t.term)) > 0
ORDER BY t.pos
"""

_SQL = """
SELECT * FROM (
    SELECT c.id AS chunk_id, c.content, c.chunk_index, c.bid_id, c.metadata,
           d.id AS document_id, d.file_name, d.document_type, p.page_number,
           ts_rank_cd(c.search_vector, q.any_term, 1) AS score,
           {matched_expr} AS matched
    FROM document_chunks c
    JOIN documents d ON d.id = c.document_id
    JOIN document_pages p ON p.id = c.page_id,
         (SELECT {any_term_expr} AS any_term) q
    WHERE c.search_vector @@ q.any_term{filters}
) hits
WHERE matched >= :min_match
ORDER BY matched DESC, score DESC, chunk_id
LIMIT :limit
"""


async def meaningful_terms(session: AsyncSession, terms: tuple[str, ...]) -> list[str]:
    """Terms that survive PostgreSQL's own stop-word removal ('what', 'is', 'for', ...)."""
    if not terms:
        return []
    result = await session.execute(text(_MEANINGFUL_TERMS_SQL), {"terms": list(terms)})
    return [row[0] for row in result]


async def keyword_search(
    session: AsyncSession,
    terms: tuple[str, ...],
    filters: SearchFilters,
    limit: int,
    min_coverage: float = 0.6,
) -> list[Candidate]:
    """Top `limit` chunks that contain enough of the query's terms.

    Each term goes through `websearch_to_tsquery` on its own, so identifiers such
    as `JA-207652` or `WD22TB4` are parsed exactly like the indexed text. A chunk
    qualifies only if it contains at least ``ceil(min_coverage * n)`` of the `n`
    terms left after stop-word removal. Matching *any* term is far too noisy for
    questions: words like "required" or "submission" occur in most procurement
    chunks and would flood the fusion with irrelevant hits. Results are ordered
    by number of terms matched, then `ts_rank_cd` (which rewards terms close
    together). The score is a relative rank within this query only. An empty list
    is a normal outcome: it means no chunk is a strong lexical match.
    """
    terms = tuple(await meaningful_terms(session, terms))
    if not terms:
        return []
    min_match = max(1, math.ceil(min_coverage * len(terms)))

    params = {f"t{i}": term for i, term in enumerate(terms)}
    queries = [f"websearch_to_tsquery('{FTS_CONFIG}', :t{i})" for i in range(len(terms))]
    matched_expr = " + ".join(f"(c.search_vector @@ {q})::int" for q in queries)
    clause, filter_params = filters.to_sql()

    result = await session.execute(
        text(_SQL.format(matched_expr=matched_expr, any_term_expr=" || ".join(queries), filters=clause)),
        {"limit": limit, "min_match": min_match, **params, **filter_params},
    )
    return [row_to_candidate(row, rank) for rank, row in enumerate(result.mappings(), start=1)]
