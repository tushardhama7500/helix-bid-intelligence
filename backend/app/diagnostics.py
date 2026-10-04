"""Database diagnostics: `python -m app.diagnostics`.

Verifies the production schema (pgvector extension, vector column and HNSW index, full-text
column and GIN index, key constraints), reports row counts, and shows whether the planner uses
the indexes. On a corpus this small PostgreSQL normally prefers a sequential scan, which is
correct; to prove the indexes *can* be used the check also plans each query with sequential
scans disabled (`SET LOCAL enable_seqscan = off`). Exit status is non-zero if any check fails.
"""

import asyncio
import json
import sys
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.config import get_settings
from app.core.database import dispose_engine, get_engine

EXPECTED_INDEXES = {
    "ix_document_chunks_embedding_hnsw": ("hnsw", "vector_cosine_ops"),
    "ix_document_chunks_search_vector": ("gin", "search_vector"),
}


def _plan_uses(plan: Any, index_name: str) -> bool:
    return index_name in json.dumps(plan)


async def run_checks(conn: AsyncConnection) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []

    def check(ok: bool, message: str) -> None:
        results.append((bool(ok), message))

    version = (await conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))).scalar()
    check(version, f"pgvector extension installed (version {version})")

    column = (
        await conn.execute(
            text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding' AND NOT attisdropped"
            )
        )
    ).scalar()
    check(column and column.startswith("vector("), f"document_chunks.embedding column type: {column}")

    tsv = (
        await conn.execute(
            text(
                "SELECT format_type(a.atttypid, a.atttypmod), a.attgenerated FROM pg_attribute a "
                "WHERE a.attrelid = 'document_chunks'::regclass AND a.attname = 'search_vector' AND NOT a.attisdropped"
            )
        )
    ).first()
    check(tsv and tsv[0] == "tsvector" and tsv[1] == "s", f"document_chunks.search_vector: {tsv[0] if tsv else None} (stored generated column)")

    indexes = {
        r[0]: r[1] for r in (await conn.execute(text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'document_chunks'"))).all()
    }
    for name, (method, detail) in EXPECTED_INDEXES.items():
        definition = indexes.get(name, "")
        check(f"USING {method}" in definition and detail in definition, f"index {name}: {definition or 'MISSING'}")

    constraints = {
        r[0] for r in (await conn.execute(text(
            "SELECT conname FROM pg_constraint WHERE conrelid IN "
            "('documents'::regclass, 'document_chunks'::regclass, 'document_pages'::regclass)"))).all()
    }  # fmt: skip
    for needed in ("documents_file_path_key", "document_chunks_document_id_chunk_index_key",
                   "document_chunks_document_id_fkey", "document_chunks_page_id_fkey", "document_pages_document_id_fkey"):  # fmt: skip
        check(needed in constraints, f"constraint {needed}")

    counts = (await conn.execute(text(
        "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM document_pages), count(*), "
        "count(embedding), count(search_vector) FROM document_chunks"))).one()  # fmt: skip
    results.append((True, f"rows: documents={counts[0]} pages={counts[1]} chunks={counts[2]} embedded={counts[3]} with_tsvector={counts[4]}"))
    check(counts[2] == counts[3] == counts[4] or counts[2] == 0, "every chunk has an embedding and a tsvector")

    if counts[3]:
        reference = (await conn.execute(text("SELECT embedding::text FROM document_chunks WHERE embedding IS NOT NULL LIMIT 1"))).scalar()
        queries = {
            "ix_document_chunks_embedding_hnsw": f"SELECT id FROM document_chunks ORDER BY embedding <=> '{reference}'::vector LIMIT 5",
            "ix_document_chunks_search_vector": "SELECT id FROM document_chunks WHERE search_vector @@ websearch_to_tsquery('english', 'deadline') LIMIT 5",
        }
        for index, sql in queries.items():
            natural = (await conn.execute(text("EXPLAIN (FORMAT JSON) " + sql))).scalar()
            await conn.execute(text("SET LOCAL enable_seqscan = off"))
            forced = (await conn.execute(text("EXPLAIN (FORMAT JSON) " + sql))).scalar()
            await conn.execute(text("SET LOCAL enable_seqscan = on"))
            check(_plan_uses(forced, index), f"{index} is usable by the planner (seqscan disabled); default plan on {counts[2]} rows uses it: {_plan_uses(natural, index)}")
    return results


async def main() -> int:
    get_settings()
    try:
        async with get_engine().connect() as conn:
            async with conn.begin():
                results = await run_checks(conn)
    finally:
        await dispose_engine()
    for ok, message in results:
        print(f"{'PASS' if ok else 'FAIL'}  {message}")
    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main()))
