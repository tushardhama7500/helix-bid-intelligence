"""CLI: `python -m app.embeddings` embeds all chunks that lack an embedding."""

import asyncio
import logging
import sys

from app.core.config import get_settings
from app.core.database import dispose_engine, get_sessionmaker, init_db
from app.embeddings.service import EmbeddingService


async def main() -> int:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    await init_db()
    print("Embedding chunks...\n")
    try:
        async with get_sessionmaker()() as session:
            summary = await EmbeddingService.from_settings(session, settings).generate()
    finally:
        await dispose_engine()

    print(f"Total chunks: {summary.total_chunks}")
    print(f"Embedded: {summary.embedded}")
    print(f"Skipped: {summary.skipped}")
    print(f"Failed: {summary.failed}")
    for error in summary.errors:
        print(f"  - {error}")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main()))
