"""CLI: `python -m app.ingestion [--force]` ingests everything under DATA_DIR.

`--force` reprocesses unchanged files too (e.g. after a parser change).
"""

import asyncio
import json
import logging
import sys
from dataclasses import asdict

from app.core.config import get_settings
from app.core.database import dispose_engine, get_sessionmaker, init_db
from app.ingestion.service import IngestionService


async def main(force: bool = False) -> int:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    await init_db()
    try:
        async with get_sessionmaker()() as session:
            summary = await IngestionService.from_settings(session, settings).run(force=force)
    finally:
        await dispose_engine()
    print(json.dumps(asdict(summary), indent=2))
    return 1 if summary.failed else 0


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main(force="--force" in sys.argv[1:])))
