"""Run the API without Docker: `python run_dev.py`.

psycopg's async mode needs the selector event loop, which is not the default on
Windows. It must be set before uvicorn creates its loop, so plain
`uvicorn app.main:app` fails to reach PostgreSQL there.
"""

import asyncio
import sys

import uvicorn

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, loop="none")
