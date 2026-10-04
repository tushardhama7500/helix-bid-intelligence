"""Lightweight structured logging for agent executions (no prompts, documents or secrets)."""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.core.context import run_id_var, stage_var

logger = logging.getLogger("helix.agents")


@asynccontextmanager
async def agent_span(agent: str, bid_id: str, stage: str) -> AsyncIterator[dict]:
    """Log `agent=... bid=... stage=... ok=... ms=...` once per execution.

    Yields a dict the caller may fill (`span["note"] = ...`); exceptions are logged
    with their type only and re-raised.
    """
    span: dict = {"duration_ms": 0}
    started = time.perf_counter()
    token = stage_var.set(stage)  # lets the Groq client label its log lines with the stage
    try:
        yield span
    except BaseException as exc:
        span["duration_ms"] = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "run=%s agent=%s bid=%s stage=%s ok=false ms=%d error=%s",
            run_id_var.get(), agent, bid_id, stage, span["duration_ms"], type(exc).__name__,
        )  # fmt: skip
        raise
    finally:
        stage_var.reset(token)
    span["duration_ms"] = int((time.perf_counter() - started) * 1000)
    logger.info(
        "run=%s agent=%s bid=%s stage=%s ok=true ms=%d%s",
        run_id_var.get(), agent, bid_id, stage, span["duration_ms"], f" note={span['note']}" if span.get("note") else "",
    )  # fmt: skip
