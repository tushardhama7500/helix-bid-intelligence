"""Evaluation-only wrapper: wait out Groq rate limits instead of failing.

Production code keeps its bounded behaviour (one short wait, then a structured error). Real-model
*evaluations* on a small token quota need to wait for the quota to refill instead; they use this
wrapper explicitly (`--patient`). The wait is bounded by `max_wait` seconds in total.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import groq

from app.llm.groq_client import GroqClient

logger = logging.getLogger(__name__)

MAX_SINGLE_WAIT = 600.0
MIN_SINGLE_WAIT = 5.0


class PatientGroq:
    def __init__(self, inner: GroqClient, max_wait: float = 12 * 3600) -> None:
        self.inner = inner
        self.max_wait = max_wait
        self.waited = 0.0
        self.rate_limit_waits = 0

    # The LLMClient surface used by the services, delegated to the real client.
    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def calls(self):
        return self.inner.calls

    def usage_since(self, start: int = 0) -> dict[str, Any]:
        return self.inner.usage_since(start)

    def reset_stats(self) -> None:
        self.inner.reset_stats()

    async def generate_json(self, messages: Sequence[dict[str, str]], **params: Any) -> str:
        while True:
            try:
                return await self.inner.generate_json(messages, **params)
            except groq.RateLimitError as exc:
                try:
                    hinted = float(exc.response.headers.get("retry-after", 30))
                except (AttributeError, TypeError, ValueError):
                    hinted = 30.0
                wait = min(max(hinted, MIN_SINGLE_WAIT), MAX_SINGLE_WAIT)
                if self.waited + wait > self.max_wait:
                    raise
                self.rate_limit_waits += 1
                self.waited += wait
                logger.warning("Rate limited; waiting %.0fs (total waited %.0fs)", wait, self.waited)
                await asyncio.sleep(wait)
