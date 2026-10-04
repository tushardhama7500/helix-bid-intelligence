"""Thin, reusable wrapper around the Groq SDK. All agents and services go through this."""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from groq import AsyncGroq

from app.core.config import ConfigurationError, Settings
from app.core.context import run_id_var, stage_var

logger = logging.getLogger(__name__)

Message = dict[str, str]

# Bounded by design: the SDK retries transient errors at most this many times, each request
# times out after DEFAULT_TIMEOUT seconds, and callers add at most one rate-limit wait on top.
DEFAULT_TIMEOUT = 60.0
SDK_MAX_RETRIES = 2


@dataclass
class CallRecord:
    """Metadata of one Groq request. Never contains prompts, answers or credentials."""

    seconds: float
    ok: bool
    prompt_chars: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None  # "length" means the reply was cut off


class GroqClient:
    def __init__(self, api_key: str | None, model: str | None, timeout: float = DEFAULT_TIMEOUT) -> None:
        if not api_key:
            raise ConfigurationError("GROQ_API_KEY is not set.")
        if not model:
            raise ConfigurationError("GROQ_MODEL is not set.")
        self.model = model
        self._client = AsyncGroq(api_key=api_key, timeout=timeout, max_retries=SDK_MAX_RETRIES)
        self.calls: list[CallRecord] = []  # one entry per request, for latency / token accounting

    @classmethod
    def from_settings(cls, settings: Settings) -> "GroqClient":
        key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
        return cls(key, settings.groq_model)

    def reset_stats(self) -> None:
        self.calls.clear()

    def usage_since(self, start: int = 0) -> dict[str, Any]:
        """Calls, tokens and latency of the requests made after `start` (an index into `calls`).

        Token counts are what Groq reports (for reasoning models they include hidden reasoning
        tokens). No prices are computed: that needs the pricing of the configured model.
        """
        mine = self.calls[start:]
        ok = [c for c in mine if c.ok]
        return {
            "calls": len(mine), "failed": len(mine) - len(ok),
            "tokens_in": sum(c.prompt_tokens or 0 for c in ok), "tokens_out": sum(c.completion_tokens or 0 for c in ok),
            "seconds": round(sum(c.seconds for c in mine), 1),
            "truncated": sum(1 for c in ok if c.finish_reason == "length"),
        }  # fmt: skip

    async def generate(self, messages: Sequence[Message], **params: Any) -> str:
        """Send chat `messages` and return the text of the model's reply.

        Extra keyword arguments (temperature, max_tokens, ...) are passed through.
        """
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        logger.debug("Groq request: model=%s messages=%d chars=%d", self.model, len(messages), prompt_chars)
        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self.model, messages=list(messages), **params
            )
        except Exception as exc:
            self.calls.append(CallRecord(time.perf_counter() - started, False, prompt_chars))
            logger.warning(
                "groq call run=%s stage=%s model=%s ok=false ms=%d error=%s",
                run_id_var.get(), stage_var.get(), self.model, self.calls[-1].seconds * 1000, type(exc).__name__,
            )  # fmt: skip
            raise
        usage = getattr(response, "usage", None)
        choice = response.choices[0]
        self.calls.append(
            CallRecord(
                seconds=time.perf_counter() - started, ok=True, prompt_chars=prompt_chars,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                finish_reason=getattr(choice, "finish_reason", None),
            )
        )  # fmt: skip
        last = self.calls[-1]
        logger.info(
            "groq call run=%s stage=%s model=%s ok=true ms=%d tokens_in=%s tokens_out=%s finish=%s",
            run_id_var.get(), stage_var.get(), self.model, last.seconds * 1000, last.prompt_tokens,
            last.completion_tokens, last.finish_reason,
        )  # fmt: skip
        if self.calls[-1].finish_reason == "length":
            logger.warning("Groq reply was truncated (finish_reason=length); consider the output token limit")
        return choice.message.content or ""

    async def generate_json(self, messages: Sequence[Message], **params: Any) -> str:
        """Like `generate`, but asks Groq for a JSON object (deterministic by default).

        JSON mode is supported by every Groq chat model, unlike strict JSON-schema
        output, so callers must still validate the result against their schema.
        """
        params.setdefault("temperature", 0)
        return await self.generate(messages, response_format={"type": "json_object"}, **params)
