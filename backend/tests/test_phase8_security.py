"""Security checks added in Phase 8: the Groq key must never leak."""

import logging

import groq
import httpx
import pytest

from app.core.config import ConfigurationError, Settings
from app.extraction.prompts import SYSTEM_PROMPT, build_user_prompt
from app.extraction.fields import FIELDS
from app.extraction.service import LLMProviderError, call_llm
from app.llm.groq_client import GroqClient
from app.reconciliation.llm import SYSTEM_PROMPT as RECON_PROMPT
from app.agents.report_qa_agent import SYSTEM_PROMPT as QA_PROMPT

KEY = "test_secret_key_1234567890"


def settings(**kw) -> Settings:
    return Settings(_env_file=None, database_url="postgresql+psycopg://user:dbpass@h/x", groq_api_key=KEY, groq_model="m", **kw)


def test_settings_never_reveal_the_key_when_printed_or_logged() -> None:
    s = settings()
    for text in (repr(s), str(s), f"{s!r}", str(s.model_dump()), s.model_dump_json()):
        assert KEY not in text
    assert s.groq_api_key.get_secret_value() == KEY  # still usable by the client


def test_missing_key_or_model_gives_a_clear_error_without_secrets() -> None:
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        GroqClient.from_settings(Settings(_env_file=None, database_url="postgresql+psycopg://x/x", groq_model="m"))
    with pytest.raises(ConfigurationError, match="GROQ_MODEL") as info:
        GroqClient.from_settings(Settings(_env_file=None, database_url="postgresql+psycopg://x/x", groq_api_key=KEY))
    assert KEY not in str(info.value)


async def test_provider_errors_and_logs_never_contain_the_key(caplog) -> None:
    caplog.set_level(logging.DEBUG)

    class Failing:
        model = "m"

        async def generate_json(self, messages, **_):
            raise groq.AuthenticationError(
                f"Invalid API Key {KEY}", response=httpx.Response(401, request=httpx.Request("POST", "http://x")), body=None
            )

    with pytest.raises(LLMProviderError) as info:
        await call_llm(Failing(), [{"role": "user", "content": "x"}], max_tokens=10)
    # Only the exception *type* is surfaced; the provider's message (which could echo a key) is not.
    assert KEY not in str(info.value) and "AuthenticationError" in str(info.value)
    assert KEY not in caplog.text


def test_no_prompt_contains_credentials_or_configuration() -> None:
    prompts = [SYSTEM_PROMPT, RECON_PROMPT, QA_PROMPT, build_user_prompt(FIELDS[:2], [], {f.key: {} for f in FIELDS[:2]}, 100)]
    for p in prompts:
        assert KEY not in p and "dbpass" not in p and "GROQ_API_KEY" not in p and "DATABASE_URL" not in p


def test_client_has_bounded_timeout_and_retries() -> None:
    from app.llm import groq_client

    assert groq_client.DEFAULT_TIMEOUT <= 120 and groq_client.SDK_MAX_RETRIES <= 3
    client = GroqClient(KEY, "m")
    assert client._client.max_retries == groq_client.SDK_MAX_RETRIES and client._client.timeout == groq_client.DEFAULT_TIMEOUT


# ---------------------------------------------------------------- Groq accounting
async def test_client_records_calls_tokens_latency_and_truncation_without_content(monkeypatch) -> None:
    from types import SimpleNamespace

    from app.llm import groq_client

    replies = iter([("a", "stop", 100, 40), ("b", "length", 120, 3000)])

    async def create(**_):
        text, reason, tin, tout = next(replies)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=reason)],
            usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout),
        )

    class Fake:
        def __init__(self, api_key, **options):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setattr(groq_client, "AsyncGroq", Fake)
    client = GroqClient(KEY, "m")
    await client.generate_json([{"role": "user", "content": "hello"}])
    await client.generate_json([{"role": "user", "content": "world!"}])

    usage = client.usage_since(0)
    assert (usage["calls"], usage["failed"], usage["tokens_in"], usage["tokens_out"], usage["truncated"]) == (2, 0, 220, 3040, 1)
    assert client.usage_since(1)["calls"] == 1
    assert all(KEY not in repr(c) and "hello" not in repr(c) for c in client.calls)  # metadata only
    assert [c.prompt_chars for c in client.calls] == [5, 6]
