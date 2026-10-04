from types import SimpleNamespace

import pytest

from app.core.config import ConfigurationError
from app.llm import groq_client
from app.llm.groq_client import GroqClient


class FakeAsyncGroq:
    last_call: dict = {}

    def __init__(self, api_key: str, **options) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        FakeAsyncGroq.last_call = kwargs
        message = SimpleNamespace(content="pong")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture(autouse=True)
def fake_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groq_client, "AsyncGroq", FakeAsyncGroq)


def test_missing_model_raises_clear_error() -> None:
    with pytest.raises(ConfigurationError, match="GROQ_MODEL"):
        GroqClient(api_key="k", model=None)


def test_missing_api_key_raises_clear_error() -> None:
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        GroqClient(api_key=None, model="m")


async def test_generate_returns_text_and_uses_configured_model() -> None:
    client = GroqClient(api_key="k", model="configured-model")
    reply = await client.generate([{"role": "user", "content": "hi"}], max_tokens=5)

    assert reply == "pong"
    assert FakeAsyncGroq.last_call["model"] == "configured-model"
    assert FakeAsyncGroq.last_call["max_tokens"] == 5
