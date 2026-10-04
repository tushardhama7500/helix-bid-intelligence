import pytest
from pydantic import ValidationError

from app.core.config import Settings


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for name in ("DATABASE_URL", "GROQ_API_KEY", "GROQ_MODEL", "FRONTEND_ORIGIN"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(
        monkeypatch,
        DATABASE_URL="postgresql://u:p@db:5432/x",
        GROQ_API_KEY="k",
        GROQ_MODEL="some-model",
    )
    assert s.app_env == "development"
    assert s.log_level == "INFO"
    assert s.groq_model == "some-model"
    assert s.frontend_origin == "http://localhost:5173"
    assert s.async_database_url == "postgresql+psycopg://u:p@db:5432/x"


def test_blank_groq_values_become_none(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(
        monkeypatch,
        DATABASE_URL="postgresql+psycopg://u:p@db/x",
        GROQ_API_KEY="",
        GROQ_MODEL="",
    )
    assert s.groq_api_key is None
    assert s.groq_model is None


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        _settings(monkeypatch)
