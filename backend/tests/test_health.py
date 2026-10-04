from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.llm import groq_client
from app.main import app


class FakeSession:
    """Stands in for AsyncSession; answers the two health queries."""

    def __init__(self, pgvector: bool = True, fail: bool = False) -> None:
        self.pgvector = pgvector
        self.fail = fail

    async def execute(self, statement):
        if self.fail:
            raise ConnectionError("db down")
        found = 1 if self.pgvector and "pg_extension" in str(statement) else None
        return SimpleNamespace(scalar_one_or_none=lambda: found)


def _client(session: FakeSession) -> TestClient:
    async def override():
        yield session

    app.dependency_overrides[get_db] = override
    return TestClient(app)  # no `with`: skips lifespan, so no real DB is needed


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_health_ok() -> None:
    res = _client(FakeSession()).get("/api/v1/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "database": True, "pgvector": True}


def test_health_database_down() -> None:
    res = _client(FakeSession(fail=True)).get("/api/v1/health")
    assert res.status_code == 503
    assert res.json()["database"] is False


def test_health_missing_pgvector_is_degraded() -> None:
    res = _client(FakeSession(pgvector=False)).get("/api/v1/health")
    assert res.status_code == 503
    body = res.json()
    assert body["database"] is True and body["pgvector"] is False


def _groq_settings(**kw) -> Settings:
    return Settings(_env_file=None, database_url="postgresql+psycopg://x/x", **kw)


def test_groq_health_unconfigured_hides_key() -> None:
    app.dependency_overrides[get_settings] = lambda: _groq_settings()
    res = TestClient(app).get("/api/v1/health/groq")
    assert res.status_code == 503
    assert res.json()["status"] == "unconfigured"


def test_groq_health_configured_never_leaks_key() -> None:
    app.dependency_overrides[get_settings] = lambda: _groq_settings(
        groq_api_key="super-secret", groq_model="m"
    )
    res = TestClient(app).get("/api/v1/health/groq")
    assert res.status_code == 200
    assert res.json()["configured"] is True
    assert "super-secret" not in res.text


def test_groq_probe_uses_mocked_api(monkeypatch: pytest.MonkeyPatch) -> None:
    async def create(**_):
        message = SimpleNamespace(content="x")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class Fake:
        def __init__(self, api_key: str, **options) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setattr(groq_client, "AsyncGroq", Fake)
    app.dependency_overrides[get_settings] = lambda: _groq_settings(
        groq_api_key="k", groq_model="m"
    )
    res = TestClient(app).get("/api/v1/health/groq?probe=true")
    assert res.status_code == 200
    assert res.json()["reachable"] is True
