"""Query embeddings use the same model as documents, plus the BGE query instruction."""

import pytest

from app.core.config import Settings
from app.embeddings.model import BGE_QUERY_PREFIX, EmbeddingModel, EmbeddingModelError, query_prefix
from app.embeddings.service import EmbeddingService
from tests.fakes import FakeEmbeddingModel


@pytest.mark.parametrize(
    ("name", "prefix"),
    [
        ("BAAI/bge-small-en-v1.5", BGE_QUERY_PREFIX),
        ("BAAI/bge-base-en-v1.5", BGE_QUERY_PREFIX),
        ("BAAI/bge-small-en", BGE_QUERY_PREFIX),
        ("sentence-transformers/all-MiniLM-L6-v2", ""),
    ],
)
def test_query_prefix_per_model(name: str, prefix: str) -> None:
    assert query_prefix(name) == prefix


def test_default_model_is_the_document_model() -> None:
    settings = Settings(_env_file=None, database_url="postgresql+psycopg://x/x")
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"


async def test_service_embeds_query_and_rejects_blank() -> None:
    service = EmbeddingService(None, FakeEmbeddingModel(), 8)  # type: ignore[arg-type]
    assert len(await service.embed_query("final deadline")) == 384
    with pytest.raises(ValueError):
        await service.embed_query("  ")


@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    m = EmbeddingModel(Settings(_env_file=None, database_url="postgresql+psycopg://x/x").embedding_model)
    try:
        m.load()
    except EmbeddingModelError as exc:
        pytest.skip(f"embedding model unavailable: {exc}")
    return m


def test_real_query_vector_has_384_dimensions(model: EmbeddingModel) -> None:
    assert len(model.embed_query("What is the final submission deadline?")) == 384


def test_real_query_gets_prefix_and_documents_do_not(model: EmbeddingModel) -> None:
    text = "What is the final submission deadline?"
    query_vec = model.embed_query(text)
    prefixed = model.embed([BGE_QUERY_PREFIX + text])[0]
    plain = model.embed([text])[0]  # how a document chunk with this text is embedded

    assert query_vec == pytest.approx(prefixed, abs=1e-6)
    assert query_vec != pytest.approx(plain, abs=1e-3)
