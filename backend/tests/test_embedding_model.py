"""Uses the real local FastEmbed model; skipped if it cannot be loaded (e.g. offline)."""

import pytest

from app.core.config import Settings
from app.embeddings.model import EmbeddingModel, EmbeddingModelError, declared_dimension


@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    name = Settings(_env_file=None, database_url="postgresql+psycopg://x/x").embedding_model
    m = EmbeddingModel(name, batch_size=8)
    try:
        m.load()
    except EmbeddingModelError as exc:
        pytest.skip(f"embedding model unavailable: {exc}")
    return m


def test_model_initializes_and_dimension_matches_actual_output(model: EmbeddingModel) -> None:
    (vector,) = model.embed(["Laptops must include a three year warranty."])
    assert len(vector) == model.dimension == declared_dimension(model.name)


def test_same_text_same_vector_and_batch_shapes(model: EmbeddingModel) -> None:
    texts = ["alpha bid", "beta addendum", "alpha bid"]
    vectors = model.embed(texts)
    assert len(vectors) == 3
    assert all(len(v) == model.dimension for v in vectors)
    assert vectors[0] == pytest.approx(vectors[2], abs=1e-5)
    assert vectors[0] != pytest.approx(vectors[1], abs=1e-3)


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(EmbeddingModelError):
        EmbeddingModel("not/a-real-model")
