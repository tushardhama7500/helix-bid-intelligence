"""Local FastEmbed model, loaded once and reused."""

import logging
import re
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)


class EmbeddingModelError(RuntimeError):
    """The embedding model could not be loaded or produced invalid output."""


@lru_cache
def _supported_dimensions() -> dict[str, int]:
    from fastembed import TextEmbedding

    return {m["model"]: m["dim"] for m in TextEmbedding.list_supported_models()}


def declared_dimension(model_name: str) -> int:
    """Dimension from FastEmbed's model metadata (no download needed).

    The real output size is re-checked when the model loads, so a wrong value
    here can never write vectors of the wrong size.
    """
    try:
        return _supported_dimensions()[model_name]
    except KeyError:
        raise EmbeddingModelError(
            f"Unsupported EMBEDDING_MODEL {model_name!r}. "
            "See fastembed.TextEmbedding.list_supported_models()."
        ) from None


# BGE retrieval models are trained to match short queries to passages when the
# query (and only the query) carries this instruction. Documents get no prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def query_prefix(model_name: str) -> str:
    """Instruction prefix the model expects on search queries ('' if none)."""
    return BGE_QUERY_PREFIX if re.fullmatch(r"BAAI/bge-.+-en(-v1\.5)?", model_name) else ""


class EmbeddingModel:
    def __init__(self, name: str, batch_size: int = 32) -> None:
        self.name = name
        self.batch_size = batch_size
        self.dimension = declared_dimension(name)
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the ONNX model (downloads it on first use) and verify its output size."""
        with self._lock:
            if self._model is not None:
                return
            try:
                from fastembed import TextEmbedding

                logger.info("Loading embedding model %s", self.name)
                model = TextEmbedding(model_name=self.name)
                actual = len(next(iter(model.embed(["dimension check"]))))
            except Exception as exc:
                raise EmbeddingModelError(
                    f"Could not load embedding model {self.name!r}: {exc}"
                ) from exc
            if actual != self.dimension:
                raise EmbeddingModelError(
                    f"Model {self.name!r} produced {actual}-d vectors, expected {self.dimension}"
                )
            self._model = model
            logger.info("Embedding model ready: %s (%d dimensions)", self.name, actual)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of non-empty texts (blocking; call from a worker thread)."""
        self.load()
        vectors = [v.tolist() for v in self._model.embed(texts, batch_size=self.batch_size)]
        if len(vectors) != len(texts) or any(len(v) != self.dimension for v in vectors):
            raise EmbeddingModelError("Embedding model returned unexpected output shape")
        return vectors


    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with the model's query prefix (blocking)."""
        (vector,) = self.embed([query_prefix(self.name) + query])
        return vector


@lru_cache
def get_embedding_model(name: str, batch_size: int) -> EmbeddingModel:
    """Process-wide model instance, so the model is loaded once."""
    return EmbeddingModel(name, batch_size)
