"""
EmbeddingComponent — Dependency-Injected Local Embedding Model
===============================================================
Wraps a local sentence-transformers model (default: multilingual-e5-small)
behind an injectable interface.

Features
--------
- Single-text and batch embedding
- Normalised embeddings for cosine similarity
- OpenAI-compatible ``embed()`` method returning float lists
- No external API calls — 100% offline

The model is loaded eagerly at DI bootstrap time so there is no
cold-start penalty on the first embedding request.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from settings import EmbeddingSettings

logger = logging.getLogger("forsa.components.embedding")


class EmbeddingComponent:
    """
    Injectable embedding component.

    Thread Safety
    -------------
    ``SentenceTransformer.encode()`` is thread-safe for read-only
    inference, so this component can be safely shared across async
    handlers via ``asyncio.to_thread()``.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        model_name = self._settings.model_name
        logger.info("EmbeddingComponent: Loading model '%s' …", model_name)
        t0 = time.perf_counter()

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        elapsed = time.perf_counter() - t0
        logger.info(
            "EmbeddingComponent: Model loaded in %.1fs (dim=%d)",
            elapsed,
            self.dimension,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def dimension(self) -> int:
        return self._settings.dimension

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string. Returns a list of floats."""
        embedding = self._model.encode(
            text,
            normalize_embeddings=self._settings.normalize,
        )
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns a list of float lists."""
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            batch_size=self._settings.batch_size,
            normalize_embeddings=self._settings.normalize,
        )
        return embeddings.tolist()

    def embed_numpy(self, text: str) -> np.ndarray:
        """Embed a single text. Returns a numpy array (for internal use)."""
        return self._model.encode(
            text,
            normalize_embeddings=self._settings.normalize,
        ).astype("float32")

    def embed_texts_numpy(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts. Returns a numpy matrix."""
        return self._model.encode(
            texts,
            batch_size=self._settings.batch_size,
            normalize_embeddings=self._settings.normalize,
        ).astype("float32")

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        emb_a = self.embed_numpy(text_a)
        emb_b = self.embed_numpy(text_b)
        return float(np.dot(emb_a, emb_b))
