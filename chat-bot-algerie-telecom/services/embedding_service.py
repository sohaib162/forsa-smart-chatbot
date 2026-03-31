"""
EmbeddingService — Text Embedding for OpenAI-Compatible API
=============================================================
Provides text embedding via the local sentence-transformers model.
Returns embeddings in the exact OpenAI ``/v1/embeddings`` response format.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Union

from components.embedding.embedding_component import EmbeddingComponent

logger = logging.getLogger("forsa.services.embedding")


class EmbeddingService:
    """
    Embedding service for the OpenAI-compatible API.

    Wraps the EmbeddingComponent with OpenAI response formatting.
    """

    def __init__(self, embedding: EmbeddingComponent) -> None:
        self._embedding = embedding

    def create_embeddings(
        self,
        input_texts: Union[str, List[str]],
        model: str = "default",
    ) -> Dict[str, Any]:
        """
        Generate embeddings for one or more texts.

        Parameters
        ----------
        input_texts : A single string or a list of strings to embed.
        model       : Model name (ignored — always uses the configured model).

        Returns
        -------
        OpenAI-compatible embeddings response dict.
        """
        t0 = time.perf_counter()

        if isinstance(input_texts, str):
            input_texts = [input_texts]

        embeddings = self._embedding.embed_texts(input_texts)

        data = [
            {
                "object": "embedding",
                "embedding": emb,
                "index": i,
            }
            for i, emb in enumerate(embeddings)
        ]

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Generated %d embeddings in %.1fms", len(input_texts), latency_ms
        )

        return {
            "object": "list",
            "data": data,
            "model": self._embedding.model_name,
            "usage": {
                "prompt_tokens": sum(len(t.split()) for t in input_texts),
                "total_tokens": sum(len(t.split()) for t in input_texts),
            },
        }
