"""
Forsa Chatbot — Dependency Injection Container
================================================
Lightweight DI container inspired by PrivateGPT's injector pattern.

Creates and caches component singletons so every service/router gets the
same pre-loaded LLM, embedding model, vector store, etc.

Usage
-----
    from di import global_injector
    llm = global_injector.get(LLMComponent)
    embeddings = global_injector.get(EmbeddingComponent)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Type, TypeVar

from settings import settings

logger = logging.getLogger("forsa.di")

T = TypeVar("T")


class DIContainer:
    """
    Simple singleton-based dependency injection container.

    Components register themselves during ``bootstrap()`` and are retrieved
    via ``get(ComponentClass)``.  All components are eagerly initialised at
    startup so there are no cold-start surprises at query time.
    """

    def __init__(self):
        self._registry: Dict[Type, Any] = {}
        self._bootstrapped = False

    def register(self, cls: Type[T], instance: T) -> None:
        """Register a component instance."""
        self._registry[cls] = instance
        logger.debug("Registered component: %s", cls.__name__)

    def get(self, cls: Type[T]) -> T:
        """Retrieve a registered component. Raises KeyError if not found."""
        if cls not in self._registry:
            raise KeyError(
                f"Component {cls.__name__} not registered. "
                f"Call bootstrap() first."
            )
        return self._registry[cls]

    def has(self, cls: Type) -> bool:
        """Check if a component is registered."""
        return cls in self._registry

    @property
    def is_bootstrapped(self) -> bool:
        return self._bootstrapped

    def bootstrap(self) -> None:
        """
        Eagerly initialise all components in dependency order.

        This is called once at application startup (in main.py lifespan).
        Component constructors may be slow (loading GPU models), so this
        ensures the system is fully ready before serving requests.
        """
        if self._bootstrapped:
            logger.warning("DI container already bootstrapped — skipping.")
            return

        cfg = settings()

        logger.info("=" * 60)
        logger.info("FORSA DI CONTAINER — Bootstrap Sequence")
        logger.info("=" * 60)

        # ── 1. LLM Component ──────────────────────────────────────────
        from components.llm.llm_component import LLMComponent

        llm = LLMComponent(cfg.llm)
        self.register(LLMComponent, llm)

        # ── 2. Embedding Component ────────────────────────────────────
        from components.embedding.embedding_component import EmbeddingComponent

        embedding = EmbeddingComponent(cfg.embedding)
        self.register(EmbeddingComponent, embedding)

        # ── 3. Vector Store Component ─────────────────────────────────
        from components.vector_store.vector_store_component import (
            VectorStoreComponent,
        )

        vector_store = VectorStoreComponent(
            settings=cfg.vector_store,
            embedding_component=embedding,
        )
        self.register(VectorStoreComponent, vector_store)

        # ── 4. Ingest Component ───────────────────────────────────────
        from components.ingest.ingest_component import IngestComponent

        ingest = IngestComponent(
            ingest_settings=cfg.ingest,
            minio_settings=cfg.minio,
            vector_store=vector_store,
            embedding_component=embedding,
        )
        self.register(IngestComponent, ingest)

        self._bootstrapped = True
        logger.info("=" * 60)
        logger.info("DI CONTAINER — All components ready")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
global_injector = DIContainer()
