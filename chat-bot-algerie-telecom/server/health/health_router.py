"""
Health Check Router
====================
Provides ``GET /v1/health`` — system health with per-component status.

Also exposes ``GET /v1/models`` for OpenAI-compatible model listing.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from di import global_injector
from components.llm.llm_component import LLMComponent
from components.embedding.embedding_component import EmbeddingComponent
from components.vector_store.vector_store_component import VectorStoreComponent
from components.ingest.ingest_component import IngestComponent

logger = logging.getLogger("forsa.server.health")

router = APIRouter(tags=["Health & Models"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ComponentStatus(BaseModel):
    name: str
    ready: bool
    details: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str  # "healthy" or "degraded"
    version: str = "3.0.0"
    service: str = "forsa-chatbot-api"
    architecture: str = "privategpt-style"
    components: List[ComponentStatus]
    uptime_seconds: float = 0.0


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "forsa-local"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_start_time = time.time()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Returns the health status of all system components.",
)
async def health_check():
    """Comprehensive health check with per-component status."""
    components = []
    all_ready = True

    # LLM
    try:
        llm = global_injector.get(LLMComponent)
        components.append(
            ComponentStatus(
                name="llm",
                ready=llm.is_ready,
                details={"model": llm.model_name},
            )
        )
        if not llm.is_ready:
            all_ready = False
    except KeyError:
        components.append(ComponentStatus(name="llm", ready=False))
        all_ready = False

    # Embedding
    try:
        embedding = global_injector.get(EmbeddingComponent)
        components.append(
            ComponentStatus(
                name="embedding",
                ready=embedding.is_ready,
                details={
                    "model": embedding.model_name,
                    "dimension": embedding.dimension,
                },
            )
        )
        if not embedding.is_ready:
            all_ready = False
    except KeyError:
        components.append(ComponentStatus(name="embedding", ready=False))
        all_ready = False

    # Vector Store
    try:
        vector_store = global_injector.get(VectorStoreComponent)
        components.append(
            ComponentStatus(
                name="vector_store",
                ready=vector_store.is_ready,
                details={
                    "collection": vector_store.collection_name,
                    "documents": vector_store.count(),
                },
            )
        )
        if not vector_store.is_ready:
            all_ready = False
    except KeyError:
        components.append(ComponentStatus(name="vector_store", ready=False))
        all_ready = False

    # Ingest (MinIO)
    try:
        ingest = global_injector.get(IngestComponent)
        components.append(
            ComponentStatus(
                name="ingest",
                ready=ingest.is_ready,
                details={"minio_available": ingest.minio_available},
            )
        )
    except KeyError:
        components.append(ComponentStatus(name="ingest", ready=False))

    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        components=components,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get(
    "/v1/models",
    response_model=ModelsResponse,
    summary="List available models",
    description="OpenAI-compatible model listing endpoint.",
)
async def list_models():
    """List available models (OpenAI-compatible)."""
    models = []

    try:
        llm = global_injector.get(LLMComponent)
        models.append(
            ModelInfo(
                id=llm.model_name,
                created=int(_start_time),
            )
        )
    except KeyError:
        pass

    try:
        embedding = global_injector.get(EmbeddingComponent)
        models.append(
            ModelInfo(
                id=embedding.model_name,
                created=int(_start_time),
            )
        )
    except KeyError:
        pass

    return ModelsResponse(data=models)
