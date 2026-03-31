"""
OpenAI-Compatible Embeddings Router
=====================================
Implements ``POST /v1/embeddings`` — a drop-in replacement for the
OpenAI Embeddings API.

Uses our local ``multilingual-e5-small`` model (or whichever model
is configured in EmbeddingSettings). All computation runs locally —
zero data leaves the server.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from di import global_injector
from components.embedding.embedding_component import EmbeddingComponent
from services.embedding_service import EmbeddingService

logger = logging.getLogger("forsa.server.embeddings")

router = APIRouter(prefix="/v1", tags=["Embeddings"])


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    input: Union[str, List[str]] = Field(
        ..., description="Input text(s) to generate embeddings for"
    )
    model: str = Field(
        default="forsa-e5-small",
        description="Model name (ignored — uses configured model)",
    )
    encoding_format: str = Field(
        default="float",
        description="Encoding format ('float' or 'base64')",
    )


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""

    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    summary="Create embeddings",
    description=(
        "OpenAI-compatible embedding endpoint. Uses local multilingual-e5-small "
        "model. All computation is performed locally — zero data egress."
    ),
)
async def create_embedding(request: EmbeddingRequest):
    """
    Create embeddings for the given input text(s).

    Fully compatible with OpenAI's ``/v1/embeddings`` endpoint.
    """
    try:
        embedding = global_injector.get(EmbeddingComponent)
        service = EmbeddingService(embedding)

        result = await asyncio.to_thread(
            service.create_embeddings,
            input_texts=request.input,
            model=request.model,
        )

        return result

    except Exception as e:
        logger.exception("Embedding error")
        raise HTTPException(status_code=500, detail=str(e))
