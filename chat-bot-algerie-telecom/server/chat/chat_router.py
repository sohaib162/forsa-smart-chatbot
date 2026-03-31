"""
OpenAI-Compatible Chat Completions Router
==========================================
Implements ``POST /v1/chat/completions`` — a direct drop-in replacement
for the OpenAI Chat API.

Any enterprise software that expects an OpenAI connection (e.g. LangChain,
LibreChat, Continue.dev) can point to this endpoint and work seamlessly.

Request/Response schemas match the OpenAI specification exactly.
Forsa-specific extensions (sources, faithfulness) are added as extra fields
that do not break compatibility.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from di import global_injector
from components.llm.llm_component import LLMComponent
from components.embedding.embedding_component import EmbeddingComponent
from components.vector_store.vector_store_component import VectorStoreComponent
from services.chat_service import ChatService

logger = logging.getLogger("forsa.server.chat")

router = APIRouter(prefix="/v1", tags=["Chat Completions"])


# ---------------------------------------------------------------------------
# Request / Response Models (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(
        default="forsa-qwen-2.5-3b",
        description="Model to use (ignored — always uses the local model)",
    )
    messages: List[ChatMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    max_tokens: Optional[int] = Field(
        default=512,
        description="Maximum tokens to generate",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature (0.0 - 2.0)",
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Top-p (nucleus) sampling",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response",
    )
    # Forsa-specific extensions
    use_context: bool = Field(
        default=True,
        description="[Forsa] Whether to retrieve document context via RAG",
    )
    include_sources: bool = Field(
        default=True,
        description="[Forsa] Whether to include source documents in response",
    )
    context_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="[Forsa] Metadata filter for context retrieval",
    )


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    # Forsa extensions
    sources: Optional[List[Dict[str, Any]]] = None
    latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create a chat completion",
    description=(
        "OpenAI-compatible chat completion endpoint. "
        "Uses local Qwen 2.5 3B model with RAG context from MinIO-stored "
        "Algérie Télécom documents."
    ),
)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion.

    This endpoint is fully compatible with the OpenAI Chat API specification.
    Point any OpenAI client at ``http://<host>:8001/v1/chat/completions``
    and it will work as a drop-in replacement.
    """
    try:
        llm = global_injector.get(LLMComponent)
        embedding = global_injector.get(EmbeddingComponent)
        vector_store = global_injector.get(VectorStoreComponent)

        service = ChatService(
            llm=llm,
            embedding=embedding,
            vector_store=vector_store,
        )

        messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]

        if request.stream:
            stream_gen = service.chat_completion(
                messages=messages,
                use_context=request.use_context,
                context_filter=request.context_filter,
                include_sources=request.include_sources,
                max_new_tokens=request.max_tokens or 512,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=True,
            )
            return StreamingResponse(
                stream_gen,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming
        result = await asyncio.to_thread(
            service.chat_completion,
            messages=messages,
            use_context=request.use_context,
            context_filter=request.context_filter,
            include_sources=request.include_sources,
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False,
        )

        return result

    except Exception as e:
        logger.exception("Chat completion error")
        raise HTTPException(status_code=500, detail=str(e))
