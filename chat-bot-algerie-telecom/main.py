"""
Forsa Chatbot API — PrivateGPT-Style Architecture
====================================================
FastAPI application with dependency-injected components,
OpenAI-compatible API layer, and zero external data leakage.

Architecture
------------
- **Components**: LLM, Embedding, VectorStore, Ingest (DI-managed)
- **Services**: ChatService, EmbeddingService, IngestService
- **API**: OpenAI-compatible endpoints (/v1/chat/completions, /v1/embeddings)
- **Legacy**: Backward-compatible /process-question endpoint

Startup Flow
------------
1. Load settings from environment variables
2. Bootstrap DI container (eagerly loads models)
3. Register OpenAI-compatible routers
4. Register legacy router for backward compatibility
5. Start serving requests
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging (configure BEFORE any component imports)
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", os.getenv("SERVER_LOG_LEVEL", "INFO")).upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("forsa.chatbot")

# ---------------------------------------------------------------------------
# Settings & DI
# ---------------------------------------------------------------------------
from settings import settings
from di import global_injector


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — bootstrap all components at startup.

    The DI container eagerly loads the LLM into GPU memory, the embedding
    model, initialises Qdrant, and connects to MinIO.  This ensures zero
    cold-start latency when the first query arrives.
    """
    logger.info("=" * 60)
    logger.info("FORSA CHATBOT API v3.0.0 — PrivateGPT Architecture")
    logger.info("=" * 60)
    logger.info("Environment: %s", settings().server.env_name)
    logger.info("LLM Model:   %s", settings().llm.model_name)
    logger.info("Embed Model: %s", settings().embedding.model_name)
    logger.info("Vector Store: %s", settings().vector_store.provider)
    logger.info("MinIO:        %s", settings().minio.endpoint)
    logger.info("=" * 60)

    # Bootstrap all components
    global_injector.bootstrap()

    # Eagerly load legacy pipelines so they're warm
    _preload_legacy_pipelines()

    logger.info("=" * 60)
    logger.info("ALL SYSTEMS READY — Accepting requests")
    logger.info("=" * 60)

    yield

    logger.info("FORSA CHATBOT API — Shutting down")


def _preload_legacy_pipelines():
    """Pre-import legacy pipeline modules so their KGs build during startup."""
    try:
        from local_llm_client import preload_llm
        _llm_client = preload_llm()

        from pipelines.guide.guide import run_guide_pipeline
        from pipelines.offers.offers import run_offers_pipeline
        from pipelines.conventions.conventions import run_conventions_pipeline
        from pipelines.depot.depot import run_depot_pipeline
        logger.info("Legacy pipeline modules pre-loaded")
    except Exception as e:
        logger.warning("Legacy pipeline pre-load failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Forsa Chatbot API",
    version="3.0.0",
    description=(
        "Enterprise-grade, OpenAI-compatible chatbot API for Algérie Télécom. "
        "100% offline, data-sovereign, with PrivateGPT-style architecture."
    ),
    lifespan=lifespan,
)

# CORS
cfg = settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors.allow_origins,
    allow_credentials=cfg.cors.allow_credentials,
    allow_methods=cfg.cors.allow_methods,
    allow_headers=cfg.cors.allow_headers,
)

# ---------------------------------------------------------------------------
# Register OpenAI-Compatible API Routers
# ---------------------------------------------------------------------------
from server.chat.chat_router import router as chat_router
from server.embeddings.embeddings_router import router as embeddings_router
from server.ingest.ingest_router import router as ingest_router
from server.health.health_router import router as health_router

app.include_router(chat_router)
app.include_router(embeddings_router)
app.include_router(ingest_router)
app.include_router(health_router)


# ---------------------------------------------------------------------------
# Legacy Endpoints (Backward Compatibility)
# ---------------------------------------------------------------------------
# These endpoints mirror the original main.py so existing clients
# (e.g. the forsa-frontend) continue to work during the transition.

class QuestionBlock(BaseModel):
    categorie_id: Dict[str, str]


class LegacyChatRequest(BaseModel):
    equipe: str
    question: QuestionBlock


class LegacyChatResponse(BaseModel):
    status: str
    category: str
    answer: str
    sources: list
    faithfulness: Optional[float] = None
    kg_nodes_used: Optional[int] = None
    argument_types: Optional[list] = None
    latency_ms: Optional[float] = None


# Category routing (preserved from original)
CATEGORY_ID_TO_PIPELINE = {
    "1": "offres",
    "2": "conventions",
    "3": "guides",
    "4": "produits",
}

_PIPELINE_TIMEOUT = int(os.getenv("PIPELINE_TIMEOUT_SECONDS", "120"))


@app.get("/")
def root_health_check():
    """Health check — backward-compatible with the original API."""
    return {
        "status": "running",
        "service": "forsa-chatbot-api",
        "version": "3.0.0",
        "architecture": "privategpt-style",
        "endpoints": {
            "openai_chat": "/v1/chat/completions",
            "openai_embeddings": "/v1/embeddings",
            "ingest": "/v1/ingest",
            "health": "/v1/health",
            "models": "/v1/models",
            "legacy_process": "/process-question",
        },
    }


@app.post("/process-question", response_model=LegacyChatResponse)
async def process_question(payload: LegacyChatRequest):
    """
    Legacy endpoint — routes to the appropriate pipeline.

    Preserved for backward compatibility with the forsa-frontend.
    New integrations should use ``/v1/chat/completions`` instead.
    """
    t0 = time.perf_counter()

    if not payload.question.categorie_id:
        raise HTTPException(status_code=400, detail="Missing question.categorie_id")

    categorie_id, question = next(iter(payload.question.categorie_id.items()))
    category_name = CATEGORY_ID_TO_PIPELINE.get(str(categorie_id))
    if not category_name:
        raise HTTPException(
            status_code=400, detail=f"Invalid categorie_id: {categorie_id}"
        )

    # Lazy import to avoid startup penalty if not used
    _PIPELINE_FN = _get_pipeline_fn_map()
    pipeline_fn = _PIPELINE_FN.get(category_name)
    if not pipeline_fn:
        raise HTTPException(
            status_code=400, detail=f"Unsupported category: {category_name}"
        )

    logger.info(
        "Legacy | Processing | category=%s | equipe=%s | q=%.80s…",
        category_name, payload.equipe, question,
    )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(pipeline_fn, question),
            timeout=_PIPELINE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Pipeline timed out after %ds | category=%s",
            _PIPELINE_TIMEOUT, category_name,
        )
        raise HTTPException(
            status_code=504,
            detail=f"Pipeline timed out after {_PIPELINE_TIMEOUT}s",
        )
    except Exception:
        logger.exception("Pipeline error | category=%s", category_name)
        raise HTTPException(status_code=500, detail="Internal pipeline error")

    answer = (result or {}).get("answer", "")
    sources = (result or {}).get("sources", [])
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Legacy | Done | category=%s | latency=%.0fms | answer_len=%d",
        category_name, latency_ms, len(answer),
    )

    return LegacyChatResponse(
        status="success",
        category=category_name,
        answer=answer,
        sources=sources,
        faithfulness=(result or {}).get("faithfulness"),
        kg_nodes_used=(result or {}).get("kg_nodes_used"),
        argument_types=(result or {}).get("argument_types"),
        latency_ms=round(latency_ms, 1),
    )


# ---------------------------------------------------------------------------
# Pipeline Function Map (lazy-loaded)
# ---------------------------------------------------------------------------
_pipeline_fn_map = None


def _get_pipeline_fn_map():
    global _pipeline_fn_map
    if _pipeline_fn_map is not None:
        return _pipeline_fn_map

    try:
        from pipelines.guide.guide import run_guide_pipeline
        from pipelines.offers.offers import run_offers_pipeline
        from pipelines.conventions.conventions import run_conventions_pipeline
        from pipelines.depot.depot import run_depot_pipeline

        _pipeline_fn_map = {
            "offres": run_offers_pipeline,
            "conventions": run_conventions_pipeline,
            "guides": run_guide_pipeline,
            "produits": run_depot_pipeline,
        }
    except ImportError as e:
        logger.warning("Legacy pipelines not available: %s", e)
        _pipeline_fn_map = {}

    return _pipeline_fn_map


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.server.log_level.lower(),
        reload=cfg.server.env_name == "local",
    )
