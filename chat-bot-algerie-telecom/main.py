"""
Forsa Chatbot API
=================
FastAPI application serving the Forsa Smart Chatbot for Algérie Télécom.

All pipelines use the AdvancedPipeline (Graph-RAG + Citation-Faithful).
The LLM and KG graphs are pre-loaded at startup for instant response.

Endpoints:
  GET  /           — Health check (service status + model readiness)
  POST /process-question — Process a user question through the appropriate pipeline
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging (configure BEFORE imports that log on load)
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("forsa.chatbot")

# ---------------------------------------------------------------------------
# ⚡ EAGER PRELOAD: Load LLM into GPU memory IMMEDIATELY at import time.
# This ensures the model is hot and ready for the first query — no cold start.
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("FORSA CHATBOT API — Starting Pre-load Sequence")
logger.info("=" * 60)

from local_llm_client import preload_llm
_llm_client = preload_llm()

# Now import the pipelines — they'll build their KGs on first call (lazy)
# but the LLM is already loaded, which is the expensive part (~15-30s).
from pipelines.guide.guide import run_guide_pipeline
from pipelines.offers.offers import run_offers_pipeline
from pipelines.conventions.conventions import run_conventions_pipeline
from pipelines.depot.depot import run_depot_pipeline

logger.info("All pipeline modules loaded. KGs will build on first use.")
logger.info("=" * 60)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Forsa Chatbot API",
    version="2.0.0",
    description="Graph-RAG chatbot with citation-faithful generation for Algérie Télécom",
)

# CORS — restrict to known origins; override via ALLOWED_ORIGINS env var
_DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:8080,http://127.0.0.1:5173"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class QuestionBlock(BaseModel):
    categorie_id: Dict[str, str]


class ChatRequest(BaseModel):
    equipe: str
    question: QuestionBlock


class ChatResponse(BaseModel):
    status: str
    category: str
    answer: str
    sources: list
    faithfulness: Optional[float] = None
    kg_nodes_used: Optional[int] = None
    argument_types: Optional[list] = None
    latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Category Routing
# ---------------------------------------------------------------------------
CATEGORY_ID_TO_PIPELINE = {
    "1": "offres",
    "2": "conventions",
    "3": "guides",
    "4": "produits",
}

_PIPELINE_FN = {
    "offres":      run_offers_pipeline,
    "conventions": run_conventions_pipeline,
    "guides":      run_guide_pipeline,
    "produits":    run_depot_pipeline,
}

# Pipeline timeout in seconds
_PIPELINE_TIMEOUT = int(os.getenv("PIPELINE_TIMEOUT_SECONDS", "120"))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    """Health check — returns service status and model readiness."""
    return {
        "status": "running",
        "service": "forsa-chatbot-api",
        "version": "2.0.0",
        "model_loaded": _llm_client._model is not None,
        "pipelines": list(_PIPELINE_FN.keys()),
    }


@app.post("/process-question", response_model=ChatResponse)
async def process_question(payload: ChatRequest):
    """Process a user question by routing to the appropriate AdvancedPipeline."""
    t0 = time.perf_counter()

    if not payload.question.categorie_id:
        raise HTTPException(status_code=400, detail="Missing question.categorie_id")

    categorie_id, question = next(iter(payload.question.categorie_id.items()))
    category_name = CATEGORY_ID_TO_PIPELINE.get(str(categorie_id))
    if not category_name:
        raise HTTPException(
            status_code=400, detail=f"Invalid categorie_id: {categorie_id}"
        )

    pipeline_fn = _PIPELINE_FN.get(category_name)
    if not pipeline_fn:
        raise HTTPException(
            status_code=400, detail=f"Unsupported category: {category_name}"
        )

    logger.info(
        "Processing | category=%s | equipe=%s | q=%.80s…",
        category_name, payload.equipe, question,
    )

    # Run the pipeline in a thread to avoid blocking the async event loop
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
        "Done | category=%s | latency=%.0fms | answer_len=%d | sources=%d | faith=%.2f",
        category_name, latency_ms, len(answer), len(sources),
        (result or {}).get("faithfulness", 0.0) or 0.0,
    )

    return ChatResponse(
        status="success",
        category=category_name,
        answer=answer,
        sources=sources,
        faithfulness=(result or {}).get("faithfulness"),
        kg_nodes_used=(result or {}).get("kg_nodes_used"),
        argument_types=(result or {}).get("argument_types"),
        latency_ms=round(latency_ms, 1),
    )
