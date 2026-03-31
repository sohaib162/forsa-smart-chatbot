"""
Document Ingestion Router
==========================
Implements ``POST /v1/ingest`` — upload and index documents into the
MinIO + Qdrant storage layer.

This is a Forsa-specific endpoint (not part of the OpenAI spec) that
enables enterprise document management without external tool dependencies.

After ingestion, documents are immediately available for RAG retrieval
via ``/v1/chat/completions``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from di import global_injector
from components.ingest.ingest_component import IngestComponent
from services.ingest_service import IngestService

logger = logging.getLogger("forsa.server.ingest")

router = APIRouter(prefix="/v1", tags=["Document Ingestion"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    """Response from a document ingestion request."""

    object: str = "ingest.result"
    filename: str
    s3_key: str = ""
    category: str = ""
    chunks_created: int = 0
    status: str  # "success" or "error"
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class IngestTextRequest(BaseModel):
    """Request to ingest raw text."""

    text: str = Field(..., description="Raw text to ingest")
    doc_id: str = Field(..., description="Unique document identifier")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata",
    )


class IngestStatusResponse(BaseModel):
    """Current ingestion status."""

    object: str = "ingest.status"
    total_chunks: int
    collection: str
    minio_available: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload and ingest a document",
    description=(
        "Upload a document (PDF, DOCX, ODT, JSON, TXT) to be stored in MinIO "
        "and indexed in the vector store for RAG retrieval."
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="Document file to ingest"),
    category: str = Form(
        default="Uncategorized",
        description="Document category (Guides, Offres, Conventions, Produits)",
    ),
    language: str = Form(
        default="FR",
        description="Document language (FR, AR)",
    ),
):
    """
    Upload and ingest a document.

    The document is:
    1. Uploaded to MinIO S3 for persistent storage
    2. Parsed to extract text content
    3. Chunked into overlapping segments
    4. Embedded and indexed in the vector store

    After ingestion, the document is immediately searchable via
    ``/v1/chat/completions``.
    """
    try:
        ingest_component = global_injector.get(IngestComponent)
        service = IngestService(ingest_component)

        result = await asyncio.to_thread(
            service.ingest_file,
            filename=file.filename or "unknown",
            file_data=file.file,
            category=category,
            language=language,
        )

        return result

    except Exception as e:
        logger.exception("Ingestion error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/ingest/text",
    summary="Ingest raw text",
    description="Directly ingest raw text content (for pre-parsed documents).",
)
async def ingest_text(request: IngestTextRequest):
    """Ingest raw text directly into the vector store."""
    try:
        ingest_component = global_injector.get(IngestComponent)
        service = IngestService(ingest_component)

        result = await asyncio.to_thread(
            service.ingest_text,
            text=request.text,
            doc_id=request.doc_id,
            metadata=request.metadata,
        )

        return result

    except Exception as e:
        logger.exception("Text ingestion error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/ingest/status",
    response_model=IngestStatusResponse,
    summary="Get ingestion status",
    description="Returns the current state of the document ingestion system.",
)
async def ingest_status():
    """Get ingestion system status."""
    try:
        ingest_component = global_injector.get(IngestComponent)
        service = IngestService(ingest_component)
        return service.get_status()
    except Exception as e:
        logger.exception("Ingest status error")
        raise HTTPException(status_code=500, detail=str(e))
