"""
NGBSS Document Retrieval API
============================
FastAPI application for serving documents from MinIO S3 storage.
"""
import logging
import os
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# Import the document router
from S3_Storage.fastapi_document_route import router as document_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("forsa.retrieval")

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — runs startup and shutdown logic."""
    logger.info("=" * 60)
    logger.info("NGBSS Document Retrieval API Starting")
    logger.info("=" * 60)
    logger.info("S3_ENDPOINT:          %s", os.getenv("S3_ENDPOINT", "minio:9000"))
    logger.info("S3_EXTERNAL_ENDPOINT: %s", os.getenv("S3_EXTERNAL_ENDPOINT", "localhost:9010"))
    logger.info("S3_BUCKET:            %s", os.getenv("S3_BUCKET", "forsa-documents"))
    logger.info("=" * 60)
    yield
    logger.info("NGBSS Document Retrieval API Shutting Down")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NGBSS Document Retrieval API",
    description="API for retrieving documents from MinIO S3 storage",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS configuration
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
allowed_origins = [FRONTEND_ORIGIN]
if FRONTEND_ORIGIN != "http://localhost:5173":
    allowed_origins.append("http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Disposition"],
)

# Include document router
app.include_router(document_router)

# Path to the s3_index.json file
S3_INDEX_PATH = Path(__file__).parent / "S3_Storage" / "s3_index.json"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/documents")
async def list_documents(
    category: Optional[str] = Query(None, description="Filter by category (Guides, Offres, Conventions, Produits)"),
    lang: Optional[str] = Query(None, description="Filter by language (AR, FR)"),
    q: Optional[str] = Query(None, description="Search substring in filename or s3_key"),
):
    """
    List all documents from the S3 index with optional filtering.

    Query parameters:
    - category: Filter by category (Guides, Offres, Conventions, Produits)
    - lang: Filter by language (AR, FR)
    - q: Search substring in filename or s3_key (case-insensitive)

    Returns:
        List of document objects with fields: s3_key, filename, category, ext, lang
    """
    try:
        if not S3_INDEX_PATH.exists():
            logger.warning("S3 index file not found at %s", S3_INDEX_PATH)
            return {
                "error": "Index file not found",
                "message": "Please run upload_docs_and_index.py to generate the index",
                "documents": [],
            }

        with open(S3_INDEX_PATH, "r", encoding="utf-8") as f:
            documents = json.load(f)

        # Apply filters
        filtered_docs = documents

        if category:
            filtered_docs = [
                doc for doc in filtered_docs
                if doc.get("category", "").lower() == category.lower()
            ]

        if lang:
            filtered_docs = [
                doc for doc in filtered_docs
                if doc.get("lang", "").upper() == lang.upper()
            ]

        if q:
            q_lower = q.lower()
            filtered_docs = [
                doc for doc in filtered_docs
                if q_lower in doc.get("filename", "").lower()
                or q_lower in doc.get("s3_key", "").lower()
            ]

        return {"total": len(filtered_docs), "documents": filtered_docs}

    except Exception as e:
        logger.exception("Failed to load documents index")
        return {
            "error": "Failed to load documents",
            "message": str(e),
            "documents": [],
        }


@app.get("/")
def read_root():
    """Root endpoint — API information."""
    return {
        "message": "NGBSS Document Retrieval API",
        "version": "1.1.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "documents": "/documents",
            "document": "/document/{s3_key:path}",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint for Docker."""
    return {"status": "healthy", "service": "ngbss-retrieval-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
