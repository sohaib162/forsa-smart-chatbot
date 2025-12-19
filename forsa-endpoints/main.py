"""
NGBSS Document Retrieval API
============================
FastAPI application for serving documents from MinIO S3 storage.
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import os
import json
from pathlib import Path

# Import the document router
from S3_Storage.fastapi_document_route import router as document_router

# Create FastAPI app
app = FastAPI(
    title="NGBSS Document Retrieval API",
    description="API for retrieving documents from MinIO S3 storage",
    version="1.0.0"
)

# CORS configuration - use env var for frontend origin
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
allowed_origins = [FRONTEND_ORIGIN]
if FRONTEND_ORIGIN != "http://localhost:5173":
    allowed_origins.append("http://localhost:5173")  # Always allow dev

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # helps downloads show filename
)


# Include document router
app.include_router(document_router)

# Path to the s3_index.json file
S3_INDEX_PATH = Path(__file__).parent / "S3_Storage" / "s3_index.json"


@app.get("/documents")
async def list_documents(
    category: Optional[str] = Query(None, description="Filter by category (Guides, Offres, Conventions, Produits)"),
    lang: Optional[str] = Query(None, description="Filter by language (AR, FR)"),
    q: Optional[str] = Query(None, description="Search substring in filename or s3_key")
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
        # Load the s3_index.json file
        if not S3_INDEX_PATH.exists():
            return {
                "error": "Index file not found",
                "message": "Please run upload_docs_and_index.py to generate the index",
                "documents": []
            }

        with open(S3_INDEX_PATH, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        # Apply filters
        filtered_docs = documents

        if category:
            filtered_docs = [doc for doc in filtered_docs if doc.get('category', '').lower() == category.lower()]

        if lang:
            filtered_docs = [doc for doc in filtered_docs if doc.get('lang', '').upper() == lang.upper()]

        if q:
            q_lower = q.lower()
            filtered_docs = [
                doc for doc in filtered_docs
                if q_lower in doc.get('filename', '').lower() or q_lower in doc.get('s3_key', '').lower()
            ]

        return {
            "total": len(filtered_docs),
            "documents": filtered_docs
        }

    except Exception as e:
        print(f"ERROR: Failed to load documents index: {e}")
        return {
            "error": "Failed to load documents",
            "message": str(e),
            "documents": []
        }


# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint - API information"""
    return {
        "message": "NGBSS Document Retrieval API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "documents": "/documents",
            "document": "/document/{s3_key:path}"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint for Docker"""
    return {
        "status": "healthy",
        "service": "ngbss-retrieval-api"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Print startup information"""
    print("="*80)
    print("NGBSS Document Retrieval API Started")
    print("="*80)
    print(f"Environment:")
    print(f"  S3_ENDPOINT: {os.getenv('S3_ENDPOINT', 'minio:9000')}")
    print(f"  S3_EXTERNAL_ENDPOINT: {os.getenv('S3_EXTERNAL_ENDPOINT', 'localhost:9010')}")
    print(f"  S3_BUCKET: {os.getenv('S3_BUCKET', 'forsa-documents')}")
    print(f"  API_BASE_URL: {os.getenv('API_BASE_URL', 'http://localhost:8000')}")
    print("="*80)
    print("API Endpoints:")
    print("  Root: http://localhost:8000/")
    print("  Health: http://localhost:8000/health")
    print("  Docs: http://localhost:8000/docs")
    print("  Document: http://localhost:8000/document/{s3_key}")
    print("="*80)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
