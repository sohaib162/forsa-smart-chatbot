"""
FastAPI Web Service for NGBSS Retrieval
=======================================

Provides REST API for searching guides and retrieving documents.

Run with: uvicorn scripts.api_server:app --reload --port 8000

Endpoints:
- GET  /search?q=query&top_k=5&doc_type=section
- GET  /document/{guide_id}
- GET  /health
"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.retrieval_api import NGBSSRetriever, SearchResult

# =============================================================================
# Configuration
# =============================================================================

# Update this to your documents root
DOCUMENTS_ROOT = None  # e.g., Path("/path/to/your/documents")

# =============================================================================
# API Models
# =============================================================================

class SearchResultModel(BaseModel):
    rank: int
    score: float
    guide_title: str
    section_title: Optional[str]
    summary: str
    tags: List[str]
    filename: str
    relative_path: str
    document_path: Optional[str]
    s3_key: Optional[str] = None
    url: Optional[str] = None
    guide_id: str
    doc_id: str
    doc_type: str
    text_preview: str

class SearchResponse(BaseModel):
    query: str
    total_results: int
    timing_ms: float
    filters_applied: List[str]
    results: List[SearchResultModel]

class HealthResponse(BaseModel):
    status: str
    indexes_loaded: bool
    documents_root_configured: bool

# =============================================================================
# Application
# =============================================================================

app = FastAPI(
    title="NGBSS Guide Retrieval API",
    description="Fast hybrid search for NGBSS guides with document retrieval",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever instance
retriever: Optional[NGBSSRetriever] = None

# =============================================================================
# Lifecycle Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize retriever on startup"""
    global retriever
    print("🚀 Starting NGBSS Retrieval API...")
    retriever = NGBSSRetriever(
        documents_root=DOCUMENTS_ROOT,
        enable_reranking=False,
        auto_connect=True
    )
    print("✓ Retriever initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global retriever
    if retriever:
        retriever.close()
    print("👋 Retriever shutdown complete")

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy",
        indexes_loaded=retriever is not None and retriever._connected,
        documents_root_configured=DOCUMENTS_ROOT is not None
    )

@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results"),
    doc_type: str = Query("section", description="Document type: guide, section, step"),
    use_filters: bool = Query(True, description="Apply metadata filters")
):
    """
    Search for relevant guides/sections
    
    - **q**: Natural language query (e.g., "TVA 2%", "réactiver abonné")
    - **top_k**: Number of results to return (1-50)
    - **doc_type**: Granularity level - guide, section (recommended), or step
    - **use_filters**: Extract keywords to filter results
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    if doc_type not in ["guide", "section", "step"]:
        raise HTTPException(status_code=400, detail="doc_type must be guide, section, or step")
    
    results, timing = retriever.search_with_timing(
        query=q,
        top_k=top_k,
        doc_type=doc_type,
        use_filters=use_filters
    )
    
    return SearchResponse(
        query=q,
        total_results=len(results),
        timing_ms=round(timing['total_ms'], 2),
        filters_applied=timing.get('filters_applied', []),
        results=[
            SearchResultModel(
                rank=r.rank,
                score=r.score,
                guide_title=r.guide_title,
                section_title=r.section_title,
                summary=r.summary,
                tags=r.tags,
                filename=r.filename,
                relative_path=r.relative_path,
                document_path=str(r.document_path) if r.document_path else None,
                s3_key=r.s3_key,
                url=r.url,
                guide_id=r.guide_id,
                doc_id=r.doc_id,
                doc_type=r.doc_type,
                text_preview=r.text_preview
            )
            for r in results
        ]
    )

@app.get("/search_with_urls", response_model=SearchResponse)
async def search_with_urls(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results"),
    doc_type: str = Query("section", description="Document type: guide, section, step"),
    use_filters: bool = Query(True, description="Apply metadata filters")
):
    """
    Search for relevant guides/sections with presigned S3 URLs for document access.
    
    This endpoint generates presigned URLs that allow direct access to documents
    stored in S3/MinIO without requiring additional authentication.
    
    - **q**: Natural language query (e.g., "TVA 2%", "réactiver abonné")
    - **top_k**: Number of results to return (1-50)
    - **doc_type**: Granularity level - guide, section (recommended), or step
    - **use_filters**: Extract keywords to filter results
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    if doc_type not in ["guide", "section", "step"]:
        raise HTTPException(status_code=400, detail="doc_type must be guide, section, or step")
    
    results, timing = retriever.search_with_urls(
        query=q,
        top_k=top_k,
        doc_type=doc_type,
        use_filters=use_filters
    )
    
    return SearchResponse(
        query=q,
        total_results=len(results),
        timing_ms=round(timing['total_ms'], 2),
        filters_applied=timing.get('filters_applied', []),
        results=[
            SearchResultModel(
                rank=r.rank,
                score=r.score,
                guide_title=r.guide_title,
                section_title=r.section_title,
                summary=r.summary,
                tags=r.tags,
                filename=r.filename,
                relative_path=r.relative_path,
                document_path=str(r.document_path) if r.document_path else None,
                s3_key=r.s3_key,
                url=r.url,
                guide_id=r.guide_id,
                doc_id=r.doc_id,
                doc_type=r.doc_type,
                text_preview=r.text_preview
            )
            for r in results
        ]
    )

@app.get("/document/{filename}")
async def get_document(filename: str):
    """
    Download a document file
    
    - **filename**: The filename to download (from search results)
    """
    if not DOCUMENTS_ROOT:
        raise HTTPException(
            status_code=503,
            detail="Documents root not configured. Set DOCUMENTS_ROOT in api_server.py"
        )
    
    # Security: prevent path traversal
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = DOCUMENTS_ROOT / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {filename}")
    
    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "NGBSS Guide Retrieval API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "search": "/search?q=your+query",
            "document": "/document/{filename}",
            "health": "/health"
        }
    }

# =============================================================================
# Run directly
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
