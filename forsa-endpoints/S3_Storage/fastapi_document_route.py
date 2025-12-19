"""
FastAPI Document Proxy Route
=============================
Ready-to-paste FastAPI route implementation for streaming documents from MinIO.

This route proxies document requests from MinIO S3, handling:
- Path parameter with slashes (s3_key can contain /)
- Streaming for efficient memory usage
- Proper Content-Disposition headers (inline for PDFs, attachment for docs)
- UTF-8 filename encoding

Add this to your FastAPI application:
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from minio import Minio
from urllib.parse import unquote, quote
import os

# ============================================================================
# Configuration (use environment variables)
# ============================================================================

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "minio:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin")
S3_SECURE = os.environ.get("S3_SECURE", "false").lower() in ("true", "1", "yes")
S3_BUCKET = os.environ.get("S3_BUCKET", "forsa-documents")

# Initialize MinIO client
minio_client = Minio(
    S3_ENDPOINT,
    access_key=S3_ACCESS_KEY,
    secret_key=S3_SECRET_KEY,
    secure=S3_SECURE
)

# Create router
router = APIRouter()


# ============================================================================
# Document Proxy Route
# ============================================================================

@router.get("/document/{s3_key:path}")
async def get_document(s3_key: str):
    """
    Stream a document from MinIO S3 storage.

    Args:
        s3_key: The S3 object key (path in bucket). Can contain slashes.
                Will be URL-decoded automatically.

    Returns:
        StreamingResponse with the document content

    Raises:
        HTTPException: 404 if document not found, 500 on server error
    """
    try:
        # URL decode the s3_key (handles special characters and UTF-8)
        s3_key = unquote(s3_key)

        # Get object from MinIO
        try:
            response = minio_client.get_object(S3_BUCKET, s3_key)
        except Exception as e:
            error_msg = str(e)
            if "NoSuchKey" in error_msg or "Not Found" in error_msg:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found: {s3_key}"
                )
            raise

        # Extract filename from s3_key
        filename = s3_key.split('/')[-1]

        # Determine Content-Type
        content_type = "application/octet-stream"
        if filename.lower().endswith('.pdf'):
            content_type = "application/pdf"
        elif filename.lower().endswith(('.doc', '.docx')):
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif filename.lower().endswith('.odt'):
            content_type = "application/vnd.oasis.opendocument.text"

        # Set Content-Disposition with RFC 5987 encoding for non-ASCII filenames
        # PDFs: inline (view in browser)
        # Word/ODT: attachment (download)
        disposition_type = 'inline' if filename.lower().endswith('.pdf') else 'attachment'

        # RFC 5987: filename*=UTF-8''encoded_filename
        encoded_filename = quote(filename, safe='')
        disposition = f"{disposition_type}; filename=\"{filename}\"; filename*=UTF-8''{encoded_filename}"

        # Stream response
        return StreamingResponse(
            response.stream(32*1024),  # 32KB chunks
            media_type=content_type,
            headers={
                "Content-Disposition": disposition,
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Failed to retrieve document {s3_key}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document: {str(e)}"
        )


# ============================================================================
# How to integrate into your FastAPI app
# ============================================================================

"""
In your main FastAPI application file (e.g., main.py):

from fastapi import FastAPI
from S3_Storage.fastapi_document_route import router as document_router

app = FastAPI()

# Include the document router
app.include_router(document_router)

# Your other routes...
@app.get("/")
def read_root():
    return {"message": "NGBSS Retrieval API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
"""


# ============================================================================
# Alternative: Standalone route (copy-paste into your existing app)
# ============================================================================

"""
If you prefer to add the route directly to your existing app without a router:

@app.get("/document/{s3_key:path}")
async def get_document(s3_key: str):
    # ... copy the implementation from above ...
    pass
"""
