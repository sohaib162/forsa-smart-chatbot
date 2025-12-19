"""
S3 URL Generator for Document Retrieval
========================================
Generates URLs for documents stored in MinIO/S3.

For Docker environments where internal and external endpoints differ,
the generator returns API proxy URLs instead of presigned S3 URLs.
This avoids signature validation issues when hosts don't match.

Configuration via environment variables:
  - S3_ENDPOINT (default: "minio:9000")
  - S3_EXTERNAL_ENDPOINT (default: "localhost:9010")
  - S3_ACCESS_KEY (default: "minioadmin")
  - S3_SECRET_KEY (default: "minioadmin")
  - S3_BUCKET_NAME (default: "forsa-documents")
  - API_BASE_URL (default: "http://localhost:8000")
"""
import os
from datetime import timedelta
from typing import Optional
from urllib.parse import quote

# Configuration from environment variables
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "minio:9000")
S3_EXTERNAL_ENDPOINT = os.environ.get("S3_EXTERNAL_ENDPOINT", "localhost:9010")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "forsa-documents")
S3_SECURE = os.environ.get("S3_SECURE", "false").lower() in ("true", "1", "yes")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Lazy load minio client
_minio_client = None


def get_minio_client():
    """Get or create MinIO client (lazy loading)."""
    global _minio_client
    from minio import Minio

    if _minio_client is None:
        _minio_client = Minio(
            S3_ENDPOINT,
            access_key=S3_ACCESS_KEY,
            secret_key=S3_SECRET_KEY,
            secure=S3_SECURE
        )
    return _minio_client


def generate_presigned_url(
    s3_key: str,
    bucket: str = None,
    expires: timedelta = timedelta(hours=1),
    inline: bool = True
) -> Optional[str]:
    """
    Generate a URL for an S3 object.

    If internal and external endpoints differ (Docker environment),
    returns a proxy URL through the API. Otherwise returns a presigned URL.

    Args:
        s3_key: The S3 object key (path in bucket)
        bucket: Bucket name (defaults to S3_BUCKET from settings)
        expires: URL expiration time (default: 1 hour) - only used for presigned URLs
        inline: If True, set content-disposition to inline (view in browser)

    Returns:
        URL string, or None if generation fails
    """
    if not s3_key:
        return None

    try:
        # If endpoints differ, use proxy URL to avoid signature issues
        if S3_EXTERNAL_ENDPOINT and S3_EXTERNAL_ENDPOINT != S3_ENDPOINT:
            # Return API proxy URL
            encoded_key = quote(s3_key, safe='')
            return f"{API_BASE_URL}/document/{encoded_key}"

        # Otherwise use presigned URL
        client = get_minio_client()
        bucket = bucket or S3_BUCKET

        # Prepare response headers
        response_headers = {}
        if inline:
            filename = s3_key.split('/')[-1]
            response_headers["response-content-disposition"] = f"inline; filename=\"{filename}\""

        url = client.presigned_get_object(
            bucket,
            s3_key,
            expires=expires,
            response_headers=response_headers if response_headers else None
        )

        return url

    except Exception as e:
        print(f"⚠ Failed to generate presigned URL for {s3_key}: {e}")
        return None


def generate_urls_batch(
    s3_keys: list,
    bucket: str = None,
    expires: timedelta = timedelta(hours=1),
    inline: bool = True
) -> dict:
    """
    Generate presigned URLs for multiple S3 objects.

    Args:
        s3_keys: List of S3 object keys
        bucket: Bucket name (defaults to S3_BUCKET from settings)
        expires: URL expiration time
        inline: If True, set content-disposition to inline

    Returns:
        Dict mapping s3_key -> presigned URL (or None if failed)
    """
    results = {}
    for key in s3_keys:
        results[key] = generate_presigned_url(key, bucket, expires, inline)
    return results


# For testing
if __name__ == "__main__":
    # Test with a sample key
    test_key = "Guides/Guide NGBSS Recharge par bon de commande - complément.pdf"
    print(f"Testing URL generation for: {test_key}")
    print(f"S3_ENDPOINT: {S3_ENDPOINT}")
    print(f"S3_EXTERNAL_ENDPOINT: {S3_EXTERNAL_ENDPOINT}")
    print(f"S3_BUCKET: {S3_BUCKET}")
    print(f"API_BASE_URL: {API_BASE_URL}")

    url = generate_presigned_url(test_key)
    if url:
        print(f"✓ Generated URL:\n{url}")
    else:
        print("✗ Failed to generate URL")
