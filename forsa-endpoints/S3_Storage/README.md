# MinIO S3 Document Storage - Setup Guide

This guide explains how to set up and use the MinIO S3 document storage system with the NGBSS Retrieval API.

## Overview

The system provides:
- **MinIO S3 storage** for documents (PDFs, DOCX, ODT files)
- **Collision-safe indexing** with metadata (category, language, extension)
- **API proxy route** for document retrieval
- **Automatic document upload** from `./docs` directory

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Browser   │────▶│  FastAPI     │────▶│   MinIO     │
│             │     │  (Port 8000) │     │ (Port 9010) │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Index Files  │
                    │ (JSON)       │
                    └──────────────┘
```

## Prerequisites

1. **Docker** and **Docker Compose** installed
2. **Python 3.11+** with pip
3. Required Python packages:
   ```bash
   pip install minio python-dotenv
   ```

## Quick Start

### Step 1: Start MinIO with Docker Compose

```bash
docker compose up -d --build
```

This starts:
- **MinIO S3 API**: `http://localhost:9010` (container port 9000)
- **MinIO Console**: `http://localhost:9011` (container port 9001)
- **FastAPI**: `http://localhost:8000`

Credentials: `minioadmin` / `minioadmin`

### Step 2: Upload Documents and Generate Index

```bash
python S3_Storage/upload_docs_and_index.py
```

This script:
- ✅ Uploads all `.pdf`, `.docx`, `.DOCX`, `.odt` files from `./docs`
- ✅ Skips Microsoft Word temp files (`~$...`) and hidden files
- ✅ Preserves directory structure as S3 keys
- ✅ Detects document language (AR/FR)
- ✅ Generates 3 index files (see below)

### Step 3: Verify Upload

```bash
python S3_Storage/S3.py
```

Expected output:
```
================================================================================
FILES IN MINIO BUCKET 'forsa-documents'
================================================================================
Total objects: 130

First 20 objects:
   1. Offres/Argumentaire Offre Gamers V5 2025 AR.docx
   2. Offres/Argumentaire MOOHTARIF FR.docx
   ...
```

### Step 4: Open MinIO Console

Visit: `http://localhost:9011`

Login: `minioadmin` / `minioadmin`

Browse the `forsa-documents` bucket to verify files are uploaded.

### Step 5: Test Document Retrieval

Test URL (replace with actual S3 key):
```
http://localhost:8000/document/Offres/Argumentaire%20Offre%20Gamers%20V5%202025%20AR.docx
```

The `/document/{s3_key:path}` route:
- ✅ Streams documents efficiently
- ✅ Sets `Content-Disposition: inline` for PDFs (view in browser)
- ✅ Sets `Content-Disposition: attachment` for DOCX/ODT (download)
- ✅ Handles UTF-8 filenames (Arabic/French)

## Generated Index Files

After running `upload_docs_and_index.py`, you'll find these files in `S3_Storage/`:

### 1. `s3_index.json`
Full index with metadata for all documents.

**Example entry:**
```json
{
  "s3_key": "Offres/Argumentaire Offre Gamers V5 2025 AR.docx",
  "filename": "Argumentaire Offre Gamers V5 2025 AR.docx",
  "category": "Offres",
  "ext": ".docx",
  "lang": "AR"
}
```

**Fields:**
- `s3_key`: Full S3 object key (with path)
- `filename`: Just the filename
- `category`: First-level directory (Offres, Guides, Produits, etc.)
- `ext`: File extension (.pdf, .docx, .odt)
- `lang`: Detected language (AR or FR)

### 2. `documents_s3_keys_multi.json`
Collision-safe mapping: `filename` → `[s3_keys]`

**Example:**
```json
{
  "Argumentaire Offre Gamers V5 2025 AR.docx": [
    "Offres/Argumentaire Offre Gamers V5 2025 AR.docx"
  ],
  "Guide Installation.pdf": [
    "Guides/Guide Installation.pdf",
    "Produits/Guide Installation.pdf"
  ]
}
```

Use this when you need to handle duplicate filenames.

### 3. `documents_s3_keys.json` (Legacy)
Simple mapping: `filename` → `s3_key` (only for unique filenames)

**Example:**
```json
{
  "Argumentaire Offre Gamers V5 2025 AR.docx": "Offres/Argumentaire Offre Gamers V5 2025 AR.docx"
}
```

⚠️ **Note**: Duplicate filenames are excluded from this file. Use `documents_s3_keys_multi.json` instead.

## Scripts Reference

### `upload_docs_and_index.py`
Uploads documents and generates indexes.

```bash
python S3_Storage/upload_docs_and_index.py
```

**Features:**
- Scans `./docs` recursively
- Uploads PDF, DOCX, ODT files
- Skips temp files (`~$`) and hidden files
- Detects language from filename
- Generates all index files

### `S3.py`
Lists objects in MinIO bucket.

```bash
python S3_Storage/S3.py
```

**Output:**
- Total object count
- First 20 objects

### `s3_url_generator.py`
Generates URLs for documents.

```bash
python S3_Storage/s3_url_generator.py
```

**Configuration (environment variables):**
- `S3_ENDPOINT` (default: `minio:9000`)
- `S3_EXTERNAL_ENDPOINT` (default: `localhost:9010`)
- `S3_ACCESS_KEY` (default: `minioadmin`)
- `S3_SECRET_KEY` (default: `minioadmin`)
- `S3_BUCKET_NAME` (default: `forsa-documents`)
- `API_BASE_URL` (default: `http://localhost:8000`)

**Behavior:**
- If `S3_ENDPOINT` ≠ `S3_EXTERNAL_ENDPOINT`: returns API proxy URL
- Otherwise: returns presigned S3 URL

### `add_s3_keys_to_json.py`
Updates JSON files with S3 keys.

```bash
python S3_Storage/add_s3_keys_to_json.py data/Guide_NGBSS.json
```

**Features:**
- Accepts input JSON path as CLI argument
- Matches using existing `s3_key` field (most reliable)
- Falls back to filename matching (with collision detection)
- Safe: only updates if file exists and is writable

## FastAPI Integration

See [fastapi_document_route.py](fastapi_document_route.py) for the complete implementation.

### Quick Integration

Add this to your `main.py`:

```python
from fastapi import FastAPI
from S3_Storage.fastapi_document_route import router as document_router

app = FastAPI()

# Include document router
app.include_router(document_router)
```

### Route Details

**Endpoint:** `GET /document/{s3_key:path}`

**Example:**
```
GET /document/Offres/Argumentaire%20Offre%20Gamers%20V5%202025%20AR.docx
```

**Response Headers:**
- `Content-Type`: Detected from file extension
- `Content-Disposition`:
  - `inline; filename="..."` for PDFs
  - `attachment; filename="..."` for DOCX/ODT
- `Cache-Control`: `public, max-age=3600`

**Error Codes:**
- `404`: Document not found
- `500`: Server error

## Ports Reference

| Service | Container Port | Host Port | URL |
|---------|---------------|-----------|-----|
| MinIO S3 API | 9000 | 9010 | `http://localhost:9010` |
| MinIO Console | 9001 | 9011 | `http://localhost:9011` |
| FastAPI | 8000 | 8000 | `http://localhost:8000` |

⚠️ **Important**: Python scripts running on the **host** must use port **9010** (not 9000).

## Troubleshooting

### Issue: "Connection refused" when running scripts

**Solution:** Make sure MinIO is running:
```bash
docker compose up -d
```

### Issue: "Bucket does not exist"

**Solution:** Run the upload script to create the bucket:
```bash
python S3_Storage/upload_docs_and_index.py
```

### Issue: Duplicate filenames

**Problem:** Multiple files with the same name in different directories.

**Solution:** Use `documents_s3_keys_multi.json` instead of the legacy format. It stores all S3 keys for each filename.

**Example:**
```python
import json

with open('S3_Storage/documents_s3_keys_multi.json', 'r') as f:
    mapping = json.load(f)

# Get all locations for a filename
s3_keys = mapping.get('Guide Installation.pdf', [])
print(f"Found {len(s3_keys)} locations:")
for key in s3_keys:
    print(f"  - {key}")
```

### Issue: Document URL returns 404

**Checklist:**
1. ✅ MinIO is running: `docker compose ps`
2. ✅ Document was uploaded: `python S3_Storage/S3.py`
3. ✅ S3 key is URL-encoded correctly
4. ✅ FastAPI route is implemented (see `fastapi_document_route.py`)

### Issue: Special characters in filenames

**Solution:** The system handles UTF-8 filenames automatically. Make sure to URL-encode the S3 key:

```python
from urllib.parse import quote

s3_key = "Offres/Argumentaire Offre Gamers V5 2025 AR.docx"
url = f"http://localhost:8000/document/{quote(s3_key, safe='')}"
```

## Docker Compose Environment Variables

The `docker-compose.yml` sets these for the API container:

```yaml
environment:
  - S3_ENDPOINT=minio:9000              # Internal Docker network
  - S3_EXTERNAL_ENDPOINT=localhost:9010 # External access
  - S3_ACCESS_KEY=minioadmin
  - S3_SECRET_KEY=minioadmin
  - S3_SECURE=false
  - S3_BUCKET=forsa-documents
```

Since `S3_ENDPOINT` ≠ `S3_EXTERNAL_ENDPOINT`, the `s3_url_generator.py` returns API proxy URLs:

```
http://localhost:8000/document/Offres/...
```

## Testing Checklist

- [ ] Docker Compose starts: `docker compose up -d --build`
- [ ] MinIO Console accessible: `http://localhost:9011`
- [ ] Upload script succeeds: `python S3_Storage/upload_docs_and_index.py`
- [ ] List script shows files: `python S3_Storage/S3.py`
- [ ] Index files generated: `ls S3_Storage/*.json`
- [ ] FastAPI route implemented (see `fastapi_document_route.py`)
- [ ] Document URL works: `http://localhost:8000/document/...`

## Files Modified/Created

### Modified:
- `S3.py` - Fixed port to 9010, added count + first 20 keys display
- `s3_url_generator.py` - Made standalone with environment variables
- `add_s3_keys_to_json.py` - Added CLI args, collision-safe matching

### Created:
- `upload_docs_and_index.py` - Main upload script with index generation
- `fastapi_document_route.py` - Ready-to-use FastAPI route
- `README.md` - This file

### Generated (after running upload script):
- `s3_index.json` - Full metadata index
- `documents_s3_keys_multi.json` - Collision-safe filename mapping
- `documents_s3_keys.json` - Legacy format (unique filenames only)

## Support

For issues, check:
1. Docker logs: `docker compose logs -f`
2. MinIO health: `docker compose ps`
3. Python script errors: Run with verbose output

---

**Ready to go!** Follow the Quick Start section to get up and running in minutes.
