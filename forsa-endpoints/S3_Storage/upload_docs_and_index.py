#!/usr/bin/env python3
"""
Upload Documents to MinIO and Generate Index Files
===================================================
Uploads all documents from ./docs to MinIO S3 bucket and generates index files.

Features:
- Uploads .pdf, .docx, .DOCX, .odt files
- Skips temp files (~$) and hidden files (.)
- Generates collision-safe indexes:
  * s3_index.json: list of {s3_key, filename, category, ext, lang}
  * documents_s3_keys_multi.json: filename -> [s3_keys]
  * documents_s3_keys.json: legacy format (filename -> s3_key, only for unique names)

Usage:
  python S3_Storage/upload_docs_and_index.py
"""
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote
from minio import Minio

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DOCS_DIR = PROJECT_ROOT / "docs"

# MinIO connection (host port)
MINIO_ENDPOINT = "localhost:9010"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_SECURE = False
BUCKET_NAME = "forsa-documents"

# Supported file extensions
SUPPORTED_EXTENSIONS = ('.pdf', '.docx', '.DOCX', '.odt')


def should_skip_file(filename):
    """Check if file should be skipped (temp files, hidden files)"""
    # Skip Microsoft Word temp files
    if filename.startswith('~$'):
        return True
    # Skip hidden files
    if filename.startswith('.'):
        return True
    return False


def detect_language(filename):
    """Detect language from filename (AR/FR) using robust heuristics"""
    # Use the filename stem (without extension) for detection
    stem = Path(filename).stem
    stem_lower = stem.lower()

    # 1. Check for Arabic Unicode characters
    arabic_ranges = [
        '\u0600-\u06FF',  # Arabic
        '\u0750-\u077F',  # Arabic Supplement
        '\u08A0-\u08FF',  # Arabic Extended-A
    ]
    arabic_pattern = f'[{"".join(arabic_ranges)}]'
    if re.search(arabic_pattern, stem):
        return 'AR'

    # 2. Check for explicit Arabic words
    arabic_words = ['arabe', 'arabic', 'version arabe', 'en arabe']
    for word in arabic_words:
        if word in stem_lower:
            return 'AR'

    # 3. Check for explicit French words
    french_words = ['français', 'francais', 'french', 'version française', 'version francaise', 'en français', 'en francais']
    for word in french_words:
        if word in stem_lower:
            return 'FR'

    # 4. Check for standalone tokens " AR " or " FR " (delimited by space/underscore/dash/dot/brackets)
    # Use word boundaries or delimiters
    token_pattern = r'(?<!\w)(AR|FR)(?!\w)'
    match = re.search(token_pattern, stem, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 5. Default to FR
    return 'FR'


def get_category_from_path(relative_path):
    """Extract category from path (first directory level)"""
    parts = Path(relative_path).parts
    if len(parts) > 0:
        return parts[0]
    return "Uncategorized"


def scan_documents():
    """
    Scan ./docs directory and collect all documents to upload.
    Returns list of tuples: (local_path, s3_key, filename, category, ext, lang)
    """
    if not DOCS_DIR.exists():
        print(f"✗ ERROR: Documents directory not found: {DOCS_DIR}")
        return []

    documents = []
    print(f"\nScanning documents in: {DOCS_DIR}")

    for root, dirs, files in os.walk(DOCS_DIR):
        for file in files:
            # Check extension
            if not file.endswith(SUPPORTED_EXTENSIONS):
                continue

            # Skip temp and hidden files
            if should_skip_file(file):
                print(f"  Skipping temp/hidden file: {file}")
                continue

            local_path = Path(root) / file
            relative_path = local_path.relative_to(DOCS_DIR)

            # S3 key preserves directory structure
            s3_key = str(relative_path).replace('\\', '/')

            # Extract metadata
            category = get_category_from_path(relative_path)
            ext = local_path.suffix.lower()
            lang = detect_language(file)

            documents.append({
                'local_path': str(local_path),
                's3_key': s3_key,
                'filename': file,
                'category': category,
                'ext': ext,
                'lang': lang
            })

    print(f"  Found {len(documents)} documents to upload")
    return documents


def upload_to_minio(client, bucket, documents):
    """Upload documents to MinIO"""
    print(f"\nUploading to MinIO bucket '{bucket}'...")

    uploaded = 0
    failed = 0

    for doc in documents:
        try:
            client.fput_object(
                bucket_name=bucket,
                object_name=doc['s3_key'],
                file_path=doc['local_path']
            )
            uploaded += 1
            if uploaded <= 5 or uploaded % 20 == 0:
                print(f"  [{uploaded}/{len(documents)}] Uploaded: {doc['s3_key']}")
        except Exception as e:
            failed += 1
            print(f"  ✗ Failed to upload {doc['s3_key']}: {e}")

    print(f"\n  ✓ Upload complete: {uploaded} succeeded, {failed} failed")
    return uploaded, failed


def generate_indexes(documents):
    """
    Generate index files:
    1. s3_index.json: full index with metadata
    2. documents_s3_keys_multi.json: filename -> [s3_keys] (collision-safe)
    3. documents_s3_keys.json: legacy format (only unique filenames)
    """
    print(f"\nGenerating index files...")

    # 1. s3_index.json - full metadata
    s3_index = []
    for doc in documents:
        s3_index.append({
            's3_key': doc['s3_key'],
            'filename': doc['filename'],
            'category': doc['category'],
            'ext': doc['ext'],
            'lang': doc['lang']
        })

    index_file = SCRIPT_DIR / 's3_index.json'
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(s3_index, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Created: {index_file}")

    # 2. documents_s3_keys_multi.json - filename -> [s3_keys]
    multi_mapping = {}
    for doc in documents:
        filename = doc['filename']
        s3_key = doc['s3_key']
        if filename not in multi_mapping:
            multi_mapping[filename] = []
        multi_mapping[filename].append(s3_key)

    multi_file = SCRIPT_DIR / 'documents_s3_keys_multi.json'
    with open(multi_file, 'w', encoding='utf-8') as f:
        json.dump(multi_mapping, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Created: {multi_file}")

    # Check for duplicates
    duplicates = {k: v for k, v in multi_mapping.items() if len(v) > 1}
    if duplicates:
        print(f"\n  ⚠ Warning: Found {len(duplicates)} filenames with multiple S3 keys:")
        for filename, keys in list(duplicates.items())[:5]:
            print(f"    - {filename}: {len(keys)} locations")
        if len(duplicates) > 5:
            print(f"    ... and {len(duplicates) - 5} more")

    # 3. documents_s3_keys.json - legacy format (only unique)
    legacy_mapping = {}
    skipped_duplicates = 0
    for filename, keys in multi_mapping.items():
        if len(keys) == 1:
            legacy_mapping[filename] = keys[0]
        else:
            skipped_duplicates += 1

    legacy_file = SCRIPT_DIR / 'documents_s3_keys.json'
    with open(legacy_file, 'w', encoding='utf-8') as f:
        json.dump(legacy_mapping, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Created: {legacy_file} (legacy format)")
    if skipped_duplicates > 0:
        print(f"    Note: {skipped_duplicates} duplicate filenames excluded from legacy format")

    return s3_index, multi_mapping, legacy_mapping


def print_examples(s3_index):
    """Print example index entries"""
    print(f"\n{'='*80}")
    print("EXAMPLE INDEX ENTRIES")
    print(f"{'='*80}")

    # Find an Arabic example
    arabic_examples = [doc for doc in s3_index if doc['lang'] == 'AR']
    if arabic_examples:
        doc = arabic_examples[0]
        print(f"\nArabic document example:")
        print(f"  {json.dumps(doc, ensure_ascii=False, indent=2)}")

    # Find a French example
    french_examples = [doc for doc in s3_index if doc['lang'] == 'FR']
    if french_examples:
        doc = french_examples[0]
        print(f"\nFrench document example:")
        print(f"  {json.dumps(doc, ensure_ascii=False, indent=2)}")

    print(f"\n{'='*80}")


def print_sanity_check(s3_index):
    """Print sanity check with counts and samples"""
    print(f"\n{'='*80}")
    print("SANITY CHECK - LANGUAGE DETECTION")
    print(f"{'='*80}")

    ar_count = sum(1 for doc in s3_index if doc['lang'] == 'AR')
    fr_count = sum(1 for doc in s3_index if doc['lang'] == 'FR')

    print(f"Total documents: {len(s3_index)}")
    print(f"Arabic (AR): {ar_count}")
    print(f"French (FR): {fr_count}")

    print(f"\nSample AR documents:")
    ar_samples = [doc for doc in s3_index if doc['lang'] == 'AR'][:3]
    for doc in ar_samples:
        print(f"  - {doc['filename']} (category: {doc['category']})")

    print(f"\nSample FR documents:")
    fr_samples = [doc for doc in s3_index if doc['lang'] == 'FR'][:3]
    for doc in fr_samples:
        print(f"  - {doc['filename']} (category: {doc['category']})")

    print(f"\n{'='*80}")


def print_instructions(uploaded_count):
    """Print step-by-step instructions for testing"""
    print(f"\n{'='*80}")
    print("UPLOAD COMPLETE - NEXT STEPS")
    print(f"{'='*80}\n")

    print(f"✓ Uploaded {uploaded_count} documents to MinIO")
    print(f"✓ Generated index files in S3_Storage/\n")

    print("To verify and test:\n")

    print("1. Verify upload:")
    print("   python S3_Storage/S3.py\n")

    print("2. Open MinIO Console:")
    print("   http://localhost:9011")
    print("   Username: minioadmin")
    print("   Password: minioadmin\n")

    print("3. Test document retrieval (example):")
    # Get first document from index
    index_file = SCRIPT_DIR / 's3_index.json'
    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
            if index_data:
                first_doc = index_data[0]
                s3_key = first_doc['s3_key']
                encoded_key = quote(s3_key, safe='')
                print(f"   Test URL: http://localhost:8000/document/{encoded_key}")
                print(f"   Document: {first_doc['filename']}")
                print(f"   Category: {first_doc['category']}")
                print(f"   Language: {first_doc['lang']}\n")

    print("4. Make sure your FastAPI backend implements the /document/{s3_key:path} route")
    print("   (See the FastAPI snippet that will be generated)\n")

    print(f"{'='*80}\n")


def main():
    """Main function"""
    print(f"\n{'='*80}")
    print("MinIO Document Upload and Index Generation")
    print(f"{'='*80}")

    # Scan documents
    documents = scan_documents()
    if not documents:
        print("\n✗ No documents found to upload.")
        return 1

    # Connect to MinIO
    print(f"\nConnecting to MinIO at {MINIO_ENDPOINT}...")
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )

        # Create bucket if not exists
        if not client.bucket_exists(BUCKET_NAME):
            print(f"  Creating bucket '{BUCKET_NAME}'...")
            client.make_bucket(BUCKET_NAME)
        else:
            print(f"  Bucket '{BUCKET_NAME}' already exists")

    except Exception as e:
        print(f"\n✗ ERROR: Failed to connect to MinIO: {e}")
        print("\nMake sure MinIO is running:")
        print("  docker compose up -d")
        return 1

    # Upload documents
    uploaded, failed = upload_to_minio(client, BUCKET_NAME, documents)

    if uploaded == 0:
        print("\n✗ No documents were uploaded successfully.")
        return 1

    # Generate indexes
    s3_index, multi_mapping, legacy_mapping = generate_indexes(documents)

    # Print examples
    print_examples(s3_index)

    # Sanity check
    print_sanity_check(s3_index)

    # Print instructions
    print_instructions(uploaded)

    return 0


if __name__ == "__main__":
    sys.exit(main())
