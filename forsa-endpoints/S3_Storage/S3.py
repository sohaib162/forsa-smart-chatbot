"""
List all objects in the MinIO S3 bucket.
"""
from minio import Minio

# Connect to MinIO on host - use port 9010 (mapped from container 9000)
client = Minio(
    "localhost:9010",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket = "forsa-documents"

try:
    objects = list(client.list_objects(bucket, recursive=True))

    print(f"\n{'='*80}")
    print(f"FILES IN MINIO BUCKET '{bucket}'")
    print(f"{'='*80}")
    print(f"Total objects: {len(objects)}\n")

    if objects:
        print("First 20 objects:")
        for i, obj in enumerate(objects[:20], 1):
            print(f"  {i:2d}. {obj.object_name}")

        if len(objects) > 20:
            print(f"\n  ... and {len(objects) - 20} more objects")
    else:
        print("No objects found in bucket.")

    print(f"\n{'='*80}\n")

except Exception as e:
    print(f"ERROR: Failed to list objects: {e}")
    print("\nMake sure:")
    print("  1. MinIO is running: docker compose up -d")
    print("  2. Bucket exists (run upload_docs_and_index.py first)")
    print("  3. Port 9010 is accessible")
