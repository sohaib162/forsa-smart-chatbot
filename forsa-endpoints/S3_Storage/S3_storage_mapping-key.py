import os
import json
from minio import Minio

# -------------------------------
# 1. Connect to local MinIO (S3)
# -------------------------------
client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

BUCKET_NAME = "forsa-documents"

# Create bucket if not exists
if not client.bucket_exists(BUCKET_NAME):
    client.make_bucket(BUCKET_NAME)

# -------------------------------------------
# 2. Function: upload file and return s3 key
# -------------------------------------------
def upload_to_minio(local_path, bucket, s3_key):
    """
    Upload a file to MinIO using the given s3_key (with relative path)
    """
    client.fput_object(
        bucket_name=bucket,
        object_name=s3_key,
        file_path=local_path
    )
    return s3_key

# ------------------------------------------------
# 3. Walk through folders and collect upload info
# ------------------------------------------------
def scan_and_upload(root_folder):
    results = {}  # {document_name: s3_key}
    root_folder = os.path.abspath(root_folder)

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith((".pdf", ".docx")):
                local_path = os.path.join(root, file)
                # Preserve relative path as s3_key
                relative_path = os.path.relpath(local_path, root_folder)
                s3_key = relative_path.replace("\\", "/")  # Use forward slashes for S3

                upload_to_minio(local_path, BUCKET_NAME, s3_key)
                results[file] = s3_key  # map filename to exact S3 key

    return results

# -------------------------
# 4. Example usage
# -------------------------
root_folder = r"C:\Users\21355\Desktop\forsa\actelchatbot"  # <-- your folder
output_json = "documents_s3_keys.json"

documents_dict = scan_and_upload(root_folder)

# Save results to JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(documents_dict, f, ensure_ascii=False, indent=2)

print(f"Done! Mapping saved to {output_json}")
