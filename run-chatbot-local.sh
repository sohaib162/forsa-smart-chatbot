#!/bin/bash
# ==================================================================
# Forsa Chatbot API v3.0 — Local GPU Launcher
# ==================================================================
# PrivateGPT-style architecture with OpenAI-compatible API.
# Requires: conda environment with all dependencies from requirements.txt
# ==================================================================

set -euo pipefail

echo "═══════════════════════════════════════════════════"
echo "  Forsa Chatbot API v3.0 — PrivateGPT Architecture"
echo "═══════════════════════════════════════════════════"

# Configuration — override via environment variables
CONDA_ENV="${CONDA_ENV:-fyp}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHATBOT_DIR="${SCRIPT_DIR}/chat-bot-algerie-telecom"
S3_INDEX_PATH="${CHATBOT_DIR}/../forsa-endpoints/S3_Storage/s3_index.json"
PORT="${PORT:-8001}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Validate chatbot directory exists
if [ ! -d "$CHATBOT_DIR" ]; then
    echo "Error: Chatbot directory not found: $CHATBOT_DIR"
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

# Check if conda environment exists
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "Error: Conda environment '${CONDA_ENV}' not found!"
    echo "Available environments:"
    conda env list
    exit 1
fi

# Navigate to chatbot directory
cd "$CHATBOT_DIR"

echo ""
echo "Configuration:"
echo "  Environment:  $CONDA_ENV"
echo "  Directory:    $CHATBOT_DIR"
echo "  Port:         $PORT"
echo "  Log Level:    $LOG_LEVEL"
echo "  Architecture: PrivateGPT-style DI"
echo ""

# Set environment variables (new architecture + legacy compat)
export LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
export LLM_MODEL_NAME="${LOCAL_MODEL_NAME}"
export S3_INDEX_PATH="$S3_INDEX_PATH"
export ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://localhost:5173,http://localhost:8080}"
export LOG_LEVEL="$LOG_LEVEL"
export SERVER_LOG_LEVEL="$LOG_LEVEL"
export SERVER_PORT="$PORT"
export SERVER_ENV_NAME="local"
export PIPELINE_TIMEOUT_SECONDS="${PIPELINE_TIMEOUT_SECONDS:-120}"

# Vector store (embedded Qdrant — no external service needed)
export VECTOR_STORE_PROVIDER="qdrant"
export VECTOR_STORE_QDRANT_PATH="./data/qdrant_storage"

# MinIO (optional — will work without it, just no S3 uploads)
export S3_ENDPOINT="${S3_ENDPOINT:-minio:9000}"
export S3_ACCESS_KEY="${S3_ACCESS_KEY:-minioadmin}"
export S3_SECRET_KEY="${S3_SECRET_KEY:-minioadmin}"

# Activate conda environment and run
echo "Activating conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Check CUDA
echo "Checking CUDA availability..."
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPU:  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "═══════════════════════════════════════════════════"
echo "  API Endpoints:"
echo "    Health:      http://localhost:$PORT/v1/health"
echo "    Chat:        http://localhost:$PORT/v1/chat/completions"
echo "    Embeddings:  http://localhost:$PORT/v1/embeddings"
echo "    Ingest:      http://localhost:$PORT/v1/ingest"
echo "    Models:      http://localhost:$PORT/v1/models"
echo "    Legacy:      http://localhost:$PORT/process-question"
echo "    Docs:        http://localhost:$PORT/docs"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server with graceful shutdown
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level "$LOG_LEVEL" \
    --timeout-keep-alive 65
