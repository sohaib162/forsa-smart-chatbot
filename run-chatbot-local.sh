#!/bin/bash
# Script to run the chatbot API locally with GPU support using conda environment

set -e

echo "=========================================="
echo "Starting Forsa Chatbot API (Local + GPU)"
echo "=========================================="

# Configuration
CONDA_ENV="fyp"
CHATBOT_DIR="/home/sohaib/Documents/forsa-smart-chatbot/chat-bot-algerie-telecom"
S3_INDEX_PATH="${CHATBOT_DIR}/../forsa-endpoints/S3_Storage/s3_index.json"
PORT=8001

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
echo "Environment: $CONDA_ENV"
echo "Directory: $CHATBOT_DIR"
echo "Port: $PORT"
echo "S3 Index: $S3_INDEX_PATH"
echo ""

# Set environment variables
export LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
export S3_INDEX_PATH="$S3_INDEX_PATH"

# Activate conda environment and run
echo "Activating conda environment: $CONDA_ENV"
echo "Starting uvicorn server..."
echo ""
echo "API will be available at: http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

# Run with conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Check if CUDA is available
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Start the server
uvicorn main:app --host 0.0.0.0 --port "$PORT" --log-level info
