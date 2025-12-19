FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1


WORKDIR /app

# Install system dependencies (often needed for numpy/torch on slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements first (to cache the heavy install step)
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image smaller, but it makes the build slower
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
