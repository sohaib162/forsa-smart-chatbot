"""
Forsa Chatbot — Centralised Settings
=====================================
All configuration flows from environment variables through this single module.
Mirrors PrivateGPT's settings.py pattern using Pydantic Settings.

Zero hardcoded secrets. Every knob is tuneable via .env or docker-compose env.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class CorsSettings(BaseSettings):
    """CORS configuration."""

    model_config = {"env_prefix": "CORS_"}

    enabled: bool = True
    allow_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:5173",
    ]
    allow_methods: List[str] = ["GET", "POST", "OPTIONS"]
    allow_headers: List[str] = ["Content-Type", "Authorization"]
    allow_credentials: bool = True


class LLMSettings(BaseSettings):
    """Local LLM configuration."""

    model_config = {"env_prefix": "LLM_"}

    model_name: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        description="HuggingFace model ID or local path",
    )
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.85
    top_k: int = 20
    repetition_penalty: float = 1.15
    device: str = Field(
        default="auto",
        description="'auto', 'cuda', or 'cpu'",
    )
    gpu_memory_limit: str = "4.5GB"
    cpu_memory_limit: str = "8GB"
    trust_remote_code: bool = True


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = {"env_prefix": "EMBEDDING_"}

    model_name: str = "intfloat/multilingual-e5-small"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    model_config = {"env_prefix": "VECTOR_STORE_"}

    provider: str = Field(
        default="qdrant",
        description="'qdrant' (local or server) or 'simple' (in-memory)",
    )
    # Qdrant-specific
    qdrant_path: str = Field(
        default="./data/qdrant_storage",
        description="Local Qdrant storage directory (for embedded mode)",
    )
    qdrant_url: Optional[str] = Field(
        default=None,
        description="Qdrant server URL (if running as a service)",
    )
    collection_name: str = "forsa_documents"


class MinIOSettings(BaseSettings):
    """MinIO / S3 storage configuration."""

    model_config = {"env_prefix": "S3_"}

    endpoint: str = "minio:9000"
    external_endpoint: str = "localhost:9010"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    secure: bool = False
    bucket: str = "forsa-documents"


class IngestSettings(BaseSettings):
    """Document ingestion configuration."""

    model_config = {"env_prefix": "INGEST_"}

    chunk_size: int = 1024
    chunk_overlap: int = 128
    supported_extensions: List[str] = [".pdf", ".docx", ".odt", ".txt", ".json"]


class ServerSettings(BaseSettings):
    """Server-level configuration."""

    model_config = {"env_prefix": "SERVER_"}

    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "INFO"
    pipeline_timeout_seconds: int = 120
    env_name: str = Field(
        default="local",
        description="'local', 'staging', 'production'",
    )


class Settings(BaseSettings):
    """Root settings object — aggregates all sub-settings."""

    server: ServerSettings = ServerSettings()
    cors: CorsSettings = CorsSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    minio: MinIOSettings = MinIOSettings()
    ingest: IngestSettings = IngestSettings()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_settings: Settings | None = None


def settings() -> Settings:
    """Return the global Settings singleton (lazy-init)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
