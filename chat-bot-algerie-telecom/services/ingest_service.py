"""
IngestService — Document Lifecycle Management
================================================
Provides the business logic layer for document ingestion,
bridging the API routers with the IngestComponent.
"""
from __future__ import annotations

import logging
from typing import Any, BinaryIO, Dict, List, Optional

from components.ingest.ingest_component import IngestComponent, IngestResult

logger = logging.getLogger("forsa.services.ingest")


class IngestService:
    """
    Document ingestion service.

    Handles upload, status queries, and collection management.
    """

    def __init__(self, ingest: IngestComponent) -> None:
        self._ingest = ingest

    def ingest_file(
        self,
        filename: str,
        file_data: BinaryIO,
        category: str = "Uncategorized",
        language: str = "FR",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a single file.

        Returns a status dict with ingestion results.
        """
        result = self._ingest.ingest_file(
            filename=filename,
            file_data=file_data,
            category=category,
            language=language,
            extra_metadata=extra_metadata,
        )

        return {
            "object": "ingest.result",
            "filename": result.filename,
            "s3_key": result.s3_key,
            "category": result.category,
            "chunks_created": result.chunks_created,
            "status": result.status,
            "error": result.error,
            "metadata": result.metadata,
        }

    def ingest_text(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest raw text directly.

        Returns a status dict with the number of chunks created.
        """
        chunks = self._ingest.ingest_text(text, doc_id, metadata)
        return {
            "object": "ingest.result",
            "doc_id": doc_id,
            "chunks_created": chunks,
            "status": "success" if chunks > 0 else "empty",
        }

    def get_status(self) -> Dict[str, Any]:
        """Return current ingestion status and stats."""
        return {
            "object": "ingest.status",
            **self._ingest.list_ingested_documents(),
        }
