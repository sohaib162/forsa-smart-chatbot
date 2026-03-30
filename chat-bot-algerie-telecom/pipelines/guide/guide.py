"""
pipelines/guide/guide.py
=========================
Guide Pipeline — upgraded to Graph-RAG + Citation-Faithful Generation.

Uses the same AdvancedPipeline as conventions for KG enrichment,
argument mining, and hallucination validation.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from local_llm_client import get_llm_client
from pipelines.advanced_pipeline import AdvancedPipeline, get_advanced_pipeline
from pipelines.guide.guide_code.scripts.retrieval_function import retrieve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).parent
_GUIDE_DATA = _HERE / "guide_code" / "data" / "Guide_NGBSS.json"
_KG_DIR     = _HERE / "guide_code" / "kg"

# ---------------------------------------------------------------------------
# Singleton pipeline
# ---------------------------------------------------------------------------
_adv_pipeline: AdvancedPipeline | None = None


def _get_advanced_pipeline() -> AdvancedPipeline:
    global _adv_pipeline
    if _adv_pipeline is not None:
        return _adv_pipeline

    pipeline = get_advanced_pipeline(
        domain="guides",
        graph_dir=str(_KG_DIR),
        faithfulness_threshold=0.65,
        annotate_response=True,
    )

    # Build KG from guides data
    if _GUIDE_DATA.exists():
        with open(_GUIDE_DATA, "r", encoding="utf-8") as f:
            raw = json.load(f)
        documents = raw.get("guides", raw) if isinstance(raw, dict) else raw
        pipeline.build_graph(documents, force_rebuild=False)
    else:
        logger.warning(
            "Guide_NGBSS.json not found at %s — KG will be empty", _GUIDE_DATA
        )
        pipeline.build_graph([], force_rebuild=False)

    _adv_pipeline = pipeline
    return _adv_pipeline


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_guide_pipeline(query: str) -> Dict[str, Any]:
    """Execute the Graph-RAG pipeline for the guides domain."""

    # Step 1: BM25 + dense retrieval
    context = retrieve(query=query, top_k=1)
    retrieved_docs = context.get("retrieved_documents", []) or []

    if not retrieved_docs:
        return {
            "answer": "Aucun guide correspondant trouvé dans les documents fournis.",
            "sources": [],
            "faithfulness": 1.0,
            "kg_nodes_used": 0,
            "argument_types": [],
        }

    # Step 2: Normalize to passage format for AdvancedPipeline
    doc = retrieved_docs[0]
    info = context.get("retrieval_info", {}) or {}
    filename = info.get("filename", "")
    s3_key = info.get("s3_key", "")
    ext = Path(filename).suffix if isinstance(filename, str) else ""

    retrieved_passages = [
        {
            "doc_id":   filename,
            "doc_name": filename,
            "text":     json.dumps(doc, ensure_ascii=False),
            "score":    info.get("score", 1.0),
        }
    ]

    # Step 3: Source metadata for frontend
    sources = []
    if s3_key and filename:
        sources.append({
            "s3_key":   s3_key,
            "filename": filename,
            "category": "Guides",
            "ext":      ext,
            "lang":     "FR",
        })

    # Step 4: Run AdvancedPipeline
    try:
        adv = _get_advanced_pipeline()
        result = adv.run(
            query=query,
            retrieved_passages=retrieved_passages,
            llm_client=get_llm_client(),
            sources=sources,
            max_new_tokens=512,
        )
        return {
            "answer":         result.answer,
            "sources":        result.sources,
            "faithfulness":   result.faithfulness_score,
            "kg_nodes_used":  result.kg_nodes_used,
            "argument_types": result.argument_types,
        }
    except Exception as exc:
        # Graceful degradation
        logger.exception("AdvancedPipeline failed for guides, falling back: %s", exc)
        from local_llm_client import call_local_llm
        context_str = f"Query: {query}\n\nRetrieved Guide:\n{json.dumps(doc, ensure_ascii=False, indent=2)}"
        response_text = call_local_llm(
            "Tu es un assistant spécialisé dans les guides d'Algérie Télécom. "
            "N'invente AUCUNE donnée.",
            context_str,
        )
        return {"answer": response_text, "sources": sources}
