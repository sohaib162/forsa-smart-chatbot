"""
pipelines/depot/depot.py
=========================
Depot (Products) Pipeline — upgraded to Graph-RAG + Citation-Faithful Generation.

Uses AdvancedPipeline for KG enrichment, argument mining, and
hallucination validation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from local_llm_client import get_llm_client
from pipelines.advanced_pipeline import AdvancedPipeline, get_advanced_pipeline
from .depot_code.src.muuh import muuh

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).parent
_KG_DIR     = _HERE / "depot_code" / "kg"
_PRODUCTS   = _HERE / "depot_code" / "data" / "products.json"

# ---------------------------------------------------------------------------
# Singleton pipeline
# ---------------------------------------------------------------------------
_adv_pipeline: AdvancedPipeline | None = None


def _get_advanced_pipeline() -> AdvancedPipeline:
    global _adv_pipeline
    if _adv_pipeline is not None:
        return _adv_pipeline

    pipeline = get_advanced_pipeline(
        domain="produits",
        graph_dir=str(_KG_DIR),
        faithfulness_threshold=0.55,
        annotate_response=True,
    )

    # Build KG from products data
    documents = []
    if _PRODUCTS.exists():
        try:
            with open(_PRODUCTS, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                documents = raw
            elif isinstance(raw, dict):
                documents = raw.get("products", raw.get("produits", [raw]))
        except Exception as e:
            logger.warning("Failed to load products.json: %s", e)

    pipeline.build_graph(documents, force_rebuild=False)
    _adv_pipeline = pipeline
    return _adv_pipeline


def _build_json(query: str) -> str:
    """Build the JSON input expected by muuh()."""
    return json.dumps({
        "equipe": "IA_Team",
        "question": {
            "categorie_01": {
                "1": query
            }
        }
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_depot_pipeline(query: str) -> Dict[str, Any]:
    """Execute the Graph-RAG pipeline for the depot (products) domain."""
    logger.info("Starting Depot Pipeline | q=%.80s…", query)

    # Step 1: Original retrieval
    json_data = _build_json(query)
    result = muuh(json_data)

    retrieved_docs = result.get("retrieved_documents", [])
    if not retrieved_docs:
        return {
            "answer": "Aucun produit correspondant trouvé dans les documents fournis.",
            "sources": [],
            "faithfulness": 1.0,
            "kg_nodes_used": 0,
            "argument_types": [],
        }

    # Step 2: Normalize to passage format for AdvancedPipeline
    first_doc = retrieved_docs[0]
    doc_text = json.dumps(first_doc, ensure_ascii=False)
    doc_title = first_doc.get("document_title", "")

    retrieved_passages = [{
        "doc_id":   doc_title,
        "doc_name": doc_title,
        "text":     doc_text,
        "score":    first_doc.get("score", 1.0),
    }]

    # Step 3: Source metadata
    sources = []
    for doc in retrieved_docs:
        s3_key = doc.get("doc_french_link") or doc.get("doc_arabic_link")
        if s3_key:
            sources.append({
                "s3_key":   s3_key,
                "filename": doc.get("document_title", ""),
                "category": "Produits",
                "ext":      ".pdf",
                "lang":     "FR" if doc.get("doc_french_link") else "AR",
            })

    # Step 4: Run AdvancedPipeline
    try:
        adv = _get_advanced_pipeline()
        adv_result = adv.run(
            query=query,
            retrieved_passages=retrieved_passages,
            llm_client=get_llm_client(),
            sources=sources,
            max_new_tokens=512,
        )
        logger.info("Depot Pipeline completed | faithfulness=%.2f", adv_result.faithfulness_score)
        return {
            "answer":         adv_result.answer,
            "sources":        adv_result.sources,
            "faithfulness":   adv_result.faithfulness_score,
            "kg_nodes_used":  adv_result.kg_nodes_used,
            "argument_types": adv_result.argument_types,
        }
    except Exception as exc:
        # Graceful degradation
        logger.exception("AdvancedPipeline failed for depot, falling back: %s", exc)
        from local_llm_client import call_local_llm
        response_text = call_local_llm(
            "Tu es un assistant spécialisé dans les produits d'Algérie Télécom. "
            "N'invente AUCUNE donnée.",
            doc_text,
        )
        return {"answer": response_text, "sources": sources}