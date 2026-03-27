"""
pipelines/conventions/conventions.py
=====================================
Upgraded Conventions Pipeline — Graph + Argument Mining + Citation-Faithful Generation.

Drop-in replacement for the previous bare call_local_llm() approach.
The FastAPI endpoint (main.py) calls run_conventions_pipeline(query) unchanged.

What changed vs the old version
---------------------------------
Old:  retrieve(query) → JSON-dump → call_local_llm(SYSTEM_PROMPT, context_str)
New:  retrieve(query) → AdvancedPipeline.run() which does:
        1. Map retrieved doc to KG node
        2. 2-hop KG expansion (related articles, penalty clauses, definitions)
        3. Argument mining over the expanded context
        4. Citation-faithful prompt construction
        5. LLM generation with lower temperature
        6. Post-generation hallucination validation
        7. Returns answer + faithfulness score + sources
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from local_llm_client import get_llm_client
from pipelines.advanced_pipeline import AdvancedPipeline, get_advanced_pipeline
from .convention_code.query_retrieve import retrieve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to this file's directory)
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).parent
_CONV_DATA   = _HERE / "convention_code" / "data" / "conventions.json"
_KG_DIR      = _HERE / "convention_code" / "kg"
_PASSAGES    = _HERE / "convention_code" / "data" / "passages.json"

# ---------------------------------------------------------------------------
# Singleton pipeline — built once on first call, then cached in memory
# ---------------------------------------------------------------------------
_adv_pipeline: AdvancedPipeline | None = None


def _get_advanced_pipeline() -> AdvancedPipeline:
    global _adv_pipeline
    if _adv_pipeline is not None:
        return _adv_pipeline

    pipeline = get_advanced_pipeline(
        domain="conventions",
        graph_dir=str(_KG_DIR),
        faithfulness_threshold=0.60,
        annotate_response=True,
    )

    # Build graph from conventions.json if not already cached on disk
    if _CONV_DATA.exists():
        with open(_CONV_DATA, "r", encoding="utf-8") as f:
            documents = json.load(f)
        pipeline.build_graph(documents, force_rebuild=False)
    else:
        logger.warning(
            "conventions.json not found at %s — KG will be empty; "
            "falling back to vector-only retrieval.",
            _CONV_DATA,
        )
        # Build an empty graph so run() doesn't crash
        pipeline.build_graph([], force_rebuild=False)

    _adv_pipeline = pipeline
    return _adv_pipeline


# ---------------------------------------------------------------------------
# Public entry point (called by main.py)
# ---------------------------------------------------------------------------

def run_conventions_pipeline(query: str) -> Dict[str, Any]:
    """
    Execute the full Graph-RAG + Argument Mining + Citation-Faithful pipeline
    for the conventions domain.

    Returns
    -------
    {
        "answer":           str,    # citation-annotated answer
        "sources":          list,   # source metadata for the frontend
        "faithfulness":     float,  # 0.0 – 1.0 citation validation score
        "kg_nodes_used":    int,
        "argument_types":   list,
    }
    """
    # ---- Step 1: original BM25 retrieval (unchanged) ----
    doc = retrieve(query)
    if not doc:
        return {
            "answer": "Aucune convention correspondante trouvée dans les documents fournis.",
            "sources": [],
            "faithfulness": 1.0,
            "kg_nodes_used": 0,
            "argument_types": [],
        }

    # ---- Step 2: normalise to a passage list (shape expected by AdvancedPipeline) ----
    filename = doc.get("filename", "")
    retrieved_passages = [
        {
            "doc_id":      filename,
            "doc_name":    filename,
            "entity_code": doc.get("entity_code", ""),
            "text":        json.dumps(doc, ensure_ascii=False),
            "score":       1.0,
        }
    ]

    # ---- Step 3: build source metadata for the frontend ----
    sources = []
    if filename:
        sources.append(
            {
                "s3_key":   filename,
                "filename": filename,
                "category": "Conventions",
                "ext":      ".pdf",
                "lang":     "FR",
            }
        )

    # ---- Step 4: run the advanced pipeline ----
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
        # Graceful degradation: fall back to plain LLM call so the API stays up
        logger.exception("AdvancedPipeline failed, falling back to plain LLM: %s", exc)
        from local_llm_client import call_local_llm
        context_str = f"Query: {query}\n\nRetrieved Convention:\n{json.dumps(doc, ensure_ascii=False, indent=2)}"
        response_text = call_local_llm(
            "Tu es un assistant spécialisé dans les conventions d'Algérie Télécom. "
            "N'invente AUCUNE donnée.",
            context_str,
        )
        return {"answer": response_text, "sources": sources}
