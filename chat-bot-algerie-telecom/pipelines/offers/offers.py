"""
pipelines/offers/offers.py
===========================
Offers Pipeline — upgraded to Graph-RAG + Citation-Faithful Generation.

Uses AdvancedPipeline for KG enrichment, argument mining, and
hallucination validation.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from local_llm_client import get_llm_client
from pipelines.advanced_pipeline import AdvancedPipeline, get_advanced_pipeline
from .offers_code.Retriever import Retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).parent
_KG_DIR     = _HERE / "offers_code" / "kg"
_DOCS_DIR   = _HERE / "offers_code" / "individual_docs"

# Max characters per passage sent to LLM (keeps prompt under token budget)
_MAX_PASSAGE_CHARS = 2500

# ---------------------------------------------------------------------------
# Singleton pipeline
# ---------------------------------------------------------------------------
_adv_pipeline: AdvancedPipeline | None = None


def _get_advanced_pipeline() -> AdvancedPipeline:
    global _adv_pipeline
    if _adv_pipeline is not None:
        return _adv_pipeline

    pipeline = get_advanced_pipeline(
        domain="offres",
        graph_dir=str(_KG_DIR),
        faithfulness_threshold=0.60,
        annotate_response=True,
    )

    # Build KG from individual offer documents
    documents = []
    if _DOCS_DIR.exists():
        for json_file in sorted(_DOCS_DIR.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                if isinstance(doc, list):
                    documents.extend(doc)
                elif isinstance(doc, dict):
                    doc.setdefault("filename", json_file.stem)
                    documents.append(doc)
            except Exception as e:
                logger.warning("Failed to load %s: %s", json_file, e)

    pipeline.build_graph(documents, force_rebuild=False)
    _adv_pipeline = pipeline
    return _adv_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_retriever_output(retrieved: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """Normalize Retriever output to (context_str, sources_list)."""
    if isinstance(retrieved, dict):
        ctx = retrieved.get("context", "") or ""
        src = retrieved.get("sources", []) or []
        if not isinstance(ctx, str):
            ctx = str(ctx)
        if not isinstance(src, list):
            src = []
        return ctx, src
    return "" if retrieved is None else str(retrieved), []


def _extract_relevant_text(doc: dict) -> str:
    """Extract only the human-readable text fields from a document JSON.

    Instead of dumping the entire 30-40KB JSON, pull out the fields that
    actually contain useful textual information for the LLM.

    The offer documents have this structure:
      - document_id, product_family, technology, customer_segment
      - metadata: {title_fr, title_ar, ...}
      - offer_core: {name_fr, description_fr, pricing_summary_fr, benefits_fr, conditions_fr, ...}
      - faq_fr: [{question, answer}, ...]
      - search_text, dense_text_primary
    """
    parts: List[str] = []

    # --- Title (try multiple paths) ---
    offer_core = doc.get("offer_core", {}) or {}
    metadata = doc.get("metadata", {}) or {}

    title = (
        offer_core.get("name_fr")
        or metadata.get("title_fr")
        or doc.get("document_id")
        or doc.get("title")
        or ""
    )
    if title:
        parts.append(f"Titre: {title}")

    # --- Product family & technology ---
    pf = doc.get("product_family", "")
    tech = doc.get("technology", [])
    if pf:
        parts.append(f"Famille: {pf}")
    if tech:
        parts.append(f"Technologies: {', '.join(tech) if isinstance(tech, list) else tech}")

    # --- Description ---
    desc = offer_core.get("description_fr") or doc.get("description") or doc.get("search_text") or ""
    if desc:
        parts.append(f"Description: {desc[:600]}")

    # --- Pricing (most critical for user queries) ---
    pricing = offer_core.get("pricing_summary_fr") or offer_core.get("pricing_summary_ar") or ""
    if pricing:
        parts.append(f"Tarification: {pricing[:600]}")

    # --- Benefits ---
    benefits = offer_core.get("benefits_fr") or offer_core.get("benefits_ar") or []
    if isinstance(benefits, list) and benefits:
        parts.append("Avantages: " + " | ".join(str(b)[:200] for b in benefits[:5]))
    elif isinstance(benefits, str) and benefits:
        parts.append(f"Avantages: {benefits[:400]}")

    # --- Conditions ---
    conditions = offer_core.get("conditions_fr") or offer_core.get("conditions_ar") or []
    if isinstance(conditions, list) and conditions:
        parts.append("Conditions: " + " | ".join(str(c)[:200] for c in conditions[:5]))
    elif isinstance(conditions, str) and conditions:
        parts.append(f"Conditions: {conditions[:400]}")

    # --- FAQ (first 3) ---
    faq = doc.get("faq_fr") or doc.get("faq_ar") or []
    if isinstance(faq, list):
        for item in faq[:3]:
            if isinstance(item, dict):
                q = item.get("question", "")
                a = item.get("answer", "")
                if q and a:
                    parts.append(f"Q: {q[:150]} R: {a[:250]}")

    if not parts:
        # Final fallback: use search_text or dump truncated JSON
        search_text = doc.get("search_text") or doc.get("dense_text_primary") or ""
        if search_text:
            return str(search_text)[:_MAX_PASSAGE_CHARS]
        return json.dumps(doc, ensure_ascii=False)[:_MAX_PASSAGE_CHARS]

    result = "\n".join(parts)
    return result[:_MAX_PASSAGE_CHARS]


def _parse_passages_from_context(context: str, query: str) -> List[Dict[str, Any]]:
    """Extract passage dicts from the Retriever context string."""
    passages = []
    if not context or context == "No relevant documents found.":
        return passages

    # The retriever returns JSON blocks between === Document N === markers
    doc_blocks = re.split(r"=== Document \d+ ===", context)
    for block in doc_blocks:
        block = block.strip()
        if not block:
            continue
        try:
            doc_data = json.loads(block)
            full_doc = doc_data.get("full_document", doc_data)

            # Extract only relevant text instead of dumping entire JSON
            if isinstance(full_doc, dict):
                doc_text = _extract_relevant_text(full_doc)
            else:
                doc_text = str(full_doc)[:_MAX_PASSAGE_CHARS]

            passages.append({
                "doc_id":   doc_data.get("document_id", ""),
                "doc_name": doc_data.get("document_id", ""),
                "text":     doc_text,
                "score":    doc_data.get("score", 1.0),
            })
        except json.JSONDecodeError:
            # Use raw text block (truncated)
            passages.append({
                "doc_id":   "unknown",
                "doc_name": "unknown",
                "text":     block[:_MAX_PASSAGE_CHARS],
                "score":    1.0,
            })

    return passages


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_offers_pipeline(query: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Execute the Graph-RAG pipeline for the offers domain."""
    logger.info("Starting Offers Pipeline | q=%.80s…", str(query))

    # Step 1: Retrieval (BM25-based)
    retrieved = Retriever(query=query, K=3)
    context, sources = _normalize_retriever_output(retrieved)

    if not context or context.strip() == "No relevant documents found.":
        return {
            "answer": "Aucune offre correspondante trouvée dans les documents fournis.",
            "sources": [],
            "faithfulness": 1.0,
            "kg_nodes_used": 0,
            "argument_types": [],
        }

    # Step 2: Parse into passage list for AdvancedPipeline
    retrieved_passages = _parse_passages_from_context(context, str(query))
    if not retrieved_passages:
        # Fallback: treat entire context as a single passage (truncated)
        retrieved_passages = [{
            "doc_id":   "offers_context",
            "doc_name": "offers_context",
            "text":     context[:_MAX_PASSAGE_CHARS],
            "score":    1.0,
        }]

    logger.info(
        "Passages prepared: %d passages, total_chars=%d",
        len(retrieved_passages),
        sum(len(p["text"]) for p in retrieved_passages),
    )

    # Step 3: Run AdvancedPipeline
    try:
        adv = _get_advanced_pipeline()
        result = adv.run(
            query=str(query),
            retrieved_passages=retrieved_passages,
            llm_client=get_llm_client(),
            sources=sources,
            max_new_tokens=384,  # Offers answers are short — no need for 512
        )
        logger.info("Offers Pipeline completed | faithfulness=%.2f", result.faithfulness_score)
        return {
            "answer":         result.answer,
            "sources":        result.sources,
            "faithfulness":   result.faithfulness_score,
            "kg_nodes_used":  result.kg_nodes_used,
            "argument_types": result.argument_types,
        }
    except Exception as exc:
        # Graceful degradation
        logger.exception("AdvancedPipeline failed for offers, falling back: %s", exc)
        from local_llm_client import call_local_llm
        response_text = call_local_llm(
            "Tu es un assistant spécialisé dans les offres commerciales d'Algérie Télécom. "
            "N'invente AUCUNE donnée. Sois direct et concis.",
            context[:3000],  # Truncate for legacy fallback too
        )
        return {"answer": response_text, "sources": sources}
