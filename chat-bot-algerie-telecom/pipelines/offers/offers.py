# pipelines/offers/offers.py
from deepseek_client import call_deepseek
from .offers_code.Retriever import Retriever
import json
import logging
from typing import Any, Dict, List, Tuple, Union

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans les offres commerciales d’Algérie Télécom.
Réponds rapidement, précisément et uniquement à la question posée concernant les offres (prix, débit, durée, conditions, éligibilité).

Va directement à l’essentiel, sans phrases inutiles.
N’invente aucune information.

Si l’information demandée n’existe pas dans les documents fournis, réponds uniquement :
« Information non disponible dans les documents fournis. »
"""

def _normalize_retriever_output(retrieved: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retriever can return:
      - dict: {"context": "...", "sources": [...]}
      - str: "...."
    Normalize to (context_str, sources_list).
    """
    if isinstance(retrieved, dict):
        ctx = retrieved.get("context", "") or ""
        src = retrieved.get("sources", []) or []
        # ensure types
        if not isinstance(ctx, str):
            ctx = str(ctx)
        if not isinstance(src, list):
            src = []
        return ctx, src
    # fallback: treat as string
    return "" if retrieved is None else str(retrieved), []

def run_offers_pipeline(query: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("Starting Offers Pipeline")
    logger.info("=" * 80)

    # input preview (stringify safely)
    try:
        user_content_str = json.dumps(query, indent=2, ensure_ascii=False)
    except Exception:
        user_content_str = str(query)

    logger.info(f"Input received (preview): {user_content_str[:200]}...")
    logger.info("Requesting K=3 documents from retriever")

    # Call retriever
    retrieved = Retriever(query=query, K=3)
    context, sources = _normalize_retriever_output(retrieved)

    # Log retriever results
    logger.info(f"Retriever returned context (length: {len(context)} characters), sources: {len(sources)}")

    if context and context.strip() and context != "No relevant documents found.":
        logger.info("-" * 80)
        logger.info("RETRIEVED CONTEXT FOR LLM:")
        logger.info("-" * 80)
        logger.info(f"\n{context[:1000]}...")
        logger.info(f"\n[Total context length: {len(context)} characters]")
        logger.info("-" * 80)
    else:
        logger.warning("⚠️ Retriever returned no usable context!")

    # Call DeepSeek LLM with retrieved context
    logger.info("Calling DeepSeek LLM with retrieved context...")
    response_text = call_deepseek(SYSTEM_PROMPT, context)

    # Ensure response is string
    if response_text is None:
        response_text = ""
    elif not isinstance(response_text, str):
        response_text = str(response_text)

    logger.info(f"DeepSeek response length: {len(response_text)} characters")
    logger.info(f"Response preview: {response_text[:200]}...")

    logger.info("=" * 80)
    logger.info("Offers Pipeline Completed")
    logger.info("=" * 80)

    # IMPORTANT: return dict so the API can expose sources to frontend
    return {"answer": response_text, "sources": sources}
