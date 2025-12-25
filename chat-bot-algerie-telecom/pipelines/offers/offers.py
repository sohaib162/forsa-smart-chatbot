# pipelines/offers/offers.py
from local_llm_client import call_local_llm
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
Tu es un assistant spécialisé dans les offres commerciales d'Algérie Télécom.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à la question précise posée
2. Ne mentionne QUE les informations demandées (prix OU débit OU conditions, etc.)
3. Si la question porte sur UNE offre spécifique, ne parle pas des autres offres
4. Si la question porte sur UN aspect (ex: prix), ne mentionne pas les autres aspects non demandés
5. Sois direct et concis - pas de phrases d'introduction ou de conclusion
6. Utilise le format markdown pour la lisibilité (listes à puces, gras pour les chiffres importants)
7. N'invente AUCUNE information - utilise uniquement les documents fournis

Si l'information demandée n'existe pas dans les documents : "Information non disponible dans les documents fournis."

EXEMPLES :
Question: "Quel est le prix de l'offre Gamers 60 Mbps ?"
Réponse: "**2500 DZD par mois** (avec modems optiques, 1 mois gratuit et frais de pose inclus)"

Question: "Quelle est la durée d'engagement ?"
Réponse: "**12 mois** pour tous les nouveaux abonnements"
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

    # Call Local LLM with retrieved context
    logger.info("Calling Local Qwen LLM with retrieved context...")
    response_text = call_local_llm(SYSTEM_PROMPT, context)

    # Ensure response is string
    if response_text is None:
        response_text = ""
    elif not isinstance(response_text, str):
        response_text = str(response_text)

    logger.info(f"Local LLM response length: {len(response_text)} characters")
    logger.info(f"Response preview: {response_text[:200]}...")

    logger.info("=" * 80)
    logger.info("Offers Pipeline Completed")
    logger.info("=" * 80)

    # IMPORTANT: return dict so the API can expose sources to frontend
    return {"answer": response_text, "sources": sources}
