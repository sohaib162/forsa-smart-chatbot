from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from deepseek_client import call_deepseek
from pipelines.guide.guide_code.scripts.retrieval_function import retrieve

SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans les guides et procédures internes d’Algérie Télécom.
Réponds rapidement, clairement et uniquement à la question posée concernant les procédures ou étapes à suivre.

Réponse courte, structurée et factuelle.

Si la procédure demandée n’est pas présente dans les documents fournis, réponds uniquement :
« Procédure non disponible dans les documents fournis. »
"""

def run_guide_pipeline(query: str) -> Dict[str, Any]:
    context = retrieve(query=query, top_k=1)
    retrieved_docs = context.get("retrieved_documents", []) or []
    if not retrieved_docs:
        return {"answer": "No relevant guide found for your question.", "sources": []}

    doc = retrieved_docs[0]
    doc_str = json.dumps(doc, ensure_ascii=False, indent=2)
    context_str = f"Query: {query}\n\nRetrieved Guide Section:\n{doc_str}"

    info = context.get("retrieval_info", {}) or {}
    filename = info.get("filename")
    s3_key = info.get("s3_key")
    ext = Path(filename).suffix if isinstance(filename, str) else ""

    sources = []
    if s3_key and filename:
        sources.append(
            {"s3_key": s3_key, "filename": filename, "category": "Guides", "ext": ext, "lang": "FR"}
        )

    response_text = call_deepseek(SYSTEM_PROMPT, context_str)
    return {"answer": response_text, "sources": sources}
