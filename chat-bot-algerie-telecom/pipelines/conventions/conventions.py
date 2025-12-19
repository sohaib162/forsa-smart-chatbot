from __future__ import annotations

import json
from typing import Any, Dict

from deepseek_client import call_deepseek
from .convention_code.query_retrieve import retrieve

SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans les conventions et partenariats d’Algérie Télécom.
Réponds rapidement, précisément et uniquement à la question posée concernant les conventions (établissements concernés, avantages, conditions, durée).

Ne réponds sur aucune autre catégorie.
N’invente aucune donnée.

Si aucune convention correspondante n’est trouvée dans les documents fournis, réponds uniquement :
« Aucune convention correspondante trouvée dans les documents fournis. »
"""

def run_conventions_pipeline(query: str) -> Dict[str, Any]:
    doc = retrieve(query)
    if not doc:
        return {"answer": "No relevant convention found for your question.", "sources": []}

    doc_str = json.dumps(doc, ensure_ascii=False, indent=2)
    context_str = f"Query: {query}\n\nRetrieved Convention:\n{doc_str}"

    filename = doc.get("filename")
    sources = []
    if filename:
        sources.append({
            "s3_key": filename,
            "filename": filename,
            "category": "Conventions",
            "ext": ".pdf",
            "lang": "FR"
        })

    response_text = call_deepseek(SYSTEM_PROMPT, context_str)
    return {"answer": response_text, "sources": sources}