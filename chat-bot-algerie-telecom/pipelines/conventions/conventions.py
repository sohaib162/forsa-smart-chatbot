from __future__ import annotations

import json
from typing import Any, Dict

from local_llm_client import call_local_llm
from .convention_code.query_retrieve import retrieve

SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans les conventions et partenariats d'Algérie Télécom.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à la question spécifique posée
2. Si la question porte sur UN aspect (ex: avantages), ne mentionne pas les autres aspects
3. Si la question porte sur UNE convention, ne liste pas toutes les conventions
4. Sois direct - commence directement par la réponse, sans introduction
5. Utilise des listes à puces pour les avantages/conditions
6. Met en **gras** les informations clés (noms d'établissements, montants, dates)
7. N'invente AUCUNE donnée - utilise uniquement les documents fournis

Si la convention n'existe pas : "Aucune convention correspondante trouvée dans les documents fournis."

EXEMPLES :
Question: "Quels sont les avantages de la convention Université ?"
Réponse:
- Réduction de **30%** sur les abonnements
- **Installation gratuite**
- Support technique prioritaire

Question: "Quelle est la durée de validité ?"
Réponse: **12 mois** renouvelable automatiquement
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

    response_text = call_local_llm(SYSTEM_PROMPT, context_str)
    return {"answer": response_text, "sources": sources}