from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from local_llm_client import call_local_llm
from pipelines.guide.guide_code.scripts.retrieval_function import retrieve

SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans les guides et procédures internes d'Algérie Télécom.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à la question posée - ne donne pas d'informations supplémentaires
2. Si la question porte sur UNE étape spécifique, ne décris pas toute la procédure
3. Sois direct et concis - commence directement par la réponse
4. Utilise des listes numérotées pour les étapes, des puces pour les points
5. Met en **gras** les éléments importants (boutons, champs, actions)
6. N'invente AUCUNE information - utilise uniquement les documents fournis

Si la procédure demandée n'existe pas : "Procédure non disponible dans les documents fournis."

EXEMPLES :
Question: "Comment créer un nouveau client ?"
Réponse:
1. Cliquer sur **"Nouveau Client"**
2. Remplir le **formulaire** avec les informations requises
3. Valider avec **"Enregistrer"**

Question: "Quel est le délai de traitement ?"
Réponse: **24 à 48 heures** ouvrables
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

    response_text = call_local_llm(SYSTEM_PROMPT, context_str)
    return {"answer": response_text, "sources": sources}
