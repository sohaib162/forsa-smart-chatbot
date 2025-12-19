from typing import List, Tuple

from ..models.product_doc import ProductDoc


def build_context(docs_with_scores: List[Tuple[ProductDoc, float]]) -> str:
    """
    Construit un bloc de contexte lisible pour le LLM.
    """
    parts = []
    for doc, score in docs_with_scores:
        name = doc.raw.get("product_info", {}).get("name", "Produit")
        parts.append(
            f"### {name} (score={score:.3f})\n{doc.text}\n"
        )
    return "\n\n".join(parts)


def build_prompt(query: str, context: str) -> str:
    """
    Prompt principal pour le LLM (en français).
    """
    return f"""
Tu es un assistant support pour les produits et services d'Algérie Télécom.

Consigne importante :
- Réponds en français.
- Ne t'appuie que sur le CONTEXTE fourni.
- Si une information n'est pas présente, dis-le clairement (ne l'invente pas).
- Donne une réponse structurée et compréhensible pour un client.

CONTEXTE:
{context}

QUESTION:
{query}

RÉPONSE (en français):
""".strip()
