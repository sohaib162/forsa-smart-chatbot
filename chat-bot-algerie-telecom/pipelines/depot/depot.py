# pipelines/depot/depot.py
from deepseek_client import call_deepseek
import json
from .depot_code.src.muuh import muuh

SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans les produits et équipements d’Algérie Télécom.
Réponds rapidement, précisément et uniquement à la question posée concernant les produits (caractéristiques, modèles, disponibilité, prix si mentionné).

Ne fournis aucune information hors produit.
N’invente aucune donnée.

Si le produit demandé n’existe pas dans les documents fournis, réponds uniquement :
« Produit non disponible dans les documents fournis. »
"""

def build_json(query):
    """
    Builds a JSON string with the given query.
    
    Args:
        query (str): The query string to be included in the JSON
        
    Returns:
        str: The complete JSON as a string
    """
    json_structure = {
        "equipe": "IA_Team",
        "question": {
            "categorie_01": {
                "1": query
            }
        }
    }
    return json.dumps(json_structure, ensure_ascii=False)


def run_depot_pipeline(query):
    # 1. Convert the input dict to a string to pass to the LLM

    json_data = build_json(query)
    result = muuh(json_data)
    
    first_document = result["retrieved_documents"][0] if result["retrieved_documents"] else {}
    result_text = json.dumps(first_document, ensure_ascii=False)
    
    # 3. Call DeepSeek
    print("--- Pipeline: Running Depot ---")
    response_text = call_deepseek(SYSTEM_PROMPT, result_text)
    
    sources = []
    for doc in result.get("retrieved_documents", []):
        s3_key = doc.get("doc_french_link") or doc.get("doc_arabic_link")
        if s3_key:
            sources.append({
                "s3_key": s3_key,
                "filename": doc.get("document_title", ""),
                "category": "Produits",
                "ext": ".pdf",
                "lang": "FR" if doc.get("doc_french_link") else "AR"
            })
    
    return {"answer": response_text, "sources": sources}