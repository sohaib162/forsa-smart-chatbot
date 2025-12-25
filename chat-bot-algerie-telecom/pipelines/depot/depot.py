# pipelines/depot/depot.py
from local_llm_client import call_local_llm
import json
from .depot_code.src.muuh import muuh

SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans les produits et équipements d'Algérie Télécom.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à la question précise posée
2. Si la question porte sur UNE caractéristique, ne liste pas toutes les caractéristiques
3. Si la question porte sur UN produit, ne parle pas des autres produits
4. Sois direct - commence directement par la réponse
5. Utilise des listes à puces pour les caractéristiques
6. Met en **gras** les spécifications importantes (modèle, prix, disponibilité)
7. N'invente AUCUNE donnée - utilise uniquement les documents fournis

Si le produit n'existe pas : "Produit non disponible dans les documents fournis."

EXEMPLES :
Question: "Quel est le prix du modem X5 ?"
Réponse: **3500 DZD**

Question: "Quelles sont les caractéristiques du routeur Pro ?"
Réponse:
- Wi-Fi **6 (802.11ax)**
- Portée: **150m²**
- Ports: **4 Gigabit Ethernet**
- Fréquences: **2.4 GHz et 5 GHz**
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
    
    # 3. Call Local LLM
    print("--- Pipeline: Running Depot ---")
    response_text = call_local_llm(SYSTEM_PROMPT, result_text)
    
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