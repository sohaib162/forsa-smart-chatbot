import json
import time
from datetime import datetime
from .models.product_doc import load_docs
from .retrievers.three_layer import ThreeLayerRetriever
from typing import Dict, Any
import json
import time
from datetime import datetime
from typing import Dict, Any, List

def load_single_question(input_json: str) -> Dict[str, Any]:
    """Load a single question from JSON input"""
    try:
        return json.loads(input_json)  # Expecting input as a JSON string
    except json.JSONDecodeError:
        return {"status": "error", "message": "Invalid JSON format."}

def call_pipeline(question_data: Dict[str, Any], docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process a single question with the retrieval pipeline"""
    
    equipe = question_data.get('equipe', 'Unknown')

    # Access the question and category based on the new structure
    categorie = list(question_data['question'].keys())[0]  # Assuming you're always working with the first category, e.g., "categorie_01"
    question_id = list(question_data['question'][categorie].keys())[0]  # Assuming you're working with the first question in that category
    query = question_data['question'][categorie].get(question_id, '')

    # Initialize the retriever
    retriever = ThreeLayerRetriever(docs)
    
    # Start the retrieval process
    start_time = time.time()
    result = retriever.retrieve(query)
    end_time = time.time()
    
    # Calculate the time taken for retrieval
    retrieval_time = end_time - start_time
    
    # Prepare the output
    output = {
        "equipe": equipe,
        "question_id": question_id,
        "query": query,
        "retrieval_info": {
            "retrieval_time_seconds": round(retrieval_time, 4),
            "timestamp": datetime.now().isoformat(),
            "success": False
        },
        "retrieved_documents": []
    }

    if result:
        doc_title, layer, score, doc_french_link, doc_arabic_link, jason = result
        
        output["retrieval_info"]["layer_used"] = layer
        output["retrieval_info"]["confidence_score"] = round(score, 4)
        output["retrieval_info"]["success"] = True
        
        # Add document info
        output["retrieved_documents"].append({
            "rank": 1,
            "document_title": doc_title,
            "layer": layer,
            "score": round(score, 4),
            "doc_arabic_link": doc_arabic_link,
            "doc_french_link": doc_french_link,
            "jason_snippet": jason,
        })

        print(f"✅ Found: {doc_title}")
        print(f"📊 Layer: {layer} | Score: {score:.4f}")
        print(f"⏱️  Time: {retrieval_time:.4f}s")
    else:
        print(f"❌ No document found")
        print(f"⏱️  Time: {retrieval_time:.4f}s")

    return output

def muuh(input_json: str):
    products_json = 'pipelines/depot/depot_code/data/products.json'  # Path to your product JSON file
    return process_single_question(input_json,products_json)
    



def process_single_question(input_json: str, products_json: str) -> Dict[str, Any]:
    """Main function to handle the processing of a single question"""

    # Load the question from the provided JSON input
    question_data = load_single_question(input_json)
    
    if "status" in question_data and question_data["status"] == "error":
        return question_data  # Return the error message
    
    # Load products (documents)
    with open(products_json, 'r', encoding='utf-8') as f:
        products_data = json.load(f)
    
    docs = load_docs(products_data)

    # Call the pipeline with the question and products
    result = call_pipeline(question_data, docs)

    # Return the result of the question processing
    return result



