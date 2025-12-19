import json
import requests
from typing import Dict, Any

# API endpoint
API_URL = "http://localhost:8001/process-question"

# Mapping from category name to categorie_id
CATEGORY_TO_ID = {
    "offre": "1",  # offres
    "guide": "3",  # guides
    "convention": "2",  # conventions
    "produit": "4"  # produits
}

def process_multi_questions(input_data: Dict[str, Any]) -> Dict[str, Any]:
    equipe = input_data["equipe"]
    questions = input_data["question"]
    
    responses = {}
    
    for category, q_dict in questions.items():
        if category not in CATEGORY_TO_ID:
            print(f"Unknown category: {category}")
            continue
        
        categorie_id = CATEGORY_TO_ID[category]
        responses[category] = {}
        
        for q_id, question_text in q_dict.items():
            payload = {
                "equipe": equipe,
                "question": {
                    "categorie_id": {
                        categorie_id: question_text
                    }
                }
            }
            
            try:
                response = requests.post(API_URL, json=payload, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided")
                else:
                    answer = f"HTTP Error: {response.status_code} {response.text}"
            except Exception as e:
                answer = f"System Error: {str(e)}"
            
            responses[category][q_id] = answer
            print(f"Processed {category} question {q_id}")
    
    output_data = {
        "equipe": equipe,
        "reponses": responses
    }
    
    return output_data

def main():
    # Read input from file
    try:
        with open("test_input.json", "r", encoding="utf-8") as f:
            input_json = json.load(f)
    except FileNotFoundError:
        print("Error: test_input.json not found. Please create the input file.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in test_input.json: {e}")
        return
    
    # Process the questions
    output_data = process_multi_questions(input_json)
    
    # Write to output file
    with open("test_output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("Output written to test_output.json")

if __name__ == "__main__":
    main()