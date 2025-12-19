#!/usr/bin/env python3
"""
Q&A Retrieval Processor

Processes question sets, retrieves relevant documents, and outputs results with timing.

Input format:
{
  "equipe": "IA_Team",
  "question": {
    "categorie_01": {
      "1": "Donnez une description du projet",
      "2": "Quelles sont les technologies utilisées ?"
    }
  }
}

Output format (per question):
{
  "equipe": "IA_Team",
  "categorie": "categorie_01",
  "question_id": "1",
  "query": "Donnez une description du projet",
  "retrieval_info": {...},
  "retrieved_documents": [...]
}

Usage:
    python3 scripts/process_qa.py input.json output.json
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.product_doc import load_docs
from src.retrievers.three_layer import ThreeLayerRetriever

def muuh(input_file: str) -> Dict[str, Any]:
    """Load questions from input JSON file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)
    
    
def load_questions(input_file: str) -> Dict[str, Any]:
    """Load questions from input JSON file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_products(products_file: str) -> List:
    """Load product documents"""
    with open(products_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return load_docs(data)


def process_questions(
    questions_data: Dict[str, Any],
    retriever: ThreeLayerRetriever,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Process all questions and retrieve relevant documents
    
    Returns list of results with timing information
    """
    equipe = questions_data.get('equipe', 'Unknown')
    questions = questions_data.get('question', {})
    
    results = []
    total_start = time.time()
    
    print(f"\n{'='*80}")
    print(f"🚀 PROCESSING QUESTIONS FOR: {equipe}")
    print(f"{'='*80}")
    
    question_count = 0
    for categorie, question_dict in questions.items():
        for question_id, query in question_dict.items():
            question_count += 1
            
            print(f"\n📝 Question {question_count}: [{categorie}][{question_id}]")
            print(f"   Query: {query}")
            
            # Time the retrieval
            start_time = time.time()
            result = retriever.retrieve(query)
            end_time = time.time()
            retrieval_time = end_time - start_time
            
            # Build output
            output = {
                "equipe": equipe,
                "categorie": categorie,
                "question_id": question_id,
                "query": query,
                "retrieval_info": {
                    "retrieval_time_seconds": round(retrieval_time, 4),
                    "timestamp": datetime.now().isoformat(),
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
                
                print(f"   ✅ Found: {doc_title}")
                print(f"   📊 Layer: {layer} | Score: {score:.4f}")
                print(f"   ⏱️  Time: {retrieval_time:.4f}s")
            else:
                output["retrieval_info"]["layer_used"] = None
                output["retrieval_info"]["confidence_score"] = 0.0
                output["retrieval_info"]["success"] = False
                
                print(f"   ❌ No document found")
                print(f"   ⏱️  Time: {retrieval_time:.4f}s")
            
            results.append(output)
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    print(f"Total questions processed: {question_count}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average time per question: {total_time/question_count:.4f}s")
    print(f"{'='*80}\n")
    
    return results


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to output JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Results saved to: {output_file}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Q&A with retrieval system')
    parser.add_argument('input', help='Input JSON file with questions')
    parser.add_argument('output', help='Output JSON file for results')
    parser.add_argument('--products', default='data/products.json', 
                       help='Products JSON file (default: data/products.json)')
    parser.add_argument('--bm25-threshold', type=float, default=0.1,
                       help='BM25 score threshold (default: 0.1)')
    parser.add_argument('--dense-threshold', type=float, default=0.1,
                       help='Dense score threshold (default: 0.1)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed retrieval logs')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 Q&A RETRIEVAL SYSTEM")
    print("="*80)
    
    # Load products
    print(f"\n📦 Loading products from: {args.products}")
    start = time.time()
    docs = load_products(args.products)
    load_time = time.time() - start
    print(f"   ✅ Loaded {len(docs)} products in {load_time:.4f}s")
    
    # Initialize retriever
    print(f"\n🔧 Initializing retriever...")
    print(f"   BM25 threshold: {args.bm25_threshold}")
    print(f"   Dense threshold: {args.dense_threshold}")
    
    start = time.time()
    retriever = ThreeLayerRetriever(
        docs,
        bm25_score_threshold=args.bm25_threshold,
        dense_score_threshold=args.dense_threshold,
        verbose=args.verbose
    )
    init_time = time.time() - start
    print(f"   ✅ Retriever initialized in {init_time:.4f}s")
    
    # Load questions
    print(f"\n📄 Loading questions from: {args.input}")
    questions_data = load_questions(args.input)
    
    # Process questions
    results = process_questions(questions_data, retriever, args.verbose)
    
    # Save results
    save_results(results, args.output)
    
    # Print timing breakdown
    print("\n⏱️  TIMING BREAKDOWN:")
    print(f"   Product loading: {load_time:.4f}s")
    print(f"   Retriever init: {init_time:.4f}s")
    total_retrieval_time = sum(r['retrieval_info']['retrieval_time_seconds'] for r in results)
    print(f"   Total retrieval: {total_retrieval_time:.4f}s")
    print(f"   Average per query: {total_retrieval_time/len(results):.4f}s")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()