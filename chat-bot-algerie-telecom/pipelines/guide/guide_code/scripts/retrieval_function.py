"""
Retrieval Function
==================
A function that takes a query string and returns structured JSON output
with retrieval info and full document metadata.

Usage:
    from scripts.retrieval_function import retrieve
    result = retrieve("Comment faire une recharge par bon de commande?")
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.step4_query_pipeline import QueryPipeline

# Global pipeline instance (lazy loaded)
_pipeline = None
_guides_data = None


def _get_pipeline() -> QueryPipeline:
    """Get or create the pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = QueryPipeline()
        _pipeline.connect()
    return _pipeline


def _load_guides_data() -> Dict[str, Any]:
    """Load the full guide metadata from Guide_NGBSS.json"""
    global _guides_data
    if _guides_data is None:
        guides_path = Path(__file__).parent.parent / "data" / "Guide_NGBSS.json"
        with open(guides_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Index by guide_id for fast lookup
        _guides_data = {guide["id"]: guide for guide in data.get("guides", [])}
    return _guides_data


def retrieve(query: str, top_k: int = 1) -> Dict[str, Any]:
    """
    Retrieve documents for a query and return structured JSON output.
    
    Args:
        query: The search query string
        top_k: Number of results to return (default: 1)
    
    Returns:
        Dictionary with structure:
        {
            "query": str,
            "retrieval_info": {
                "score": float,
                "guide_title": str,
                "section_title": str,
                "filename": str,
                "s3_key": str,
                "guide_id": str,
                "doc_type": str,
                "latency_ms": float,
                "bm25_score": float,
                "dense_score": float,
                "hybrid_score": float,
                "rerank_score": float
            },
            "retrieved_documents": [
                {
                    "id": str,
                    "title": str,
                    "system": str,
                    "business_process": str,
                    "filename": str,
                    "relative_path": str,
                    "language": list,
                    "date": str,
                    "tags": list,
                    "summary": str,
                    "prerequisites": list,
                    "sections": list,
                    "s3_key": str
                }
            ]
        }
    """
    pipeline = _get_pipeline()
    guides_data = _load_guides_data()
    
    # Perform search
    results, timing = pipeline.search(query, top_k=top_k)
    
    if not results:
        return {
            "query": query,
            "retrieval_info": None,
            "retrieved_documents": []
        }
    
    # Build retrieval_info from the top result
    top_result = results[0]
    retrieval_info = {
        "score": round(top_result.final_score, 4),
        "guide_title": top_result.guide_title,
        "section_title": top_result.section_title,
        "filename": top_result.filename,
        "s3_key": top_result.s3_key,
        "guide_id": top_result.guide_id,
        "doc_type": top_result.doc_type,
        "doc_id": top_result.doc_id,
        "latency_ms": round(timing.get("total_ms", 0), 2),
        "bm25_score": round(top_result.bm25_score, 4),
        "dense_score": round(top_result.dense_score, 4),
        "hybrid_score": round(top_result.hybrid_score, 4),
        "rerank_score": round(top_result.rerank_score, 4),
        "text_preview": top_result.text[:500] + "..." if len(top_result.text) > 500 else top_result.text
    }
    
    # Build retrieved_documents with full guide info
    retrieved_documents = []
    seen_guide_ids = set()
    
    for result in results:
        guide_id = result.guide_id
        
        # Skip if we've already added this guide
        if guide_id in seen_guide_ids:
            continue
        seen_guide_ids.add(guide_id)
        
        # Get full guide data
        guide_full = guides_data.get(guide_id, {})
        
        if guide_full:
            doc = {
                "id": guide_full.get("id", guide_id),
                "title": guide_full.get("title", result.guide_title),
                "system": guide_full.get("system", "NGBSS"),
                "business_process": guide_full.get("business_process", result.business_process),
                "filename": guide_full.get("filename", result.filename),
                "relative_path": guide_full.get("relative_path", result.relative_path),
                "language": guide_full.get("language", []),
                "date": guide_full.get("date", result.date),
                "tags": guide_full.get("tags", result.tags),
                "summary": guide_full.get("summary", result.summary),
                "prerequisites": guide_full.get("prerequisites", []),
                "sections": guide_full.get("sections", []),
                "s3_key": guide_full.get("s3_key", result.s3_key)
            }
        else:
            # Fallback to result metadata if guide not found
            doc = {
                "id": guide_id,
                "title": result.guide_title,
                "system": "NGBSS",
                "business_process": result.business_process,
                "filename": result.filename,
                "relative_path": result.relative_path,
                "language": [],
                "date": result.date,
                "tags": result.tags,
                "summary": result.summary,
                "prerequisites": [],
                "sections": [],
                "s3_key": result.s3_key
            }
        
        retrieved_documents.append(doc)
    
    return {
        "query": query,
        "retrieval_info": retrieval_info,
        "retrieved_documents": retrieved_documents
    }


def retrieve_json(query: str, top_k: int = 1, pretty: bool = True) -> str:
    """
    Same as retrieve() but returns a JSON string.
    
    Args:
        query: The search query string
        top_k: Number of results to return
        pretty: Whether to pretty-print the JSON
    
    Returns:
        JSON string
    """
    result = retrieve(query, top_k)
    if pretty:
        return json.dumps(result, ensure_ascii=False, indent=2)
    return json.dumps(result, ensure_ascii=False)


def retrieve_to_file(query: str, output_path: str, top_k: int = 1) -> Dict[str, Any]:
    """
    Retrieve and save results to a JSON file.
    
    Args:
        query: The search query string
        output_path: Path to save the JSON file
        top_k: Number of results to return
    
    Returns:
        The result dictionary
    """
    result = retrieve(query, top_k)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# Demo / Test
if __name__ == "__main__":
    print("=" * 60)
    print("Retrieval Function Demo")
    print("=" * 60)
    
    # Test query
    test_query = "Comment faire une recharge par bon de commande?"
    
    print(f"\nQuery: {test_query}\n")
    print("-" * 60)
    
    # Get result
    result = retrieve(test_query, top_k=1)
    
    # Print formatted output
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Also save to file
    output_file = Path(__file__).parent.parent / "retrieval_output.json"
    retrieve_to_file(test_query, str(output_file))
    print(f"\n✓ Results saved to: {output_file}")
