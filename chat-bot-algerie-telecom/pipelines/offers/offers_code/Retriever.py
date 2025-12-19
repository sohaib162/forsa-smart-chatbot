import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pipeline import load_documents, RetrievalPipeline

_S3_INDEX_CACHE: Optional[List[Dict[str, Any]]] = None

def _load_s3_index() -> List[Dict[str, Any]]:
    global _S3_INDEX_CACHE
    if _S3_INDEX_CACHE is not None:
        return _S3_INDEX_CACHE

    s3_index_path = os.getenv("S3_INDEX_PATH", "/app/s3_index.json")
    try:
        with open(s3_index_path, "r", encoding="utf-8") as f:
            _S3_INDEX_CACHE = json.load(f)
    except Exception:
        _S3_INDEX_CACHE = []
    return _S3_INDEX_CACHE

def _index_by_filename(s3_index: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in s3_index or []:
        fn = (item.get("filename") or "").strip()
        if fn:
            out[fn.lower()] = item
    return out

def _doc_sources_from_retrieved(retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    s3_index = _load_s3_index()
    by_fn = _index_by_filename(s3_index)
    sources: List[Dict[str, Any]] = []
    seen = set()

    for doc_item in retrieved_docs:
        full_doc = doc_item.get("full_document_json") or doc_item.get("document") or doc_item
        file_path = full_doc.get("_file_path") if isinstance(full_doc, dict) else None
        if not file_path:
            continue

        json_name = Path(file_path).name
        stem = Path(json_name).stem
        candidates = [f"{stem}.docx", f"{stem}.DOCX", f"{stem}.pdf", f"{stem}.odt"]

        matched = None
        for cand in candidates:
            matched = by_fn.get(cand.lower())
            if matched:
                break

        if not matched:
            for k, v in by_fn.items():
                if Path(k).stem.lower() == stem.lower():
                    matched = v
                    break

        if matched:
            key = matched.get("s3_key")
            if key and key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "s3_key": matched.get("s3_key"),
                        "filename": matched.get("filename"),
                        "category": matched.get("category"),
                        "ext": matched.get("ext"),
                        "lang": matched.get("lang"),
                    }
                )

    return sources

def Retriever(query: str, K: int) -> Dict[str, Any]:
    data_dir = Path(__file__).resolve().parent / "individual_docs"

    docs = load_documents(str(data_dir))
    pipeline = RetrievalPipeline(docs=docs, use_dense=False)

    result = pipeline.search(query, top_k=K)
    retrieved = result.get("retrieved_documents", [])

    organized = []
    for rank, doc_item in enumerate(retrieved, start=1):
        score = float(doc_item.get("score", 0.0) or 0.0)
        layer = doc_item.get("layer", doc_item.get("source", "sparse"))
        full_doc = doc_item.get("full_document_json") or doc_item.get("document") or doc_item

        metadata = full_doc.get("metadata") if isinstance(full_doc, dict) else None
        entry = {
            "rank": rank,
            "score": score,
            "layer": layer,
            "document_id": full_doc.get("document_id") if isinstance(full_doc, dict) else None,
            "metadata": metadata,
            "full_document": full_doc,
        }
        organized.append(json.dumps(entry, ensure_ascii=False))

    sources = _doc_sources_from_retrieved(retrieved)

    if not organized:
        return {"context": "No relevant documents found.", "sources": []}

    context_str = f"Retrieved {len(organized)} documents for query: '{query}'\n\n"
    for i, doc_json_str in enumerate(organized, 1):
        context_str += f"=== Document {i} ===\n{doc_json_str}\n\n"

    return {"context": context_str, "sources": sources}
