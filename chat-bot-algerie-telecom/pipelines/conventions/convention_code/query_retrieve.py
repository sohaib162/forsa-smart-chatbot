# -*- coding: utf-8 -*-
from .retrieval_pipeline import RetrievalPipeline, PipelineConfig
import json
import io
import sys
import os

def retrieve(query: str):
    """Retrieve the top document matching the query."""
    # Suppress pipeline output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    config = PipelineConfig(use_dense_retrieval=False, use_cross_encoder=False)
    pipeline = RetrievalPipeline(config)
    passages_path = os.path.join(os.path.dirname(__file__), "data", "passages.json")
    pipeline.initialize(passages_path=passages_path)
    
    result = pipeline.search(query, top_k=1)
    
    # Restore stdout
    sys.stdout = old_stdout
    
    conventions_path = os.path.join(os.path.dirname(__file__), "data", "conventions.json")
    with open(conventions_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    doc_map = {doc.get("filename", ""): doc for doc in docs}
    
    doc_id = result.top_documents[0].get("doc_id") if result.top_documents else None
    return doc_map.get(doc_id, {}) if doc_id else {}



