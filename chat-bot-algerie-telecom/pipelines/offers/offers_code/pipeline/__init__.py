"""
3-Layer Retrieval Pipeline for Algérie Télécom Documents

This package implements a sophisticated retrieval system with:
- Layer 1: Rule-based routing using metadata and tags
- Layer 2: Sparse retrieval using BM25
- Layer 3: Dense retrieval using multilingual embeddings

The pipeline automatically selects the best layer based on confidence scores.
"""

from .loader import load_documents, normalize_text
from .rule_router import RuleRouter
from .sparse_index import SparseIndex

# Dense index is optional (requires sentence-transformers)
try:
    from .dense_index import DenseIndex
    DENSE_AVAILABLE = True
except ImportError:
    DenseIndex = None
    DENSE_AVAILABLE = False

from .pipeline import RetrievalPipeline, build_llm_context

__all__ = [
    'load_documents',
    'normalize_text',
    'RuleRouter',
    'SparseIndex',
    'DenseIndex',
    'RetrievalPipeline',
    'build_llm_context',
    'DENSE_AVAILABLE'
]

__version__ = '1.0.0'
