# src/retrievers/dense.py
from typing import List, Tuple, Optional
import os
import numpy as np
from ..models.product_doc import ProductDoc

# Choose embedding method: "local" or "gemini"
EMBEDDING_METHOD = os.getenv("EMBEDDING_METHOD", "local")  # Default to local

# ============================================================================
# LOCAL EMBEDDINGS (sentence-transformers)
# ============================================================================
_local_model = None

def _get_local_model():
    """Load sentence-transformers model (cached)"""
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("📦 Loading local embedding model (paraphrase-multilingual-MiniLM-L12-v2)...")
            # This model supports multiple languages including French and Arabic
            _local_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("   ✅ Local model loaded successfully")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. Install with:\n"
                "pip install sentence-transformers"
            )
    return _local_model


# ============================================================================
# GEMINI EMBEDDINGS (original code)
# ============================================================================
GEMINI_EMBED_MODEL = "text-embedding-004"
_gemini_client = None

def _get_gemini_client():
    """Get Gemini API client (original code)"""
    global _gemini_client
    if _gemini_client is None:
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. "
                    "Create a Gemini API key and export it before running the app."
                )
            _gemini_client = genai.Client(api_key=api_key)
        except ImportError:
            raise RuntimeError(
                "google-generativeai not installed. Install with:\n"
                "pip install google-generativeai"
            )
    return _gemini_client


class DenseRetriever:
    """
    Dense retriever supporting both local and Gemini embeddings.
    
    Set EMBEDDING_METHOD environment variable:
    - "local": Use sentence-transformers (no API key needed)
    - "gemini": Use Gemini API embeddings (requires GEMINI_API_KEY)
    
    Example:
        export EMBEDDING_METHOD=local
        python your_script.py
    """
    def __init__(self, docs: List[ProductDoc], method: Optional[str] = None):
        self.docs = docs
        self.method = method or EMBEDDING_METHOD
        
        print(f"🔧 Initializing DenseRetriever with method: {self.method}")
        
        if self.method == "local":
            self._embed_docs_func = self._embed_docs_local
            self._embed_query_func = self._embed_query_local
        elif self.method == "gemini":
            self._embed_docs_func = self._embed_docs_gemini
            self._embed_query_func = self._embed_query_gemini
        else:
            raise ValueError(f"Unknown embedding method: {self.method}. Use 'local' or 'gemini'")
        
        self.doc_embeddings = self._embed_docs_func(docs)  # shape: [N, D]
        print(f"   ✅ Embedded {len(docs)} documents")

    # ========================================================================
    # LOCAL EMBEDDING METHODS
    # ========================================================================
    
    def _embed_docs_local(self, docs: List[ProductDoc]) -> np.ndarray:
        """Embed documents using local sentence-transformers model"""
        model = _get_local_model()
        texts = [d.text for d in docs]
        
        # Encode in batches for efficiency
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # We'll normalize during search
        )
        return embeddings.astype("float32")
    
    def _embed_query_local(self, query: str) -> np.ndarray:
        """Embed query using local sentence-transformers model"""
        model = _get_local_model()
        embedding = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        return embedding.astype("float32")
    
    # ========================================================================
    # GEMINI EMBEDDING METHODS (original code)
    # ========================================================================
    
    def _embed_docs_gemini(self, docs: List[ProductDoc]) -> np.ndarray:
        """Embed documents using Gemini API (original code)"""
        client = _get_gemini_client()
        texts = [d.text for d in docs]
        
        resp = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=texts,
        )
        vectors = [e.values for e in resp.embeddings]
        return np.asarray(vectors, dtype="float32")
    
    def _embed_query_gemini(self, query: str) -> np.ndarray:
        """Embed query using Gemini API (original code)"""
        client = _get_gemini_client()
        
        resp = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=[query],
        )
        vec = resp.embeddings[0].values
        return np.asarray(vec, dtype="float32")
    
    # ========================================================================
    # SEARCH METHOD (unchanged logic, works with both methods)
    # ========================================================================
    
    def search(
        self,
        query: str,
        k: int = 5,
        candidates: Optional[List[ProductDoc]] = None,
    ) -> List[Tuple[ProductDoc, float]]:
        """
        Recherche dense:
        - si `candidates` est fourni, on restreint le calcul à ces docs,
          sinon on utilise tous les docs.
        - retourne [(doc, score_cosinus), ...]
        """
        q_vec = self._embed_query_func(query)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        
        if candidates is not None and len(candidates) > 0:
            candidate_ids = [d.id for d in candidates]
            doc_vecs = self.doc_embeddings[candidate_ids]
            docs = candidates
        else:
            doc_vecs = self.doc_embeddings
            docs = self.docs
        
        norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10
        doc_normed = doc_vecs / norms
        sims = doc_normed @ q_norm  # [N]
        
        order = np.argsort(sims)[::-1][:k]
        
        results: List[Tuple[ProductDoc, float]] = []
        for i in order:
            score = float(sims[i])
            if score <= 0:
                continue
            results.append((docs[i], score))
        
        return results