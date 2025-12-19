from typing import Optional, Tuple
from ..models.product_doc import ProductDoc
from .rules import rule_based_filter
from .sparse import SparseRetriever
from .dense import DenseRetriever

class ThreeLayerRetriever:
    """
    Cascading retrieval system that returns ONLY ONE document:
    1. Layer 1 (Rules): If finds match → return it
    2. Layer 2 (BM25): If Layer 1 fails → try BM25 with score threshold
    3. Layer 3 (Dense): If Layer 2 fails → use dense semantic search
    """
    def __init__(
        self, 
        docs: list[ProductDoc],
        # Layer blocking flags
        block_rule_layer: bool = False,
        block_bm25_layer: bool = False,
        block_dense_layer: bool = False,
        # Score thresholds for accepting results
        bm25_score_threshold: float = 0.55,  # Min BM25 score to accept result
        dense_score_threshold: float = 0.1,  # Min dense score to accept result
        verbose: bool = True,  # Control print output
    ):
        self.docs = docs
        self.sparse = SparseRetriever(docs)
        self.dense = DenseRetriever(docs)
        
        # Layer blocking flags
        self.block_rule_layer = block_rule_layer
        self.block_bm25_layer = block_bm25_layer
        self.block_dense_layer = block_dense_layer
        
        # Thresholds
        self.bm25_score_threshold = bm25_score_threshold
        self.dense_score_threshold = dense_score_threshold
        
        # Verbose mode
        self.verbose = verbose

    def retrieve(self, query: str) -> Optional[Tuple[str, str, float , str, str, str]]:
        """
        Returns: (document_title, layer_name, score) or None if no match found
        
        Cascades through layers until ONE good document is found.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔍 Query: {query}")
            print(f"{'='*60}")
        
        # ---- Layer 1: Rule-based filtering ----
        if not self.block_rule_layer:
            if self.verbose:
                print("\n📋 Layer 1 (Rule-based) - Starting...")
            
            candidates = rule_based_filter(query, self.docs)
            
            if self.verbose:
                print(f"   Found {len(candidates)} candidates")
            
            if candidates:
                # Return the first match (best match)
                doc = candidates[0]
                doc_title = doc.raw.get('metadata', {}).get('document_title', 'No Title')
                doc_french_link = doc.raw.get('metadata', {}).get('doc_french_link', 'no link')
                doc_arabic_link = doc.raw.get('metadata', {}).get('doc_arabic_link', 'no link')
                doc_jason = doc.raw
                
                if self.verbose:
                    print(f"   ✅ Layer 1 found match: {doc_title}")
                    print(f"   🎯 Returning result from Layer 1")
                
                return (doc_title, "Layer 1 (Rule-based)", 1.0, doc_french_link, doc_arabic_link, doc_jason)
            
            if self.verbose:
                print("   ⚠️ Layer 1 found nothing - moving to Layer 2")
        else:
            if self.verbose:
                print("\n📋 Layer 1 (Rule-based) - BLOCKED")
        
        # ---- Layer 2: BM25 sparse retrieval ----
        if not self.block_bm25_layer:
            if self.verbose:
                print(f"\n🔤 Layer 2 (BM25) - Starting...")
                print(f"   Score threshold: {self.bm25_score_threshold}")
            
            sparse_results = self.sparse.search(
                query,
                k=1,  # Only need top 1 result
                candidates=None,
            )
            
            if sparse_results:
                doc, score = sparse_results[0]
                
                if self.verbose:
                    print(f"   Best score: {score:.3f}")
                
                if score >= self.bm25_score_threshold:
                    doc_title = doc.raw.get('metadata', {}).get('document_title', 'No Title')
                    doc_french_link = doc.raw.get('metadata', {}).get('doc_french_link', 'no link')
                    doc_arabic_link = doc.raw.get('metadata', {}).get('doc_arabic_link', 'no link')
                    doc_jason = doc.raw
                    
                    if self.verbose:
                        print(f"   ✅ Layer 2 found match: {doc_title} (score: {score:.3f})")
                        print(f"   🎯 Returning result from Layer 2")
                    
                    return (doc_title, "Layer 2 (BM25)", score, doc_arabic_link, doc_french_link, doc_jason)
                else:
                    if self.verbose:
                        print(f"   ⚠️ Score {score:.3f} below threshold {self.bm25_score_threshold} - moving to Layer 3")
            else:
                if self.verbose:
                    print("   ⚠️ Layer 2 found nothing - moving to Layer 3")
        else:
            if self.verbose:
                print("\n🔤 Layer 2 (BM25) - BLOCKED")
        
        # ---- Layer 3: Dense semantic search ----
        if not self.block_dense_layer:
            if self.verbose:
                print(f"\n🧠 Layer 3 (Dense/Semantic) - Starting...")
                print(f"   Score threshold: {self.dense_score_threshold}")
            
            dense_results = self.dense.search(
                query,
                k=1,  # Only need top 1 result
                candidates=None,
            )
            
            if dense_results:
                doc, score = dense_results[0]
                
                if self.verbose:
                    print(f"   Best score: {score:.3f}")
                
                if score >= self.dense_score_threshold:
                    doc_title = doc.raw.get('metadata', {}).get('document_title', 'No Title')
                    doc_french_link = doc.raw.get('metadata', {}).get('doc_french_link', 'no link')
                    doc_arabic_link = doc.raw.get('metadata', {}).get('doc_arabic_link', 'no link')
                    doc_jason = doc.raw
                    if self.verbose:
                        print(f"   ✅ Layer 3 found match: {doc_title} (score: {score:.3f})")
                        print(f"   🎯 Returning result from Layer 3")
                    
                    return (doc_title, "Layer 3 (Dense)", score,doc_arabic_link, doc_french_link, doc_jason)
                else:
                    if self.verbose:
                        print(f"   ⚠️ Score {score:.3f} below threshold {self.dense_score_threshold}")
            else:
                if self.verbose:
                    print("   ⚠️ Layer 3 found nothing")
        else:
            if self.verbose:
                print("\n🧠 Layer 3 (Dense/Semantic) - BLOCKED")
        
        if self.verbose:
            print("\n❌ No document found in any layer")
        
        return None