import sys
from pathlib import Path
from typing import Optional, Tuple
from ..models.product_doc import ProductDoc
from .rules import rule_based_filter
from .sparse import SparseRetriever
from .dense import DenseRetriever

# Shared RAG enhancements (RRF + parallel search)
try:
    _rag_root = str(Path(__file__).resolve().parents[5])
    if _rag_root not in sys.path:
        sys.path.insert(0, _rag_root)
    from rag_enhancements import rrf_fusion, parallel_search as _parallel_search
    _RRF_AVAILABLE = True
except ImportError:
    _RRF_AVAILABLE = False
    rrf_fusion = None
    _parallel_search = None

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
        
        # ---- Layers 2 + 3: BM25 and Dense run in PARALLEL then RRF-fused ----
        #
        # Original design was strictly sequential (BM25 → Dense).
        # New design:
        #   - Both retrievers run concurrently (saves ~50 % of layer-2/3 latency)
        #   - Scores merged with Reciprocal Rank Fusion (more robust than
        #     thresholding individual scores)
        #   - Individual thresholds still checked so we can label which layer "won"
        #   - Falls back to original sequential cascading if rag_enhancements
        #     is unavailable.

        run_bm25  = not self.block_bm25_layer
        run_dense = not self.block_dense_layer

        if not run_bm25 and not run_dense:
            if self.verbose:
                print("\n🔤 Layer 2 (BM25) - BLOCKED")
                print("\n🧠 Layer 3 (Dense/Semantic) - BLOCKED")
                print("\n❌ No document found in any layer")
            return None

        # ── Parallel retrieval (RRF path) ────────────────────────────────────
        if _RRF_AVAILABLE and _parallel_search and run_bm25 and run_dense:
            if self.verbose:
                print("\n⚡ Layers 2+3: parallel BM25 + Dense with RRF fusion")

            tasks = {
                "bm25":  lambda: self.sparse.search(query, k=5, candidates=None),
                "dense": lambda: self.dense.search(query, k=5, candidates=None),
            }
            fetched      = _parallel_search(tasks)
            bm25_pairs   = fetched.get("bm25")  or []
            dense_pairs  = fetched.get("dense") or []

            # Build ranked lists using doc.id as identifier
            bm25_ranked  = [(doc.id, score) for doc, score in bm25_pairs]
            dense_ranked = [(doc.id, score) for doc, score in dense_pairs]

            rrf_scores   = rrf_fusion([bm25_ranked, dense_ranked])

            # Quick score maps for threshold checking
            bm25_score_map  = {doc.id: score for doc, score in bm25_pairs}
            dense_score_map = {doc.id: score for doc, score in dense_pairs}
            id_to_doc       = {doc.id: doc for doc, _ in bm25_pairs + dense_pairs}

            if not rrf_scores:
                if self.verbose:
                    print("\n❌ No document found in any layer")
                return None

            best_id    = max(rrf_scores, key=rrf_scores.__getitem__)
            best_rrf   = rrf_scores[best_id]
            best_doc   = id_to_doc[best_id]

            bm25_ok  = bm25_score_map.get(best_id, 0) >= self.bm25_score_threshold
            dense_ok = dense_score_map.get(best_id, 0) >= self.dense_score_threshold

            if not bm25_ok and not dense_ok:
                if self.verbose:
                    print(f"\n   ⚠️ Best RRF doc below both thresholds "
                          f"(bm25={bm25_score_map.get(best_id,0):.3f}, "
                          f"dense={dense_score_map.get(best_id,0):.3f})")
                    print("\n❌ No document found in any layer")
                return None

            layer_name  = "Layer 2+3 (BM25+Dense/RRF)"
            doc_title   = best_doc.raw.get('metadata', {}).get('document_title', 'No Title')
            doc_fr_link = best_doc.raw.get('metadata', {}).get('doc_french_link', 'no link')
            doc_ar_link = best_doc.raw.get('metadata', {}).get('doc_arabic_link', 'no link')
            doc_json    = best_doc.raw

            if self.verbose:
                print(f"   ✅ {layer_name} found match: {doc_title} "
                      f"(rrf={best_rrf:.4f}, "
                      f"bm25={bm25_score_map.get(best_id,0):.3f}, "
                      f"dense={dense_score_map.get(best_id,0):.3f})")

            return (doc_title, layer_name, best_rrf, doc_ar_link, doc_fr_link, doc_json)

        # ── Sequential fallback (original behaviour) ─────────────────────────
        if run_bm25:
            if self.verbose:
                print(f"\n🔤 Layer 2 (BM25) - Starting...")
                print(f"   Score threshold: {self.bm25_score_threshold}")

            sparse_results = self.sparse.search(query, k=1, candidates=None)
            if sparse_results:
                doc, score = sparse_results[0]
                if self.verbose:
                    print(f"   Best score: {score:.3f}")
                if score >= self.bm25_score_threshold:
                    doc_title   = doc.raw.get('metadata', {}).get('document_title', 'No Title')
                    doc_fr_link = doc.raw.get('metadata', {}).get('doc_french_link', 'no link')
                    doc_ar_link = doc.raw.get('metadata', {}).get('doc_arabic_link', 'no link')
                    if self.verbose:
                        print(f"   ✅ Layer 2 match: {doc_title}")
                    return (doc_title, "Layer 2 (BM25)", score, doc_ar_link, doc_fr_link, doc.raw)
                elif self.verbose:
                    print(f"   ⚠️ Score below threshold — moving to Layer 3")
            elif self.verbose:
                print("   ⚠️ Layer 2 found nothing — moving to Layer 3")
        else:
            if self.verbose:
                print("\n🔤 Layer 2 (BM25) - BLOCKED")

        if run_dense:
            if self.verbose:
                print(f"\n🧠 Layer 3 (Dense/Semantic) - Starting...")
                print(f"   Score threshold: {self.dense_score_threshold}")

            dense_results = self.dense.search(query, k=1, candidates=None)
            if dense_results:
                doc, score = dense_results[0]
                if self.verbose:
                    print(f"   Best score: {score:.3f}")
                if score >= self.dense_score_threshold:
                    doc_title   = doc.raw.get('metadata', {}).get('document_title', 'No Title')
                    doc_fr_link = doc.raw.get('metadata', {}).get('doc_french_link', 'no link')
                    doc_ar_link = doc.raw.get('metadata', {}).get('doc_arabic_link', 'no link')
                    if self.verbose:
                        print(f"   ✅ Layer 3 match: {doc_title}")
                    return (doc_title, "Layer 3 (Dense)", score, doc_ar_link, doc_fr_link, doc.raw)
                elif self.verbose:
                    print(f"   ⚠️ Score below threshold")
            elif self.verbose:
                print("   ⚠️ Layer 3 found nothing")
        else:
            if self.verbose:
                print("\n🧠 Layer 3 (Dense/Semantic) - BLOCKED")

        if self.verbose:
            print("\n❌ No document found in any layer")
        return None