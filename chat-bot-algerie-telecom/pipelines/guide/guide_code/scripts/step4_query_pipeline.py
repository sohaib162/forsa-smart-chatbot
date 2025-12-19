"""
Step 4: Query Pipeline
======================
Complete retrieval pipeline combining:
1. Query preprocessing & keyword extraction
2. Metadata filtering
3. BM25 (sparse) retrieval
4. Dense (vector) retrieval
5. Hybrid fusion scoring
6. Optional cross-encoder reranking

Target: <1 second latency, >85% accuracy

Run independently: python -m scripts.step4_query_pipeline
"""
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    TAG_KEYWORDS, DENSE_WEIGHT, BM25_WEIGHT,
    BM25_TOP_K, DENSE_TOP_K, FINAL_TOP_K, RERANK_TOP_K,
    ENABLE_RERANKING, RERANKER_MODEL, USE_BIENCODER_RERANKER,
    E5_QUERY_PREFIX, TITLE_BOOST
)
from scripts.step2_sparse_index import BM25Index
from scripts.step3_dense_index import DenseIndex, get_embedding_model

# Lazy load rerankers
_cross_encoder_reranker = None
_biencoder_model = None


def get_cross_encoder_reranker():
    """Lazy load cross-encoder reranker (slow but accurate)"""
    global _cross_encoder_reranker
    if _cross_encoder_reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"→ Loading cross-encoder reranker: {RERANKER_MODEL}")
        _cross_encoder_reranker = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder_reranker


def get_biencoder_reranker():
    """
    Get bi-encoder model for fast reranking.
    Reuses the same embedding model used for dense retrieval.
    """
    global _biencoder_model
    if _biencoder_model is None:
        print("→ Loading bi-encoder reranker (reusing embedding model)")
        _biencoder_model = get_embedding_model()
    return _biencoder_model


def get_reranker():
    """Legacy function - returns cross-encoder for backward compatibility"""
    return get_cross_encoder_reranker()


@dataclass
class RetrievalResult:
    """Single retrieval result with scores and metadata"""
    doc_id: str
    doc_type: str
    text: str
    guide_id: str
    guide_title: str
    section_title: Optional[str]
    filename: str
    relative_path: str
    tags: List[str]
    
    # Scores
    bm25_score: float = 0.0
    dense_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    
    # Additional metadata
    summary: str = ""
    business_process: str = ""
    date: str = ""
    
    # S3 storage
    s3_key: str = ""
    url: str = ""


class QueryPipeline:
    """
    Hybrid retrieval pipeline for NGBSS guides
    
    Pipeline stages:
    1. Preprocess query
    2. Extract metadata filters
    3. Sparse retrieval (BM25)
    4. Dense retrieval (vectors)
    5. Hybrid fusion
    6. (Optional) Reranking
    """
    
    def __init__(
        self,
        bm25_index: BM25Index = None,
        dense_index: DenseIndex = None,
        enable_reranking: bool = ENABLE_RERANKING
    ):
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.enable_reranking = enable_reranking
        self._connected = False
    
    def connect(self):
        """Initialize index connections"""
        if self.bm25_index is None:
            self.bm25_index = BM25Index()
        
        self.bm25_index.connect()
        
        # Try to connect dense index, but don't fail if unavailable
        try:
            if self.dense_index is None:
                self.dense_index = DenseIndex()
            self.dense_index.connect()
            self._dense_available = True
        except Exception as e:
            print(f"⚠ Dense index unavailable: {e}")
            self._dense_available = False
        
        self._connected = True
        return self
    
    def close(self):
        """Close connections"""
        if self.bm25_index:
            self.bm25_index.close()
        self._connected = False
    
    def __enter__(self):
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # =========================================================================
    # Stage 1: Query Preprocessing
    # =========================================================================
    
    def preprocess_query(self, query: str) -> str:
        """
        Normalize and clean query text
        - Lowercase
        - Remove extra whitespace
        - Keep special characters that might be meaningful (%, numbers)
        """
        # Normalize whitespace
        query = ' '.join(query.split())
        return query
    
    # =========================================================================
    # Stage 2: Metadata Filter Extraction
    # =========================================================================
    
    def extract_filters(self, query: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Extract metadata filters from query keywords
        
        Returns:
            Tuple of (tag_filters, additional_filters)
        """
        query_lower = query.lower()
        tag_filters = []
        
        # Check for known keywords
        for keyword, tag_values in TAG_KEYWORDS.items():
            # Use word boundaries for matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, query_lower):
                tag_filters.extend(tag_values)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tag_filters:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        return unique_tags, {}
    
    # =========================================================================
    # Stage 3 & 4: Sparse and Dense Retrieval
    # =========================================================================
    
    def sparse_retrieve(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
        doc_type: str = "section",
        tag_filter: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve using BM25"""
        return self.bm25_index.search(
            query=query,
            top_k=top_k,
            doc_type=doc_type,
            tag_filter=tag_filter if tag_filter else None
        )
    
    def dense_retrieve(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        doc_type: str = "section",
        tag_filter: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve using vector similarity"""
        return self.dense_index.search(
            query=query,
            top_k=top_k,
            doc_type=doc_type,
            tag_filter=tag_filter if tag_filter else None
        )
    
    # =========================================================================
    # Stage 5: Hybrid Fusion
    # =========================================================================
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalize scores to [0, 1]"""
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
    
    def hybrid_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        query: str = "",
        dense_weight: float = DENSE_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        title_boost: float = TITLE_BOOST
    ) -> List[RetrievalResult]:
        """
        Fuse BM25 and dense results using weighted combination
        
        Normalization + weighted sum approach with title boosting
        """
        # Collect all unique documents
        all_docs = {}
        query_lower = query.lower()
        
        # Process BM25 results
        bm25_scores = [r.get('bm25_score', 0) for r in bm25_results]
        normalized_bm25 = self.normalize_scores(bm25_scores)
        
        for i, result in enumerate(bm25_results):
            doc_id = result['doc_id']
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    'data': result,
                    'bm25_score': result.get('bm25_score', 0),
                    'bm25_norm': normalized_bm25[i],
                    'dense_score': 0,
                    'dense_norm': 0
                }
            else:
                all_docs[doc_id]['bm25_score'] = result.get('bm25_score', 0)
                all_docs[doc_id]['bm25_norm'] = normalized_bm25[i]
        
        # Process dense results
        dense_scores = [r.get('dense_score', 0) for r in dense_results]
        normalized_dense = self.normalize_scores(dense_scores)
        
        for i, result in enumerate(dense_results):
            doc_id = result['doc_id']
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    'data': result,
                    'bm25_score': 0,
                    'bm25_norm': 0,
                    'dense_score': result.get('dense_score', 0),
                    'dense_norm': normalized_dense[i]
                }
            else:
                all_docs[doc_id]['dense_score'] = result.get('dense_score', 0)
                all_docs[doc_id]['dense_norm'] = normalized_dense[i]
        
        # Calculate hybrid scores with title boosting
        results = []
        for doc_id, doc_info in all_docs.items():
            data = doc_info['data']
            
            # Base hybrid score
            hybrid_score = (
                dense_weight * doc_info['dense_norm'] +
                bm25_weight * doc_info['bm25_norm']
            )
            
            # Title boost: if query terms appear in guide title, boost the score
            guide_title = (data.get('guide_title') or '').lower()
            if query_lower and guide_title:
                # Check for keyword overlap
                query_words = set(query_lower.split())
                title_words = set(guide_title.split())
                overlap = query_words & title_words
                if len(overlap) >= 2:  # At least 2 words match
                    hybrid_score += title_boost
                elif len(overlap) == 1 and len(query_words) <= 3:
                    hybrid_score += title_boost * 0.5
            
            results.append(RetrievalResult(
                doc_id=doc_id,
                doc_type=data.get('doc_type', ''),
                text=data.get('text', ''),
                guide_id=data.get('guide_id', ''),
                guide_title=data.get('guide_title', ''),
                section_title=data.get('section_title'),
                filename=data.get('filename', ''),
                relative_path=data.get('relative_path', ''),
                tags=data.get('tags', []),
                bm25_score=doc_info['bm25_score'],
                dense_score=doc_info['dense_score'],
                hybrid_score=hybrid_score,
                final_score=hybrid_score,
                summary=data.get('summary', ''),
                business_process=data.get('business_process', ''),
                date=data.get('date', ''),
                s3_key=data.get('s3_key', '')
            ))
        
        # Sort by hybrid score
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return results
    
    # =========================================================================
    # Stage 6: Reranking
    # =========================================================================
    
    def rerank_biencoder(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = RERANK_TOP_K
    ) -> List[RetrievalResult]:
        """
        Rerank results using bi-encoder (fast).
        
        Uses the same embedding model to compute query-document similarity.
        Much faster than cross-encoder but still effective for reranking.
        """
        if not results:
            return []
        
        import numpy as np
        
        model = get_biencoder_reranker()
        
        # Get query embedding with E5 prefix
        query_embedding = model.encode(
            E5_QUERY_PREFIX + query, 
            normalize_embeddings=True
        )
        
        # Get document embeddings with passage prefix
        doc_texts = [f"passage: {r.text}" for r in results[:top_k * 2]]
        doc_embeddings = model.encode(
            doc_texts, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Calculate cosine similarity (embeddings are normalized, so dot product = cosine)
        scores = np.dot(doc_embeddings, query_embedding)
        
        # Update scores
        for i, score in enumerate(scores):
            if i < len(results):
                results[i].rerank_score = float(score)
                results[i].final_score = float(score)
        
        # Re-sort by rerank score
        results[:len(scores)] = sorted(
            results[:len(scores)],
            key=lambda x: x.rerank_score,
            reverse=True
        )
        
        return results
    
    def rerank_crossencoder(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = RERANK_TOP_K
    ) -> List[RetrievalResult]:
        """
        Rerank top results using cross-encoder (slow but accurate).
        
        This is more expensive but significantly improves precision.
        """
        if not results:
            return []
        
        reranker = get_cross_encoder_reranker()
        
        # Prepare pairs for cross-encoder
        pairs = [(query, r.text) for r in results[:top_k * 2]]
        
        # Get reranking scores
        scores = reranker.predict(pairs)
        
        # Update scores
        for i, score in enumerate(scores):
            if i < len(results):
                results[i].rerank_score = float(score)
                results[i].final_score = float(score)
        
        # Re-sort by rerank score
        results[:len(scores)] = sorted(
            results[:len(scores)],
            key=lambda x: x.rerank_score,
            reverse=True
        )
        
        return results
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = RERANK_TOP_K,
        use_biencoder: bool = USE_BIENCODER_RERANKER
    ) -> List[RetrievalResult]:
        """
        Rerank results using either bi-encoder (fast) or cross-encoder (accurate).
        
        Args:
            query: Search query
            results: Results to rerank
            top_k: Number of top results to return
            use_biencoder: If True, use fast bi-encoder; else use cross-encoder
        """
        if use_biencoder:
            return self.rerank_biencoder(query, results, top_k)
        else:
            return self.rerank_crossencoder(query, results, top_k)
    
    # =========================================================================
    # Main Search Method
    # =========================================================================
    
    def search(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        doc_type: str = "section",
        use_filters: bool = True,
        use_reranking: bool = None
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Execute full retrieval pipeline
        
        Args:
            query: User query
            top_k: Number of final results
            doc_type: Document granularity ('guide', 'section', 'step')
            use_filters: Whether to extract and apply metadata filters
            use_reranking: Override default reranking setting
        
        Returns:
            Tuple of (results, timing_info)
        """
        timing = {}
        start_total = time.time()
        
        # Stage 1: Preprocess
        t0 = time.time()
        clean_query = self.preprocess_query(query)
        timing['preprocess_ms'] = (time.time() - t0) * 1000
        
        # Stage 2: Extract filters
        t0 = time.time()
        tag_filter = []
        if use_filters:
            tag_filter, _ = self.extract_filters(clean_query)
        timing['filter_extract_ms'] = (time.time() - t0) * 1000
        
        # Stage 3: BM25 retrieval
        t0 = time.time()
        bm25_results = self.sparse_retrieve(
            clean_query,
            top_k=BM25_TOP_K,
            doc_type=doc_type,
            tag_filter=tag_filter if tag_filter else None
        )
        timing['bm25_ms'] = (time.time() - t0) * 1000
        
        # Stage 4: Dense retrieval (if available)
        dense_results = []
        if getattr(self, '_dense_available', False):
            t0 = time.time()
            dense_results = self.dense_retrieve(
                clean_query,
                top_k=DENSE_TOP_K,
                doc_type=doc_type,
                tag_filter=tag_filter if tag_filter else None
            )
            timing['dense_ms'] = (time.time() - t0) * 1000
        else:
            timing['dense_ms'] = 0
        
        # Stage 5: Hybrid fusion (or BM25-only if dense unavailable)
        t0 = time.time()
        if dense_results:
            fused_results = self.hybrid_fusion(bm25_results, dense_results, query=clean_query)
        else:
            # BM25-only mode - convert to RetrievalResult format
            fused_results = []
            for r in bm25_results:
                fused_results.append(RetrievalResult(
                    doc_id=r['doc_id'],
                    doc_type=r.get('doc_type', ''),
                    text=r.get('text', ''),
                    guide_id=r.get('guide_id', ''),
                    guide_title=r.get('guide_title', ''),
                    section_title=r.get('section_title'),
                    filename=r.get('filename', ''),
                    relative_path=r.get('relative_path', ''),
                    tags=r.get('tags', []),
                    bm25_score=r.get('bm25_score', 0),
                    dense_score=0,
                    hybrid_score=r.get('bm25_score', 0),
                    final_score=r.get('bm25_score', 0),
                    summary=r.get('summary', ''),
                    business_process=r.get('business_process', ''),
                    date=r.get('date', ''),
                    s3_key=r.get('s3_key', '')
                ))
        timing['fusion_ms'] = (time.time() - t0) * 1000
        
        # Stage 6: Optional reranking
        do_rerank = use_reranking if use_reranking is not None else self.enable_reranking
        if do_rerank and fused_results:
            t0 = time.time()
            fused_results = self.rerank(clean_query, fused_results, top_k=top_k)
            timing['rerank_ms'] = (time.time() - t0) * 1000
        
        timing['total_ms'] = (time.time() - start_total) * 1000
        
        # Add debug info
        timing['bm25_candidates'] = len(bm25_results)
        timing['dense_candidates'] = len(dense_results)
        timing['fused_candidates'] = len(fused_results)
        timing['filters_applied'] = tag_filter
        timing['dense_available'] = getattr(self, '_dense_available', False)
        
        return fused_results[:top_k], timing
    
    def search_with_urls(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        doc_type: str = "section",
        use_filters: bool = True,
        use_reranking: bool = None
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Execute full retrieval pipeline and generate presigned URLs for documents.
        
        Same as search() but also populates the 'url' field with presigned S3 URLs.
        
        Args:
            query: User query
            top_k: Number of final results
            doc_type: Document granularity ('guide', 'section', 'step')
            use_filters: Whether to extract and apply metadata filters
            use_reranking: Override default reranking setting
        
        Returns:
            Tuple of (results with URLs, timing_info)
        """
        # Get search results
        results, timing = self.search(
            query=query,
            top_k=top_k,
            doc_type=doc_type,
            use_filters=use_filters,
            use_reranking=use_reranking
        )
        
        # Generate URLs for results
        t0 = time.time()
        try:
            from S3_Storage.s3_url_generator import generate_presigned_url
            for result in results:
                if result.s3_key:
                    result.url = generate_presigned_url(result.s3_key) or ""
        except ImportError as e:
            print(f"⚠ URL generation not available: {e}")
        except Exception as e:
            print(f"⚠ URL generation failed: {e}")
        timing['url_generation_ms'] = (time.time() - t0) * 1000
        
        return results, timing

    def quick_search(
        self,
        query: str,
        top_k: int = 5,
        doc_type: str = "section"
    ) -> List[RetrievalResult]:
        """
        Fast search without reranking (for <100ms latency)
        """
        results, _ = self.search(
            query=query,
            top_k=top_k,
            doc_type=doc_type,
            use_reranking=False
        )
        return results


def format_result(result: RetrievalResult, idx: int) -> str:
    """Format a single result for display"""
    lines = [
        f"\n{'='*60}",
        f"Result #{idx}",
        f"{'='*60}",
        f"Guide:    {result.guide_title}",
        f"Section:  {result.section_title or 'N/A'}",
        f"File:     {result.filename}",
        f"Path:     {result.relative_path}",
        f"S3 Key:   {result.s3_key or 'N/A'}",
        f"URL:      {result.url[:80] + '...' if result.url and len(result.url) > 80 else result.url or 'N/A'}",
        f"Tags:     {', '.join(result.tags)}",
        f"",
        f"Scores:",
        f"  • BM25:   {result.bm25_score:.4f}",
        f"  • Dense:  {result.dense_score:.4f}",
        f"  • Hybrid: {result.hybrid_score:.4f}",
        f"  • Final:  {result.final_score:.4f}",
    ]
    return "\n".join(lines)


def demo_search():
    """Demonstrate the search pipeline"""
    print("=" * 60)
    print("Step 4: Query Pipeline Demo")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "TVA 2%",
        "comment faire une facture détaillée FADET",
        "réactiver un abonné après suspension",
        "retour ressource remboursement",
        "recharge par bon de commande",
    ]
    
    with QueryPipeline(enable_reranking=False) as pipeline:
        print("\n→ Pipeline initialized")
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print("=" * 60)
            
            results, timing = pipeline.search(query, top_k=3)
            
            # Print timing
            print(f"\n⏱ Timing:")
            print(f"  • Preprocess:  {timing['preprocess_ms']:.1f}ms")
            print(f"  • Filters:     {timing['filter_extract_ms']:.1f}ms")
            print(f"  • BM25:        {timing['bm25_ms']:.1f}ms")
            print(f"  • Dense:       {timing['dense_ms']:.1f}ms")
            print(f"  • Fusion:      {timing['fusion_ms']:.1f}ms")
            print(f"  • TOTAL:       {timing['total_ms']:.1f}ms")
            
            if timing.get('filters_applied'):
                print(f"\n🏷 Filters applied: {timing['filters_applied']}")
            
            # Print results
            print(f"\n📄 Top {len(results)} Results:")
            for i, result in enumerate(results, 1):
                print(f"\n  {i}. [{result.final_score:.3f}] {result.guide_title}")
                if result.section_title:
                    print(f"     Section: {result.section_title}")
                print(f"     File: {result.filename}")


def main():
    """Run query pipeline demo"""
    demo_search()


if __name__ == "__main__":
    main()
