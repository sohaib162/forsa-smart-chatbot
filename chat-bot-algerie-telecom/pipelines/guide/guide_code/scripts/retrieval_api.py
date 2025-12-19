"""
Retrieval API - User-Friendly Interface
========================================
High-level API for searching NGBSS guides and retrieving documents.

Features:
- Simple search interface
- Document path resolution
- Multiple output formats
- Caching for performance

Usage:
    from scripts.retrieval_api import NGBSSRetriever
    
    retriever = NGBSSRetriever()
    results = retriever.search("TVA 2%")
    
    for r in results:
        print(r.guide_title, r.document_path)
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DOCUMENTS_BASE_PATH, JSON_FILE
from scripts.step4_query_pipeline import QueryPipeline, RetrievalResult

# Cache for guide data (loaded once)
_guides_cache: Dict[str, Dict] = None


def _load_guides_cache() -> Dict[str, Dict]:
    """Load and cache all guides from JSON file"""
    global _guides_cache
    if _guides_cache is None:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _guides_cache = {guide['id']: guide for guide in data.get('guides', [])}
    return _guides_cache


def get_guide_by_id(guide_id: str) -> Optional[Dict]:
    """Get complete guide data by ID"""
    guides = _load_guides_cache()
    return guides.get(guide_id)


@dataclass
class SearchResult:
    """User-friendly search result"""
    rank: int
    score: float
    
    # Document info
    guide_title: str
    section_title: Optional[str]
    summary: str
    tags: List[str]
    
    # File paths
    filename: str
    relative_path: str
    document_path: Optional[Path]  # Resolved absolute path
    
    # S3 storage
    s3_key: Optional[str] = None
    url: Optional[str] = None  # Presigned URL for document access
    
    # Identifiers
    guide_id: str = ""
    doc_id: str = ""
    doc_type: str = ""
    
    # Text preview
    text_preview: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        d['document_path'] = str(d['document_path']) if d['document_path'] else None
        return d
    
    def exists(self) -> bool:
        """Check if document file exists"""
        return self.document_path is not None and self.document_path.exists()


class NGBSSRetriever:
    """
    High-level retrieval interface for NGBSS guides
    
    Example:
        retriever = NGBSSRetriever(documents_root="/path/to/docs")
        results = retriever.search("comment facturer TVA 2%", top_k=5)
        
        for r in results:
            print(f"{r.rank}. {r.guide_title}")
            if r.exists():
                print(f"   Document: {r.document_path}")
    """
    
    def __init__(
        self,
        documents_root: Union[str, Path] = None,
        enable_reranking: bool = False,
        auto_connect: bool = True
    ):
        """
        Initialize the retriever
        
        Args:
            documents_root: Root directory where PDF/DOCX files are stored
            enable_reranking: Whether to use cross-encoder reranking (slower but more accurate)
            auto_connect: Automatically connect to indexes on init
        """
        self.documents_root = Path(documents_root) if documents_root else None
        self.pipeline = QueryPipeline(enable_reranking=enable_reranking)
        self._connected = False
        
        if auto_connect:
            self.connect()
    
    def connect(self):
        """Connect to search indexes"""
        self.pipeline.connect()
        self._connected = True
        return self
    
    def close(self):
        """Close connections"""
        self.pipeline.close()
        self._connected = False
    
    def __enter__(self):
        if not self._connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def set_documents_root(self, path: Union[str, Path]):
        """Set the documents root directory"""
        self.documents_root = Path(path)
    
    def _resolve_document_path(self, relative_path: str) -> Optional[Path]:
        """
        Resolve relative path to absolute document path
        
        Tries multiple strategies:
        1. Direct path from documents_root
        2. Just filename in documents_root
        3. Original relative path
        """
        if not self.documents_root:
            return None
        
        # Strategy 1: Use relative path from root
        # e.g., "data/Guide NGBSS/file.pdf" -> documents_root + "Guide NGBSS/file.pdf"
        parts = Path(relative_path).parts
        if len(parts) > 1:
            # Try stripping first directory (usually "data")
            subpath = Path(*parts[1:]) if parts[0] == 'data' else Path(relative_path)
            full_path = self.documents_root / subpath
            if full_path.exists():
                return full_path
        
        # Strategy 2: Just filename
        filename = Path(relative_path).name
        full_path = self.documents_root / filename
        if full_path.exists():
            return full_path
        
        # Strategy 3: Full relative path
        full_path = self.documents_root / relative_path
        if full_path.exists():
            return full_path
        
        # Return expected path even if doesn't exist
        return self.documents_root / filename
    
    def _to_search_result(
        self,
        result: RetrievalResult,
        rank: int
    ) -> SearchResult:
        """Convert internal result to user-friendly format"""
        # Resolve document path
        doc_path = self._resolve_document_path(result.relative_path)
        
        # Create preview (first 500 chars)
        text_preview = result.text[:500] + "..." if len(result.text) > 500 else result.text
        
        return SearchResult(
            rank=rank,
            score=round(result.final_score, 4),
            guide_title=result.guide_title,
            section_title=result.section_title,
            summary=result.summary,
            tags=result.tags,
            filename=result.filename,
            relative_path=result.relative_path,
            document_path=doc_path,
            s3_key=result.s3_key,
            url=result.url,
            guide_id=result.guide_id,
            doc_id=result.doc_id,
            doc_type=result.doc_type,
            text_preview=text_preview
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_type: str = "section",
        use_filters: bool = True
    ) -> List[SearchResult]:
        """
        Search for relevant guides/sections
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            doc_type: Granularity - 'guide', 'section', or 'step'
            use_filters: Extract and apply metadata filters from query
        
        Returns:
            List of SearchResult objects
        """
        if not self._connected:
            self.connect()
        
        results, timing = self.pipeline.search(
            query=query,
            top_k=top_k,
            doc_type=doc_type,
            use_filters=use_filters
        )
        
        return [
            self._to_search_result(r, i + 1)
            for i, r in enumerate(results)
        ]
    
    def search_with_timing(
        self,
        query: str,
        top_k: int = 5,
        doc_type: str = "section",
        use_filters: bool = True
    ) -> tuple:
        """
        Search with timing information
        
        Returns:
            Tuple of (results, timing_dict)
        """
        if not self._connected:
            self.connect()
        
        results, timing = self.pipeline.search(
            query=query,
            top_k=top_k,
            doc_type=doc_type,
            use_filters=use_filters
        )
        
        search_results = [
            self._to_search_result(r, i + 1)
            for i, r in enumerate(results)
        ]
        
        return search_results, timing
    
    def search_with_urls(
        self,
        query: str,
        top_k: int = 5,
        doc_type: str = "section",
        use_filters: bool = True
    ) -> tuple:
        """
        Search with presigned URL generation for S3 documents.
        
        Returns:
            Tuple of (results with URLs, timing_dict)
        """
        if not self._connected:
            self.connect()
        
        results, timing = self.pipeline.search_with_urls(
            query=query,
            top_k=top_k,
            doc_type=doc_type,
            use_filters=use_filters
        )
        
        search_results = [
            self._to_search_result(r, i + 1)
            for i, r in enumerate(results)
        ]
        
        return search_results, timing

    def search_for_rag(
        self,
        query: str,
        top_k: int = 3,
        doc_type: str = "section",
        use_filters: bool = True,
        include_url: bool = True
    ) -> Dict[str, Any]:
        """
        Search and return complete guide data for RAG/LLM consumption.
        
        Returns a structured JSON-ready dict with:
        - query: Original query
        - retrieved_documents: List of complete guide metadata with sections
        - timing: Performance metrics
        
        Perfect for sending to DeepSeek or other LLMs for answer generation.
        
        Args:
            query: User's question
            top_k: Number of results to retrieve
            doc_type: Document granularity ('guide', 'section', 'step')
            use_filters: Apply metadata filtering
            include_url: Generate presigned S3 URLs
        
        Returns:
            Dict ready for JSON serialization and LLM consumption
        """
        if not self._connected:
            self.connect()
        
        # Get search results with URLs if requested
        if include_url:
            results, timing = self.pipeline.search_with_urls(
                query=query,
                top_k=top_k,
                doc_type=doc_type,
                use_filters=use_filters
            )
        else:
            results, timing = self.pipeline.search(
                query=query,
                top_k=top_k,
                doc_type=doc_type,
                use_filters=use_filters
            )
        
        # Build complete response with full guide metadata
        retrieved_documents = []
        seen_guides = set()  # Avoid duplicate guides
        
        for rank, result in enumerate(results, 1):
            guide_id = result.guide_id
            
            # Get complete guide data from cache
            guide_data = get_guide_by_id(guide_id)
            
            if guide_data and guide_id not in seen_guides:
                seen_guides.add(guide_id)
                
                # Build document entry with full metadata
                doc_entry = {
                    "rank": rank,
                    "relevance_score": round(result.final_score, 4),
                    "matched_section": result.section_title,
                    "matched_text": result.text[:1000],  # Preview of matched text
                    
                    # Complete guide metadata
                    "guide": {
                        "id": guide_data.get("id"),
                        "title": guide_data.get("title"),
                        "system": guide_data.get("system"),
                        "business_process": guide_data.get("business_process"),
                        "summary": guide_data.get("summary"),
                        "tags": guide_data.get("tags", []),
                        "prerequisites": guide_data.get("prerequisites", []),
                        "language": guide_data.get("language", []),
                        "date": guide_data.get("date"),
                        "filename": guide_data.get("filename"),
                        "s3_key": guide_data.get("s3_key"),
                        "url": result.url if include_url else None,
                        
                        # Full sections with steps
                        "sections": guide_data.get("sections", [])
                    }
                }
                retrieved_documents.append(doc_entry)
        
        # Build final response
        response = {
            "query": query,
            "retrieval_info": {
                "total_results": len(retrieved_documents),
                "retrieval_time_ms": round(timing['total_ms'], 2),
                "filters_applied": timing.get('filters_applied', []),
                "doc_type": doc_type
            },
            "retrieved_documents": retrieved_documents
        }
        
        return response
    
    def search_for_rag_json(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> str:
        """
        Same as search_for_rag but returns JSON string.
        
        Convenience method for direct API responses.
        """
        result = self.search_for_rag(query, top_k, **kwargs)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def search_guides(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search at guide level (broader results)"""
        return self.search(query, top_k=top_k, doc_type="guide")
    
    def search_sections(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search at section level (recommended)"""
        return self.search(query, top_k=top_k, doc_type="section")
    
    def search_steps(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search at step level (most precise)"""
        return self.search(query, top_k=top_k, doc_type="step")
    
    def get_document_path(self, result: SearchResult) -> Optional[Path]:
        """Get the full path to the document file"""
        return result.document_path if result.exists() else None
    
    def to_json(self, results: List[SearchResult]) -> str:
        """Convert results to JSON string"""
        return json.dumps(
            [r.to_dict() for r in results],
            ensure_ascii=False,
            indent=2
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_search(query: str, top_k: int = 5) -> List[SearchResult]:
    """
    Quick one-off search (creates new connection each time)
    
    For repeated searches, use NGBSSRetriever instance instead.
    """
    with NGBSSRetriever(enable_reranking=False) as retriever:
        return retriever.search(query, top_k=top_k)


def search_and_print(query: str, top_k: int = 5):
    """Search and print formatted results"""
    with NGBSSRetriever(enable_reranking=False) as retriever:
        results, timing = retriever.search_with_timing(query, top_k=top_k)
        
        print(f"\n🔍 Query: {query}")
        print(f"⏱ Time: {timing['total_ms']:.0f}ms")
        print(f"📊 Results: {len(results)}")
        
        for r in results:
            print(f"\n{'─'*50}")
            print(f"#{r.rank} [{r.score:.3f}] {r.guide_title}")
            if r.section_title:
                print(f"    Section: {r.section_title}")
            print(f"    File: {r.filename}")
            print(f"    Tags: {', '.join(r.tags[:5])}")
            if r.document_path:
                exists = "✓" if r.exists() else "✗"
                print(f"    Path: {r.document_path} {exists}")


# =============================================================================
# Demo
# =============================================================================

def main():
    """Demo the retrieval API"""
    print("=" * 60)
    print("NGBSS Retrieval API Demo")
    print("=" * 60)
    
    queries = [
        "TVA 2% facture complémentaire",
        "retour ressource remboursement",
        "facture détaillée FADET",
    ]
    
    for query in queries:
        search_and_print(query, top_k=3)
        print()


if __name__ == "__main__":
    main()
