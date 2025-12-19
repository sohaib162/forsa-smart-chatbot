"""
Step 3: Dense Indexing (Vector Embeddings)
==========================================
Creates a vector index using sentence-transformers + Qdrant for semantic search.

Benefits:
- Semantic understanding ("réactiver abonné après suspension" matches "réactivation")
- Multilingual support (French/Arabic)
- Fast approximate nearest neighbor search (HNSW)
- Rich metadata filtering

Run independently: python -m scripts.step3_dense_index
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    QDRANT_PATH, PROCESSED_DOCS_FILE, EMBEDDING_MODEL,
    EMBEDDING_DIMENSION, QDRANT_COLLECTION_NAME, DENSE_TOP_K,
    E5_QUERY_PREFIX, E5_PASSAGE_PREFIX
)

# Lazy imports for optional dependencies
_embedding_model = None
_qdrant_client = None


def get_embedding_model():
    """Lazy load embedding model"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"→ Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_qdrant_client(path: Path = QDRANT_PATH):
    """Get or create Qdrant client"""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        path.parent.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(path))
    return _qdrant_client


class DenseIndex:
    """Qdrant-based vector index for semantic search"""
    
    def __init__(
        self, 
        collection_name: str = QDRANT_COLLECTION_NAME,
        qdrant_path: Path = QDRANT_PATH
    ):
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        self.client = None
        self.model = None
    
    def connect(self):
        """Initialize connections"""
        self.client = get_qdrant_client(self.qdrant_path)
        self.model = get_embedding_model()
        return self
    
    def __enter__(self):
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # Qdrant client doesn't need explicit close
    
    def create_collection(self, recreate: bool = True):
        """Create or recreate the vector collection"""
        from qdrant_client.models import Distance, VectorParams
        
        if recreate:
            # Delete if exists
            try:
                self.client.delete_collection(self.collection_name)
                print(f"→ Deleted existing collection: {self.collection_name}")
            except Exception:
                pass
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
        print(f"✓ Created collection: {self.collection_name}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, is_query: bool = False) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        For E5 models, we add instruction prefixes:
        - "query: " for search queries
        - "passage: " for documents being indexed
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            is_query: If True, use query prefix; otherwise use passage prefix
        """
        # Add E5 instruction prefix based on whether it's a query or passage
        prefix = E5_QUERY_PREFIX if is_query else E5_PASSAGE_PREFIX
        prefixed_texts = [prefix + text for text in texts]
        
        embeddings = []
        for i in range(0, len(prefixed_texts), batch_size):
            batch = prefixed_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            embeddings.extend(batch_embeddings.tolist())
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query with E5 query prefix"""
        prefixed_query = E5_QUERY_PREFIX + query
        embedding = self.model.encode(prefixed_query, normalize_embeddings=True)
        return embedding.tolist()
    
    def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Index documents with embeddings"""
        from qdrant_client.models import PointStruct
        
        total = len(documents)
        print(f"→ Indexing {total} documents...")
        
        points = []
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings in batches
        print("  → Generating embeddings...")
        start_time = time.time()
        embeddings = self.embed_texts(texts)
        embed_time = time.time() - start_time
        print(f"  ✓ Generated {len(embeddings)} embeddings in {embed_time:.2f}s")
        
        # Prepare points
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "doc_id": doc['doc_id'],
                    "doc_type": doc['doc_type'],
                    "text": doc['text'][:2000],  # Truncate for storage
                    "guide_id": doc.get('guide_id'),
                    "guide_title": doc.get('guide_title'),
                    "section_title": doc.get('section_title'),
                    "filename": doc.get('filename'),
                    "relative_path": doc.get('relative_path'),
                    "system": doc.get('system'),
                    "tags": doc.get('tags', []),
                    "date": doc.get('date'),
                    "summary": doc.get('summary'),
                    "business_process": doc.get('business_process'),
                    "s3_key": doc.get('s3_key'),
                }
            )
            points.append(point)
        
        # Upload in batches
        print("  → Uploading to Qdrant...")
        start_time = time.time()
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        upload_time = time.time() - start_time
        print(f"  ✓ Uploaded {len(points)} points in {upload_time:.2f}s")
    
    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        doc_type: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        guide_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity
        
        Args:
            query: Search query string
            top_k: Number of results to return
            doc_type: Filter by document type
            tag_filter: Filter by tags (documents must have ANY of these tags)
            guide_id: Filter by specific guide
        
        Returns:
            List of matching documents with similarity scores
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
        
        # Generate query embedding with E5 query prefix
        query_embedding = self.embed_query(query)
        
        # Build filters
        filter_conditions = []
        
        if doc_type:
            filter_conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
            )
        
        if tag_filter:
            # Match any of the tags
            filter_conditions.append(
                FieldCondition(key="tags", match=MatchAny(any=tag_filter))
            )
        
        if guide_id:
            filter_conditions.append(
                FieldCondition(key="guide_id", match=MatchValue(value=guide_id))
            )
        
        query_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Search - use query_points for newer versions, search for older versions
        try:
            # New API (qdrant-client >= 1.12)
            from qdrant_client.models import QueryRequest
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            ).points
        except (AttributeError, ImportError):
            # Legacy API (qdrant-client < 1.12)
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
        
        # Format results
        return [
            {
                **hit.payload,
                "dense_score": hit.score,
                "qdrant_id": hit.id
            }
            for hit in results
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        info = self.client.get_collection(self.collection_name)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.name
        }


def build_index():
    """Build dense vector index from processed documents"""
    print("=" * 60)
    print("Step 3: Dense Indexing (Vector Embeddings)")
    print("=" * 60)
    
    # Load processed documents
    if not PROCESSED_DOCS_FILE.exists():
        print(f"✗ Processed documents not found: {PROCESSED_DOCS_FILE}")
        print("  Please run step1_data_preparation first.")
        return None
    
    with open(PROCESSED_DOCS_FILE, 'r', encoding='utf-8') as f:
        all_docs = json.load(f)
    
    # Combine all documents for indexing
    # Primary focus on sections, but include all levels
    all_documents = []
    for doc_type in ['guides', 'sections', 'steps']:
        all_documents.extend(all_docs.get(doc_type, []))
    
    print(f"\n→ Total documents to index: {len(all_documents)}")
    
    # Create and populate index
    with DenseIndex() as index:
        print(f"→ Creating vector index at: {QDRANT_PATH}")
        index.create_collection(recreate=True)
        index.index_documents(all_documents)
        
        # Print stats
        stats = index.get_stats()
        print(f"\n✓ Dense Index Statistics:")
        for key, value in stats.items():
            print(f"  • {key}: {value}")
        
        # Test search
        print("\n→ Testing dense search: 'comment réactiver un abonné suspendu'")
        start_time = time.time()
        results = index.search(
            "comment réactiver un abonné suspendu",
            top_k=3,
            doc_type="section"
        )
        search_time = time.time() - start_time
        print(f"  Search time: {search_time*1000:.0f}ms")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['dense_score']:.3f}] {r['guide_title']} - {r.get('section_title', 'N/A')}")
    
    return QDRANT_PATH


def main():
    """Run dense indexing step"""
    return build_index()


if __name__ == "__main__":
    main()
