# NGBSS Guide Retrieval Pipeline

A fast, accurate hybrid retrieval system for NGBSS guides with PDF/DOCX document serving.

## Features

- ⚡ **Fast**: <250ms average search latency with reranking, <1s for all queries
- 🎯 **Accurate**: 90.5% Recall@1, 98.4% Recall@5 with hybrid BM25 + semantic search
- 📄 **Document Serving**: Retrieve PDF/Word documents corresponding to guides
- 🔧 **Modular**: Each pipeline step can run independently
- 🌐 **Multilingual**: French/Arabic support via multilingual E5 embeddings

## Performance Results

| Metric | Score |
|--------|-------|
| **Recall@1** | 90.5% |
| **Recall@3** | 96.8% |
| **Recall@5** | 98.4% |
| **MRR** | 0.940 |
| **Avg Time** | 112ms |
| **Max Time** | 223ms |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Query Pipeline                             │
├──────────────────────────────────────────────────────────────────┤
│  Query → Preprocess → Extract Filters → [BM25] + [Dense] →       │
│          → Hybrid Fusion → (Rerank) → Results + Documents         │
└──────────────────────────────────────────────────────────────────┘
```

### Components

1. **Data Preparation** (`step1_data_preparation.py`)
   - Extracts documents at 3 granularity levels: Guide, Section, Step
   - Creates structured JSON with metadata

2. **BM25 Index** (`step2_sparse_index.py`)
   - SQLite FTS5-based keyword search
   - Excellent for acronyms (FADET, TVA, IDOOM)
   - Ultra-fast: ~5ms search time

3. **Dense Index** (`step3_dense_index.py`)
   - Qdrant vector database
   - **intfloat/multilingual-e5-small** embeddings with query/passage prefixes
   - Semantic understanding: ~30-50ms search time

4. **Query Pipeline** (`step4_query_pipeline.py`)
   - Hybrid fusion of BM25 + dense results with title boosting
   - Metadata filtering for faster, more relevant results
   - Cross-encoder reranking for high precision

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the Pipeline

```bash
# Run all steps
python run_pipeline.py --all

# Or run individually
python run_pipeline.py --step 1  # Data preparation
python run_pipeline.py --step 2  # BM25 indexing
python run_pipeline.py --step 3  # Dense indexing
python run_pipeline.py --step 4  # Demo
```

### 3. Search

```bash
# Quick search
python run_pipeline.py --search "TVA 2%"

# With document path
python run_pipeline.py --search "facture" --docs-root /path/to/documents

# Interactive mode
python run_pipeline.py --interactive
```

### 4. Use in Code

```python
from scripts.retrieval_api import NGBSSRetriever

# Initialize (set documents_root to your PDF/DOCX directory)
retriever = NGBSSRetriever(documents_root="/path/to/Guide NGBSS")

# Search
results = retriever.search("comment facturer TVA 2%", top_k=5)

for r in results:
    print(f"{r.rank}. {r.guide_title}")
    print(f"   Section: {r.section_title}")
    print(f"   File: {r.filename}")
    if r.exists():
        print(f"   Path: {r.document_path}")
```

### 5. REST API

```bash
# Start server
uvicorn scripts.api_server:app --port 8000

# Search
curl "http://localhost:8000/search?q=TVA%202%25&top_k=5"

# Download document
curl "http://localhost:8000/document/guide.pdf" -o guide.pdf
```

### 6. Docker Deployment (RAG API)

The Docker API accepts batch queries and returns complete JSON for LLM consumption (DeepSeek).

```bash
# Build the image
docker build -t ngbss-retrieval .

# Run the container
docker run -d --name ngbss-api -p 8000:8000 \
  -v $(pwd)/indexes:/app/indexes \
  -v $(pwd)/data:/app/data:ro \
  ngbss-retrieval

# Or use docker-compose
docker-compose up -d
```

**Input JSON format:**
```json
{
  "equipe": "IA_Team",
  "question": {
    "categorie_01": {
      "1": "Comment enregistrer un paiement effectué à la poste ?",
      "2": "Quelles sont les étapes pour TVA 2% ?"
    }
  },
  "top_k": 3
}
```

**Query the API:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d @test_query.json
```

**Response includes complete guide metadata for LLM:**
```json
{
  "equipe": "IA_Team",
  "total_questions": 2,
  "results": [
    {
      "categorie": "categorie_01",
      "question_id": "1",
      "query": "...",
      "retrieval_info": {...},
      "retrieved_documents": [
        {
          "rank": 1,
          "relevance_score": 1.0,
          "matched_section": "...",
          "guide": {
            "title": "...",
            "sections": [...],
            "prerequisites": [...],
            ...
          }
        }
      ]
    }
  ]
}
```

## Configuration

Edit `config/settings.py`:

```python
# Embedding model (multilingual E5 - requires query/passage prefixes)
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION = 384

# E5 instruction prefixes (required for optimal performance)
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# Search parameters
BM25_TOP_K = 30
DENSE_TOP_K = 30
FINAL_TOP_K = 10

# Hybrid fusion weights (E5 embeddings are strong)
DENSE_WEIGHT = 0.80
BM25_WEIGHT = 0.20
TITLE_BOOST = 0.20  # Boost for title matches

# Reranking (cross-encoder for best accuracy)
ENABLE_RERANKING = True
USE_BIENCODER_RERANKER = False  # False = use cross-encoder (more accurate)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

## Document Storage

When you decide where to store your PDF/Word documents:

1. **Option A: Local filesystem**
   ```python
   retriever = NGBSSRetriever(documents_root="/path/to/Guide NGBSS")
   ```

2. **Option B: Cloud storage (S3, GCS, etc.)**
   - Modify `retrieval_api.py` to generate signed URLs
   - Or use a CDN with document paths

3. **Option C: Database BLOBs**
   - Store documents in PostgreSQL/MongoDB
   - Modify API to stream from database

The `relative_path` in search results maps to your storage location.

## Performance Tuning

### For fastest latency (<50ms):
- Use bi-encoder reranker: `USE_BIENCODER_RERANKER=True`
- Or disable reranking: `ENABLE_RERANKING=False`
- Use section-level search (default)
- Apply metadata filters: `use_filters=True`

### For best accuracy (90%+ Recall@1):
- Use cross-encoder reranking: `USE_BIENCODER_RERANKER=False`
- Keep E5 embedding model with proper prefixes
- Use title boosting: `TITLE_BOOST=0.20`
- Dense weight higher: `DENSE_WEIGHT=0.80`

### Warm-up for production:
- Run a dummy query at startup to load models
- This avoids cold-start latency on first real query

### Memory optimization:
- Use Qdrant with on-disk storage
- Reduce embedding dimension with quantization
- Cache frequently accessed documents

## Project Structure

```
ngbss_retrieval/
├── config/
│   └── settings.py          # Configuration
├── data/
│   └── Guide_NGBSS.json     # Input data
├── indexes/
│   ├── bm25_index.db        # SQLite FTS5 index
│   ├── qdrant_db/           # Vector index
│   └── processed_documents.json
├── scripts/
│   ├── step1_data_preparation.py
│   ├── step2_sparse_index.py
│   ├── step3_dense_index.py
│   ├── step4_query_pipeline.py
│   ├── retrieval_api.py     # High-level API
│   └── api_server.py        # FastAPI server
├── run_pipeline.py          # Main runner
├── evaluate.py              # Evaluation script
├── test_questions.json      # Test dataset
└── requirements.txt
```

## Evaluation

Run the evaluation suite:

```bash
# Run evaluation
python evaluate.py

# Save results to JSON
python evaluate.py --save

# Show failed queries
python evaluate.py --show-failed
```

## Extending

### Add new document types
Edit `step1_data_preparation.py` to handle new JSON structures.

### Add new filters
Edit `TAG_KEYWORDS` in `config/settings.py`:
```python
TAG_KEYWORDS = {
    "my_keyword": ["tag1", "tag2"],
    ...
}
```

### Custom scoring
Modify `hybrid_fusion()` in `step4_query_pipeline.py`.

## License

MIT License
