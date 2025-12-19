# 3-Layer Retrieval Pipeline for Algérie Télécom Documents

A sophisticated multi-layer retrieval system designed to find the most relevant documents from Algérie Télécom's corpus of offers, policies, and memos.

## Overview

This pipeline implements a 3-layer retrieval architecture:

1. **Rule-Based Routing Layer**: Fast metadata-based routing using document tags (doc_type, product_family, technology, customer_segment, etc.)
2. **Sparse Retrieval Layer (BM25)**: Lexical matching using BM25 algorithm
3. **Dense Retrieval Layer (Embeddings)**: Semantic matching using multilingual sentence embeddings

The pipeline automatically selects the best layer based on confidence scores, optimizing for both speed and accuracy.

## Features

- ✅ **Multilingual Support**: Handles French and Arabic queries seamlessly
- ✅ **Smart Layer Selection**: Automatically chooses the most confident layer
- ✅ **Fast Rule-Based Routing**: Instant results for pattern-matched queries
- ✅ **Semantic Understanding**: Falls back to embeddings when lexical matching is insufficient
- ✅ **Rich Context Generation**: Builds structured LLM context from retrieved documents
- ✅ **Interactive CLI**: Easy testing and exploration

## Installation

### 1. Clone or navigate to the project directory

```bash
cd /path/to/ARAB
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first time you run the pipeline, sentence-transformers will download the multilingual model (~400MB). This is a one-time download.

## Quick Start

### Interactive Mode

```bash
python main_pipeline.py --data-dir individual_docs
```

This launches an interactive session where you can enter queries and see results in real-time.

### Single Query Mode

```bash
python main_pipeline.py --data-dir individual_docs --query "Je cherche une offre 4G LTE sans engagement"
```

### Get Top 3 Results

```bash
python main_pipeline.py --data-dir individual_docs --query "offre fibre pour école" --top-k 3
```

### Disable Dense Layer (Faster Startup)

```bash
python main_pipeline.py --data-dir individual_docs --no-dense
```

### JSON Output

```bash
python main_pipeline.py --data-dir individual_docs --query "timbre fiscal" --json
```

## Usage Examples

### Example 1: Tax Policy Query (French)

```bash
python main_pipeline.py --data-dir individual_docs --query "Comment s'applique le timbre fiscal sur les abonnements ?"
```

**Expected**: Rule-based layer routes to tax_policy document

### Example 2: Locataire Query (French)

```bash
python main_pipeline.py --data-dir individual_docs --query "Je suis locataire et je veux internet fibre"
```

**Expected**: Rule-based layer identifies locataire segment

### Example 3: Arabic Query

```bash
python main_pipeline.py --data-dir individual_docs --query "أنا مستأجر وأبحث عن عرض إنترنت ثابت"
```

**Expected**: Multilingual handling finds locataire offer

### Example 4: 4G LTE Query

```bash
python main_pipeline.py --data-dir individual_docs --query "Idoom 4G LTE sans engagement beaucoup de data"
```

**Expected**: Routes to 4G LTE no-commitment offer

## Project Structure

```
.
├── pipeline/
│   ├── __init__.py           # Package initialization
│   ├── loader.py             # Document loading and normalization
│   ├── rule_router.py        # Layer 1: Rule-based routing
│   ├── sparse_index.py       # Layer 2: BM25 sparse retrieval
│   ├── dense_index.py        # Layer 3: Dense embeddings
│   └── pipeline.py           # Main orchestrator
├── tests/
│   ├── __init__.py
│   ├── test_pipeline_basic.py  # End-to-end tests
│   └── data/                   # Test documents
├── individual_docs/            # Your JSON documents
├── main_pipeline.py            # CLI entry point
├── requirements.txt
└── README.md
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test class
pytest tests/test_pipeline_basic.py::TestRuleBasedRouting -v

# Run with coverage
pytest tests/ --cov=pipeline --cov-report=html
```

## How It Works

### Layer 1: Rule-Based Routing

The rule router builds an index of routing tokens from each document's metadata:
- `doc_type` (offer, tax_policy, memo, etc.)
- `product_family` (idoom_4g_lte, tax_stamp, etc.)
- `technology` (fibre_optique, 4g_lte, adsl, etc.)
- `customer_segment` (residential, locataire, schools, etc.)
- `keywords` (extracted terms)

When a query arrives, it's matched against these tokens with weighted scoring:
- Exact doc_type match: +5 points
- Product family match: +5 points
- Customer segment match: +3 points
- Special patterns (e.g., "école" → schools): +10 points

**High confidence** decision if:
- Only one strong match exists, OR
- Top score is 2x better than second, OR
- Absolute score > 15

### Layer 2: Sparse Retrieval (BM25)

If rule layer isn't confident, BM25 performs lexical matching:
- Uses `search_text` field + boosted keywords
- Can be restricted to rule layer candidates for efficiency
- BM25Okapi algorithm (better than TF-IDF for search)

**Confidence estimation**:
- High if top score > 10 and gap > 2x second score
- Medium if top score > 5 and gap > 1.5x
- Low otherwise

### Layer 3: Dense Retrieval (Embeddings)

Final fallback for semantic matching:
- Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Encodes `dense_text_primary` field into 384-dim vectors
- Cosine similarity search
- Handles multilingual and paraphrased queries

**When used**:
- Rule layer has no strong candidates
- BM25 confidence is low (< 0.6)

### LLM Context Generation

For the best matching document, the pipeline builds a structured context containing:
- Document metadata (ID, type, title, date)
- Product characteristics (family, technology, segments)
- Offer details (name, description, conditions, benefits, pricing)
- Policy summaries (for policy docs)
- Top 3 FAQ items (most relevant)
- Contact information

This context is designed to be fed directly to an LLM chatbot.

## Configuration

### Confidence Thresholds

Edit `pipeline/pipeline.py` to adjust thresholds:

```python
pipeline = RetrievalPipeline(
    docs=docs,
    rule_confidence_threshold=0.7,   # Default: 0.7
    sparse_confidence_threshold=0.6,  # Default: 0.6
    use_dense=True
)
```

### Keyword Boosting

Edit `pipeline/sparse_index.py`:

```python
sparse = SparseIndex(docs, keyword_boost=3)  # Default: 3x repetition
```

## Performance Considerations

- **Startup time**:
  - Without dense layer: ~1-2 seconds
  - With dense layer: ~10-30 seconds (model loading + embedding computation)

- **Query time**:
  - Rule layer: <10ms
  - Sparse layer: <50ms
  - Dense layer: <100ms

- **Memory**:
  - ~100MB for docs + BM25 index
  - ~500MB additional for dense embeddings model

## Troubleshooting

### Issue: "No module named 'rank_bm25'"

```bash
pip install rank-bm25
```

### Issue: Slow first run with dense layer

This is normal - the model is being downloaded. Subsequent runs will be fast.

### Issue: Out of memory

Disable dense layer:

```bash
python main_pipeline.py --data-dir individual_docs --no-dense
```

## Future Enhancements

Possible improvements:
- [ ] Add query expansion and synonym handling
- [ ] Implement query rewriting for common patterns
- [ ] Add structured query support (JSON format)
- [ ] Cache embeddings to disk for faster restarts
- [ ] Add multilingual BM25 with better Arabic tokenization
- [ ] Implement hybrid scoring (combine layers)
- [ ] Add feedback loop for continuous improvement

## License

Internal use - Algérie Télécom

## Contact

For questions about this pipeline, contact the development team.
