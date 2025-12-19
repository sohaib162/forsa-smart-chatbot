"""
Configuration settings for NGBSS Retrieval Pipeline
"""
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INDEXES_DIR = BASE_DIR / "indexes"

# Input JSON file
JSON_FILE = DATA_DIR / "Guide_NGBSS.json"

# Document storage base path (update this when you decide where to store PDFs/DOCXs)
DOCUMENTS_BASE_PATH = Path("data/Guide NGBSS")  # Relative path from your storage root

# Index files
SQLITE_DB = INDEXES_DIR / "bm25_index.db"
QDRANT_PATH = INDEXES_DIR / "qdrant_db"
PROCESSED_DOCS_FILE = INDEXES_DIR / "processed_documents.json"

# =============================================================================
# EMBEDDING MODEL
# =============================================================================
# Multilingual E5 model - better semantic understanding for French/Arabic
# Note: E5 models require "query: " prefix for queries and "passage: " for documents
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# Alternative: "intfloat/multilingual-e5-base" (better quality, slower)
# Previous: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

EMBEDDING_DIMENSION = 384  # For multilingual-e5-small

# E5 model instruction prefixes (required for optimal performance)
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# =============================================================================
# RERANKER MODEL
# =============================================================================
# Use cross-encoder for better accuracy (slower but more precise ranking)
USE_BIENCODER_RERANKER = False  # Cross-encoder gives better recall@1

# Cross-encoder reranker - better quality ranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Alternative for multilingual: "nreimers/mmarco-mMiniLMv2-L12-H384-v1"

# =============================================================================
# RETRIEVAL SETTINGS
# =============================================================================
# Number of candidates to retrieve from each index
BM25_TOP_K = 30  # Reduced for faster retrieval
DENSE_TOP_K = 30  # Reduced for faster retrieval

# Final results
FINAL_TOP_K = 10

# Hybrid scoring weights - E5 embeddings are strong, give them more weight
DENSE_WEIGHT = 0.80  # Strong semantic matching with E5
BM25_WEIGHT = 0.20   # BM25 for exact keyword matches

# Guide title boost - boost results where query matches guide title
TITLE_BOOST = 0.20  # Increased for better guide-level matching

# Reranking
RERANK_TOP_K = 10  # Reduced for faster reranking
ENABLE_RERANKING = True

# =============================================================================
# METADATA FILTER KEYWORDS
# =============================================================================
# Keywords that trigger metadata filtering for faster/more accurate search
TAG_KEYWORDS = {
    # Service types
    "4g": ["4G", "LTE", "4G LTE", "réactivation"],
    "lte": ["4G", "LTE", "4G LTE", "réactivation"],
    "fibre": ["Fibre", "FTTH", "fibre optique", "IDOOM Fibre"],
    "ftth": ["Fibre", "FTTH", "fibre optique", "IDOOM Fibre"],
    "pstn": ["PSTN", "ligne fixe", "téléphone fixe", "enquête PSTN"],
    "idoom": ["IDOOM", "idoom adsl", "internet", "IDOOM Fibre"],
    "adsl": ["ADSL", "idoom adsl"],
    "ont": ["FTTH", "IDOOM Fibre", "VOIP"],
    
    # Business processes
    "tva": ["TVA 2%", "TVA", "taxe"],
    "tva 2%": ["TVA 2%"],
    "fadet": ["FADET", "facture détaillée"],
    "facture détaillée": ["FADET", "facture détaillée"],
    "cdr": ["FADET", "facture détaillée"],  # Call Detail Record
    "appels": ["FADET", "facture détaillée"],
    "facture": ["facture", "facturation", "encaissement"],
    "duplicata": ["duplicata", "copie facture", "réimprimer"],
    "réimprimer": ["duplicata", "copie facture"],
    "remboursement": ["remboursement", "retour ressource"],
    "retour": ["retour ressource", "remboursement"],
    "annuler": ["retour ressource", "annulation"],
    "suspension": ["suspension", "désactivation"],
    "réactivation": ["réactivation", "activation", "4G LTE"],
    "résiliation": ["résiliation", "clôture"],
    "migration": ["migration", "changement offre"],
    "recharge": ["recharge", "crédit", "bon de commande"],
    "paiement": ["paiement", "encaissement"],
    "poste": ["bureau de poste", "encaissement externe"],
    "externe": ["bureau de poste", "encaissement externe"],
    
    # Inventory & Resources
    "inventaire": ["inventaire", "stock", "ACTEL"],
    "stock": ["inventaire", "stock", "transfert"],
    "actel": ["ACTEL", "inventaire", "distribution"],
    "cartes": ["inventaire", "ventes par lot", "recharge"],
    "lot": ["ventes par lot", "inventaire"],
    "distribuer": ["inventaire", "transfert", "ACTEL"],
    "ressource": ["retour ressource", "inventaire"],
    "modem": ["retour ressource", "vente"],
    
    # Payment arrangements
    "échéancier": ["arrangement", "AOD", "échéancier"],
    "aod": ["arrangement", "AOD", "échéancier"],
    "arrangement": ["arrangement", "AOD", "échéancier"],
    
    # Ligne temporaire
    "temporaire": ["ligne temporaire", "élections"],
    "élections": ["ligne temporaire"],
    
    # Orders
    "ordre": ["gestion d'ordre", "enquête PSTN"],
    "enquête": ["enquête PSTN", "gestion d'ordre"],
}

# =============================================================================
# S3 / MinIO SETTINGS
# =============================================================================
# Read from environment variables (for Docker) with defaults for local dev
import os
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "localhost:9000")
S3_EXTERNAL_ENDPOINT = os.getenv("S3_EXTERNAL_ENDPOINT", S3_ENDPOINT)  # For generating public URLs
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_SECURE = os.getenv("S3_SECURE", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "forsa-documents")

# =============================================================================
# QDRANT SETTINGS
# =============================================================================
QDRANT_COLLECTION_NAME = "ngbss_guides"
