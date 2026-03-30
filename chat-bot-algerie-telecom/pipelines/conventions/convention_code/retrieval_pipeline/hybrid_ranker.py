"""
Hybrid Ranker - ÉTAPES 2, 3.1, 3.2
Dual Retrieval (BM25 + Dense) avec scoring hybride basé sur l'intent.
Inclut le Numeric Hard Boost (CRITIQUE).
"""

import re
import math
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .intent_classifier import Intent, get_hybrid_weights
from .normalizer import parse_price, parse_speed, Normalizer

# Shared RAG enhancements (RRF + parallel search)
try:
    _rag_root = str(Path(__file__).resolve().parents[4])
    if _rag_root not in sys.path:
        sys.path.insert(0, _rag_root)
    from rag_enhancements import rrf_fusion
except ImportError:
    # Fallback: plain weighted average (original behaviour preserved)
    rrf_fusion = None


def normalize_accents(text: str) -> str:
    """Remove accents from text for accent-insensitive matching."""
    nfkd_form = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfkd_form if not unicodedata.combining(c))


@dataclass
class ScoredPassage:
    """Passage avec son score hybride."""
    passage: Dict[str, Any]
    bm25_score: float
    dense_score: float
    hybrid_score: float
    numeric_boost: float
    signature_boost: float
    final_score: float


class BM25Index:
    """
    Index BM25 simple pour le retrieval sparse.
    Optimisé pour les passages factuels courts.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        self.documents: List[Dict] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.term_doc_freq: Dict[str, int] = defaultdict(int)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.total_docs: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize le texte en termes."""
        # Normalize accents first for accent-insensitive matching
        text = normalize_accents(text)
        # Lowercase + suppression ponctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        # Filtre les mots vides courts
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def build_index(self, documents: List[Dict], text_field: str = "text"):
        """
        Construit l'index BM25.
        
        Args:
            documents: Liste de documents (dicts)
            text_field: Champ contenant le texte à indexer
        """
        self.documents = documents
        self.total_docs = len(documents)
        
        # Tokenize chaque document
        for doc_id, doc in enumerate(documents):
            text = doc.get(text_field, "") or ""
            
            # Ajoute aussi les champs structurés au texte de recherche
            search_text = text
            if doc.get("establishment"):
                search_text += " " + doc["establishment"]
            if doc.get("beneficiary"):
                search_text += " " + doc["beneficiary"]
            if doc.get("offer_type"):
                search_text += " " + doc["offer_type"]
            
            tokens = self._tokenize(search_text)
            self.doc_lengths.append(len(tokens))
            
            # Compte les termes
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            
            # Met à jour l'index inversé
            for term, freq in term_freq.items():
                self.inverted_index[term].append((doc_id, freq))
                
            # Met à jour les doc frequencies
            for term in set(tokens):
                self.term_doc_freq[term] += 1
        
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1
    
    def _idf(self, term: str) -> float:
        """Calcule l'IDF d'un terme."""
        df = self.term_doc_freq.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Recherche BM25.
        
        Args:
            query: Requête textuelle
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste de (doc_id, score) triée par score décroissant
        """
        query_tokens = self._tokenize(query)
        
        scores = defaultdict(float)
        
        for term in query_tokens:
            idf = self._idf(term)
            
            for doc_id, tf in self.inverted_index.get(term, []):
                doc_length = self.doc_lengths[doc_id]
                
                # Formule BM25
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                
                scores[doc_id] += idf * numerator / denominator
        
        # Trie et retourne top_k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
    
    def search_with_boost(
        self, 
        query: str, 
        boost_fields: Dict[str, float] = None,
        top_k: int = 50
    ) -> List[Tuple[int, float]]:
        """
        Recherche BM25 avec boost sur certains champs.
        
        Args:
            query: Requête textuelle
            boost_fields: Dict de {field_name: boost_factor}
            top_k: Nombre de résultats
        """
        base_scores = dict(self.search(query, top_k=top_k * 2))
        
        if not boost_fields:
            return sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Applique les boosts
        for doc_id, score in base_scores.items():
            doc = self.documents[doc_id]
            
            for field, boost in boost_fields.items():
                field_value = str(doc.get(field, "")).lower()
                query_lower = query.lower()
                
                # Boost si le champ contient un terme de la query
                for term in query_lower.split():
                    if term in field_value:
                        base_scores[doc_id] *= boost
                        break
        
        return sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class DenseRetriever:
    """
    Retriever dense basé sur les embeddings.
    Supporte plusieurs backends (sentence-transformers, Qdrant, etc.)
    """
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self.model_name = model_name
        self.embeddings: Optional[List] = None
        self.documents: List[Dict] = []
        self._model = None
    
    def _load_model(self):
        """Charge le modèle d'embedding à la demande."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                print("⚠️ sentence-transformers non installé. Install: pip install sentence-transformers")
                self._model = None
        return self._model
    
    def build_index(self, documents: List[Dict], text_field: str = "text"):
        """
        Construit l'index dense en générant les embeddings.
        """
        self.documents = documents
        model = self._load_model()
        
        if model is None:
            print("⚠️ Dense retrieval désactivé (modèle non disponible)")
            self.embeddings = None
            return
        
        texts = [doc.get(text_field, "") or "" for doc in documents]
        
        print(f"📊 Génération des embeddings pour {len(texts)} passages...")
        self.embeddings = model.encode(texts, show_progress_bar=True)
        print("✅ Embeddings générés")
    
    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Recherche par similarité cosinus.
        """
        if self.embeddings is None or self._model is None:
            return []
        
        import numpy as np
        
        query_embedding = self._model.encode([query])[0]
        
        # Similarité cosinus
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def save_embeddings(self, filepath: str):
        """Sauvegarde les embeddings pour un chargement rapide."""
        if self.embeddings is not None:
            import numpy as np
            np.save(filepath, self.embeddings)
    
    def load_embeddings(self, filepath: str):
        """Charge des embeddings pré-calculés."""
        import numpy as np
        self.embeddings = np.load(filepath)


class HybridRanker:
    """
    Ranker hybride combinant BM25 et Dense retrieval.
    Applique des poids basés sur l'intent (ÉTAPE 3.1).
    Applique le Numeric Hard Boost (ÉTAPE 3.2).
    """
    
    def __init__(
        self,
        use_dense: bool = True,
        dense_model: str = "intfloat/multilingual-e5-small"
    ):
        self.bm25 = BM25Index()
        self.dense = DenseRetriever(dense_model) if use_dense else None
        self.documents: List[Dict] = []
        self.use_dense = use_dense
    
    def build_index(self, passages: List[Dict], text_field: str = "text"):
        """
        Construit les index BM25 et Dense.
        """
        self.documents = passages
        
        print("📚 Construction de l'index BM25...")
        self.bm25.build_index(passages, text_field)
        
        if self.use_dense and self.dense:
            print("🧠 Construction de l'index Dense...")
            self.dense.build_index(passages, text_field)
    
    def _normalize_scores(self, scores: List[Tuple[int, float]]) -> Dict[int, float]:
        """Normalise les scores entre 0 et 1."""
        if not scores:
            return {}
        
        max_score = max(s[1] for s in scores) or 1
        min_score = min(s[1] for s in scores)
        range_score = max_score - min_score or 1
        
        return {
            doc_id: (score - min_score) / range_score
            for doc_id, score in scores
        }
    
    def _apply_numeric_boost(
        self,
        doc_id: int,
        query_prices: List[int],
        query_speeds: List[float]
    ) -> float:
        """
        ÉTAPE 3.2 - Hard Numeric Boost (CRITIQUE)
        Multiplie le score si le prix/débit exact match.
        """
        if doc_id >= len(self.documents):
            return 1.0
        
        doc = self.documents[doc_id]
        boost = 1.0
        
        # Boost pour correspondance de prix
        doc_price = doc.get("price_value")
        if doc_price is not None and query_prices:
            for qp in query_prices:
                if doc_price == qp:
                    boost *= 2.0  # 🎯 +100% si prix exact
                    break
                elif abs(doc_price - qp) <= 100:  # Tolérance de 100 DA
                    boost *= 1.5
                    break
        
        # Boost pour correspondance de débit
        doc_speed = doc.get("speed_mbps")
        if doc_speed is not None and query_speeds:
            for qs in query_speeds:
                if doc_speed == qs:
                    boost *= 2.0  # 🎯 +100% si débit exact
                    break
                elif abs(doc_speed - qs) <= doc_speed * 0.1:  # Tolérance 10%
                    boost *= 1.5
                    break
        
        # Boost pour gratuit
        if doc.get("is_free") and any(p == 0 for p in query_prices):
            boost *= 1.5
        
        return boost
    
    def search(
        self,
        query: str,
        intent: Intent,
        query_prices: List[int] = None,
        query_speeds: List[float] = None,
        top_k: int = 30
    ) -> List[ScoredPassage]:
        """
        Hybrid search with RRF fusion and parallel BM25 + dense retrieval.

        Improvements over original:
        - BM25 and dense retrieval run **concurrently** (ThreadPoolExecutor)
        - Scores fused with **Reciprocal Rank Fusion** instead of weighted average
          (RRF is more robust: no score normalisation artefacts, no weight tuning)
        - Numeric Hard Boost preserved unchanged

        Falls back to original weighted-average if rag_enhancements unavailable.
        """
        query_prices = query_prices or []
        query_speeds = query_speeds or []

        dense_weight, sparse_weight = get_hybrid_weights(intent)
        dense_available = self.use_dense and self.dense and self.dense.embeddings is not None

        # ── 1. Parallel BM25 + Dense retrieval ──────────────────────────────
        fetch_k = top_k * 2

        if dense_available:
            with ThreadPoolExecutor(max_workers=2) as pool:
                bm25_future  = pool.submit(self.bm25.search,  query, fetch_k)
                dense_future = pool.submit(self.dense.search, query, fetch_k)
                bm25_results  = bm25_future.result()
                dense_results = dense_future.result()
        else:
            bm25_results  = self.bm25.search(query, fetch_k)
            dense_results = []

        # ── 2. Hybrid scoring: RRF (preferred) or weighted average (fallback) ─
        if rrf_fusion is not None and (bm25_results or dense_results):
            rrf_scores = rrf_fusion([bm25_results, dense_results])
            bm25_raw   = dict(bm25_results)
            dense_raw  = dict(dense_results)
            all_ids    = set(rrf_scores)

            scored_passages = []
            for doc_id in all_ids:
                b_score  = bm25_raw.get(doc_id, 0.0)
                d_score  = dense_raw.get(doc_id, 0.0)
                hybrid   = rrf_scores[doc_id]
                boost    = self._apply_numeric_boost(doc_id, query_prices, query_speeds)
                scored_passages.append(ScoredPassage(
                    passage=self.documents[doc_id],
                    bm25_score=b_score,
                    dense_score=d_score,
                    hybrid_score=hybrid,
                    numeric_boost=boost,
                    signature_boost=1.0,
                    final_score=hybrid * boost,
                ))
        else:
            # Original weighted-average path (unchanged)
            bm25_scores  = self._normalize_scores(bm25_results)
            dense_scores = self._normalize_scores(dense_results) if dense_results else {}
            all_ids      = set(bm25_scores) | set(dense_scores)

            scored_passages = []
            for doc_id in all_ids:
                b_score = bm25_scores.get(doc_id, 0.0)
                d_score = dense_scores.get(doc_id, 0.0)
                hybrid  = sparse_weight * b_score + dense_weight * d_score
                boost   = self._apply_numeric_boost(doc_id, query_prices, query_speeds)
                scored_passages.append(ScoredPassage(
                    passage=self.documents[doc_id],
                    bm25_score=b_score,
                    dense_score=d_score,
                    hybrid_score=hybrid,
                    numeric_boost=boost,
                    signature_boost=1.0,
                    final_score=hybrid * boost,
                ))

        # ── 3. Sort and return top_k ─────────────────────────────────────────
        scored_passages.sort(key=lambda x: x.final_score, reverse=True)
        return scored_passages[:top_k]
    
    def get_boost_fields_for_intent(self, intent: Intent) -> Dict[str, float]:
        """Retourne les champs à booster selon l'intent."""
        if intent == Intent.PRICE:
            return {"price_value": 2.0, "is_free": 1.5}
        elif intent == Intent.SPEED:
            return {"speed_mbps": 2.0, "offer_type": 1.5}
        elif intent == Intent.DOCUMENTS:
            return {"passage_type": 2.0}  # Boost les passages DOCUMENTS
        elif intent == Intent.BENEFICIARY:
            return {"beneficiary": 2.0, "category_segment": 1.5}
        else:
            return {}


# Tests
if __name__ == "__main__":
    # Créer des passages de test
    test_passages = [
        {
            "id": "1",
            "text": "[Etab=P][Type=Offer] Idoom Fibre 1.5 Gbps à 1100 DA",
            "entity_code": "P",
            "price_value": 1100,
            "speed_mbps": 1500,
            "is_free": False,
            "offer_type": "FIBRE"
        },
        {
            "id": "2",
            "text": "[Etab=P][Type=Offer] Idoom ADSL 20 Mbps gratuit pour cadres",
            "entity_code": "P",
            "price_value": 0,
            "speed_mbps": 20,
            "is_free": True,
            "offer_type": "ADSL"
        },
        {
            "id": "3",
            "text": "[Etab=V][Type=Offer] Idoom Fibre 300 Mbps à 1500 DA",
            "entity_code": "V",
            "price_value": 1500,
            "speed_mbps": 300,
            "is_free": False,
            "offer_type": "FIBRE"
        },
        {
            "id": "4",
            "text": "[Etab=P][Type=Documents] Attestation de travail requise",
            "entity_code": "P",
            "passage_type": "DOCUMENTS"
        },
    ]
    
    print("=== Test Hybrid Ranker ===\n")
    
    ranker = HybridRanker(use_dense=False)  # Sans dense pour le test
    ranker.build_index(test_passages)
    
    # Test avec intent PRICE
    print("Query: Offre à 1100 DA")
    results = ranker.search(
        "Offre à 1100 DA",
        intent=Intent.PRICE,
        query_prices=[1100]
    )
    
    for r in results[:3]:
        print(f"  Score: {r.final_score:.3f} | Boost: {r.numeric_boost} | {r.passage['text'][:50]}")
    
    print("\nQuery: Débit 1.5 Gbps")
    results = ranker.search(
        "Débit 1.5 Gbps",
        intent=Intent.SPEED,
        query_speeds=[1500]
    )
    
    for r in results[:3]:
        print(f"  Score: {r.final_score:.3f} | Boost: {r.numeric_boost} | {r.passage['text'][:50]}")
