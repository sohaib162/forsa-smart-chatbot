"""
Cross-Encoder Reranker - ÉTAPE 5
Le composant clé pour passer de Recall@5 à Recall@1.
Rerank les top-30 passages avec un cross-encoder multilingue.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RerankResult:
    """Résultat du reranking."""
    passage: Dict[str, Any]
    cross_encoder_score: float
    original_rank: int
    final_rank: int


class CrossEncoderReranker:
    """
    Cross-Encoder pour le reranking final.
    Utilise un modèle cross-encoder multilingue pour comparer query-passage.
    """
    
    # Modèles recommandés (du plus léger au plus lourd)
    RECOMMENDED_MODELS = [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Rapide, anglais
        "nreimers/mmarco-mMiniLMv2-L12-H384-v1",  # Multilingue, léger
        "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Plus précis, anglais
        "amberoad/bert-multilingual-passage-reranking-msmarco",  # Multilingue
    ]
    
    def __init__(
        self, 
        model_name: str = "nreimers/mmarco-mMiniLMv2-L12-H384-v1",
        device: str = None,
        batch_size: int = 16
    ):
        """
        Args:
            model_name: Nom du modèle HuggingFace
            device: 'cpu', 'cuda', ou None pour auto-détection
            batch_size: Taille des batches pour l'inférence
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._is_available = None
    
    def _load_model(self):
        """Charge le modèle cross-encoder à la demande."""
        if self._model is not None:
            return self._model
            
        try:
            from sentence_transformers import CrossEncoder
            
            print(f"📥 Chargement du cross-encoder: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device
            )
            self._is_available = True
            print("✅ Cross-encoder chargé")
            
        except ImportError:
            print("⚠️ sentence-transformers non installé.")
            print("   Install: pip install sentence-transformers")
            self._model = None
            self._is_available = False
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du cross-encoder: {e}")
            self._model = None
            self._is_available = False
        
        return self._model
    
    def is_available(self) -> bool:
        """Vérifie si le cross-encoder est disponible."""
        if self._is_available is None:
            self._load_model()
        return self._is_available
    
    def rerank(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        text_field: str = "text",
        top_k: int = 20
    ) -> List[RerankResult]:
        """
        Rerank les passages avec le cross-encoder.
        
        Args:
            query: Requête utilisateur
            passages: Liste de passages à reranker
            text_field: Champ contenant le texte du passage
            top_k: Nombre de passages à retourner après reranking
            
        Returns:
            Liste de RerankResult triée par score cross-encoder
        """
        model = self._load_model()
        
        if model is None:
            # Fallback: retourne les passages dans l'ordre original
            return [
                RerankResult(
                    passage=p,
                    cross_encoder_score=0.0,
                    original_rank=i,
                    final_rank=i
                )
                for i, p in enumerate(passages[:top_k])
            ]
        
        # Prépare les paires query-passage
        pairs = []
        for passage in passages:
            text = passage.get(text_field, "") or ""
            # Tronque si nécessaire
            if len(text) > 450:
                text = text[:450]
            pairs.append([query, text])
        
        # Score avec le cross-encoder
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        # Crée les résultats avec les scores
        results = []
        for i, (passage, score) in enumerate(zip(passages, scores)):
            results.append(RerankResult(
                passage=passage,
                cross_encoder_score=float(score),
                original_rank=i,
                final_rank=-1  # Sera mis à jour après le tri
            ))
        
        # Trie par score cross-encoder
        results.sort(key=lambda x: x.cross_encoder_score, reverse=True)
        
        # Met à jour les rangs finaux
        for i, result in enumerate(results):
            result.final_rank = i
        
        return results[:top_k]
    
    def rerank_with_aggregation(
        self,
        query: str,
        scored_passages: List[Any],  # ScoredPassage from hybrid_ranker
        text_field: str = "text",
        top_k_passages: int = 30,
        top_k_docs: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank et agrège par document (ÉTAPE 5 + 6).
        
        Pipeline:
        1. Take top-30 passages
        2. Regroupe par document
        3. Rerank top-20 documents avec cross-encoder
        4. Retourne le top-K
        
        Args:
            query: Requête utilisateur
            scored_passages: Liste de ScoredPassage du hybrid ranker
            text_field: Champ contenant le texte
            top_k_passages: Nombre de passages pour le reranking
            top_k_docs: Nombre de documents finaux
            
        Returns:
            Liste de documents agrégés avec scores
        """
        # 1. Prend les top passages
        passages = [sp.passage for sp in scored_passages[:top_k_passages]]
        passage_scores = {
            sp.passage.get("id", i): sp.final_score 
            for i, sp in enumerate(scored_passages[:top_k_passages])
        }
        
        # 2. Regroupe par document
        doc_passages: Dict[str, List[Dict]] = {}
        doc_hybrid_scores: Dict[str, List[float]] = {}
        
        for passage in passages:
            doc_id = passage.get("doc_id", passage.get("id", "unknown"))
            
            if doc_id not in doc_passages:
                doc_passages[doc_id] = []
                doc_hybrid_scores[doc_id] = []
            
            doc_passages[doc_id].append(passage)
            passage_id = passage.get("id", "")
            doc_hybrid_scores[doc_id].append(passage_scores.get(passage_id, 0))
        
        # 3. Pour chaque document, sélectionne le meilleur passage pour le reranking
        doc_best_passages = {}
        for doc_id, doc_passages_list in doc_passages.items():
            # Prend le passage avec le meilleur score hybrid
            scores = doc_hybrid_scores[doc_id]
            best_idx = scores.index(max(scores))
            doc_best_passages[doc_id] = doc_passages_list[best_idx]
        
        # 4. Rerank avec cross-encoder
        doc_ids = list(doc_best_passages.keys())
        best_passages = list(doc_best_passages.values())
        
        reranked = self.rerank(query, best_passages, text_field, top_k=len(best_passages))
        
        # 5. Agrège les scores (ÉTAPE 6)
        # Score document = max(passage_scores) + mean(top3_passages)
        final_results = []
        
        for rr in reranked:
            passage = rr.passage
            doc_id = passage.get("doc_id", passage.get("id", "unknown"))
            
            # Calcule le score agrégé
            hybrid_scores = doc_hybrid_scores.get(doc_id, [0])
            max_hybrid = max(hybrid_scores)
            
            top3_hybrid = sorted(hybrid_scores, reverse=True)[:3]
            mean_top3 = sum(top3_hybrid) / len(top3_hybrid) if top3_hybrid else 0
            
            # Score final combiné
            aggregated_score = (
                0.5 * rr.cross_encoder_score +  # Cross-encoder weight
                0.3 * max_hybrid +               # Best passage score
                0.2 * mean_top3                  # Top-3 mean
            )
            
            final_results.append({
                "doc_id": doc_id,
                "cross_encoder_score": rr.cross_encoder_score,
                "hybrid_score_max": max_hybrid,
                "hybrid_score_mean_top3": mean_top3,
                "aggregated_score": aggregated_score,
                "best_passage": passage,
                "all_passages": doc_passages.get(doc_id, []),
                "original_rank": rr.original_rank,
                "final_rank": rr.final_rank,
            })
        
        # Trie par score agrégé
        final_results.sort(key=lambda x: x["aggregated_score"], reverse=True)
        
        # Met à jour les rangs
        for i, result in enumerate(final_results):
            result["final_rank"] = i
        
        return final_results[:top_k_docs]


class FallbackReranker:
    """
    Reranker de fallback quand le cross-encoder n'est pas disponible.
    Utilise une combinaison de règles heuristiques.
    """
    
    def __init__(self):
        pass
    
    def rerank(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        text_field: str = "text",
        top_k: int = 20
    ) -> List[RerankResult]:
        """
        Rerank simple basé sur des heuristiques.
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        scored = []
        for i, passage in enumerate(passages):
            text = passage.get(text_field, "").lower()
            
            # Score basé sur le nombre de termes communs
            common_terms = sum(1 for term in query_terms if term in text)
            term_coverage = common_terms / len(query_terms) if query_terms else 0
            
            # Bonus pour les correspondances exactes
            exact_match_bonus = 0.2 if query_lower in text else 0
            
            # Bonus pour la longueur appropriée
            length_bonus = 0.1 if 50 < len(text) < 300 else 0
            
            score = term_coverage + exact_match_bonus + length_bonus
            
            scored.append(RerankResult(
                passage=passage,
                cross_encoder_score=score,
                original_rank=i,
                final_rank=-1
            ))
        
        # Trie par score
        scored.sort(key=lambda x: x.cross_encoder_score, reverse=True)
        
        for i, result in enumerate(scored):
            result.final_rank = i
        
        return scored[:top_k]


def get_reranker(
    use_cross_encoder: bool = True,
    model_name: str = "nreimers/mmarco-mMiniLMv2-L12-H384-v1"
) -> CrossEncoderReranker:
    """
    Factory function pour obtenir le reranker approprié.
    """
    if use_cross_encoder:
        reranker = CrossEncoderReranker(model_name=model_name)
        if reranker.is_available():
            return reranker
    
    print("⚠️ Cross-encoder non disponible, utilisation du fallback")
    return FallbackReranker()


# Tests
if __name__ == "__main__":
    test_passages = [
        {"id": "1", "doc_id": "doc1", "text": "Idoom Fibre 1.5 Gbps à 1100 DA pour les retraités"},
        {"id": "2", "doc_id": "doc1", "text": "Idoom ADSL 20 Mbps gratuit pour cadres supérieurs"},
        {"id": "3", "doc_id": "doc2", "text": "Offre téléphonie Forfait 500 DA"},
        {"id": "4", "doc_id": "doc2", "text": "Documents requis: attestation de travail"},
        {"id": "5", "doc_id": "doc3", "text": "ONT Wi-Fi 6 gratuit pour retraités"},
    ]
    
    print("=== Test Cross-Encoder Reranker ===\n")
    
    reranker = get_reranker(use_cross_encoder=True)
    
    query = "Tarif fibre pour les retraités"
    print(f"Query: {query}\n")
    
    results = reranker.rerank(query, test_passages)
    
    print("Résultats reranked:")
    for r in results:
        print(f"  Rank {r.final_rank} (was {r.original_rank}): "
              f"score={r.cross_encoder_score:.3f} - {r.passage['text'][:50]}")
