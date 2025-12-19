"""
Pipeline Principal - Intégration de toutes les étapes
Architecture complète pour atteindre Recall@1 ≈ 85%
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from .passage_generator import PassageGenerator, Passage
from .normalizer import Normalizer, QueryNormalizer, parse_price, parse_speed
from .intent_classifier import IntentClassifier, Intent, IntentResult
from .entity_detector import EntityDetector, EntityFilter, EntityDetectionResult
from .hybrid_ranker import HybridRanker, ScoredPassage
from .signature_booster import SignatureBooster
from .cross_encoder_reranker import CrossEncoderReranker, get_reranker


@dataclass
class PipelineConfig:
    """Configuration du pipeline."""
    # Passage generation
    passages_file: str = "passages.json"
    
    # Retrieval
    use_dense_retrieval: bool = True
    dense_model: str = "intfloat/multilingual-e5-small"
    
    # Reranking
    use_cross_encoder: bool = True
    cross_encoder_model: str = "nreimers/mmarco-mMiniLMv2-L12-H384-v1"
    
    # Pipeline parameters
    top_k_retrieval: int = 50  # Passages du hybrid ranker
    top_k_rerank: int = 30    # Passages pour le cross-encoder
    top_k_final: int = 10      # Documents finaux
    
    # Filtering
    apply_hard_entity_filter: bool = True
    
    # Boosting
    enable_numeric_boost: bool = True
    enable_signature_boost: bool = True


@dataclass
class SearchResult:
    """Résultat d'une recherche."""
    query: str
    intent: str
    intent_confidence: float
    detected_entities: List[str]
    entity_filter_applied: bool
    
    # Résultats
    top_documents: List[Dict[str, Any]]
    top_passages: List[Dict[str, Any]]
    
    # Métriques de debug
    total_passages_retrieved: int
    total_passages_after_filter: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


class RetrievalPipeline:
    """
    Pipeline complet de retrieval optimisé.
    
    Architecture:
    1. Query Processing (intent + entity detection)
    2. Dual Retrieval (BM25 + Dense)
    3. Hybrid Scoring (intent-based)
    4. Numeric Hard Boost
    5. Signature Boost
    6. Cross-Encoder Rerank
    7. Document Aggregation
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Composants
        self.passage_generator = PassageGenerator()
        self.query_normalizer = QueryNormalizer()
        self.intent_classifier = IntentClassifier()
        self.entity_detector = EntityDetector(
            apply_hard_filter_by_default=self.config.apply_hard_entity_filter
        )
        self.entity_filter = EntityFilter()
        
        self.hybrid_ranker: Optional[HybridRanker] = None
        self.signature_booster: Optional[SignatureBooster] = None
        self.cross_encoder: Optional[CrossEncoderReranker] = None
        
        # Données
        self.passages: List[Dict] = []
        self.documents: List[Dict] = []
        
        self._is_initialized = False
    
    def initialize(
        self, 
        documents_path: str = None,
        passages_path: str = None,
        documents: List[Dict] = None
    ):
        """
        Initialise le pipeline avec les données.
        
        Args:
            documents_path: Chemin vers le fichier JSON des documents originaux
            passages_path: Chemin vers le fichier JSON des passages (si déjà générés)
            documents: Documents directement en mémoire
        """
        print("🚀 Initialisation du pipeline de retrieval...")
        
        # 1. Charge ou génère les passages
        if passages_path and os.path.exists(passages_path):
            print(f"📂 Chargement des passages depuis {passages_path}")
            with open(passages_path, 'r', encoding='utf-8') as f:
                self.passages = json.load(f)
            print(f"   → {len(self.passages)} passages chargés")
        else:
            # Génère les passages depuis les documents
            if documents_path:
                print(f"📂 Chargement des documents depuis {documents_path}")
                with open(documents_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            elif documents:
                self.documents = documents
            else:
                raise ValueError("Fournir documents_path, passages_path, ou documents")
            
            print(f"⚙️ Génération des passages depuis {len(self.documents)} documents...")
            passages_obj = self.passage_generator.generate_all_passages(self.documents)
            self.passages = [p.to_dict() for p in passages_obj]
            print(f"   → {len(self.passages)} passages générés")
            
            # Sauvegarde les passages
            if passages_path:
                os.makedirs(os.path.dirname(passages_path) or '.', exist_ok=True)
                with open(passages_path, 'w', encoding='utf-8') as f:
                    json.dump(self.passages, f, ensure_ascii=False, indent=2)
                print(f"   → Passages sauvegardés dans {passages_path}")
        
        # 2. Initialise le Hybrid Ranker
        print("📊 Construction des index de recherche...")
        self.hybrid_ranker = HybridRanker(
            use_dense=self.config.use_dense_retrieval,
            dense_model=self.config.dense_model
        )
        self.hybrid_ranker.build_index(self.passages, text_field="text")
        
        # 3. Initialise le Signature Booster
        print("🔑 Construction des signatures...")
        self.signature_booster = SignatureBooster()
        self.signature_booster.build_signatures(self.passages)
        
        # 4. Initialise le Cross-Encoder (lazy loading)
        if self.config.use_cross_encoder:
            print("🧠 Préparation du cross-encoder...")
            self.cross_encoder = get_reranker(
                use_cross_encoder=True,
                model_name=self.config.cross_encoder_model
            )
        
        self._is_initialized = True
        print("✅ Pipeline initialisé avec succès!")
        
        # Affiche un résumé
        self._print_summary()
    
    def _print_summary(self):
        """Affiche un résumé du pipeline."""
        print("\n" + "="*50)
        print("📋 RÉSUMÉ DU PIPELINE")
        print("="*50)
        print(f"• Passages: {len(self.passages)}")
        print(f"• Dense retrieval: {'✅' if self.config.use_dense_retrieval else '❌'}")
        print(f"• Cross-encoder: {'✅' if self.config.use_cross_encoder else '❌'}")
        print(f"• Entity hard filter: {'✅' if self.config.apply_hard_entity_filter else '❌'}")
        print(f"• Numeric boost: {'✅' if self.config.enable_numeric_boost else '❌'}")
        print(f"• Signature boost: {'✅' if self.config.enable_signature_boost else '❌'}")
        
        # Stats sur les passages
        entity_counts = {}
        type_counts = {}
        for p in self.passages:
            entity = p.get("entity_code", "UNK")
            ptype = p.get("passage_type", "UNK")
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        
        print(f"\n• Passages par établissement:")
        for entity, count in sorted(entity_counts.items()):
            print(f"    {entity}: {count}")
        
        print(f"\n• Passages par type:")
        for ptype, count in sorted(type_counts.items()):
            print(f"    {ptype}: {count}")
        
        print("="*50 + "\n")
    
    def search(self, query: str, top_k: int = None) -> SearchResult:
        """
        Exécute une recherche complète.
        
        Args:
            query: Requête utilisateur
            top_k: Nombre de résultats (override config)
            
        Returns:
            SearchResult avec les documents et passages trouvés
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline non initialisé. Appelez initialize() d'abord.")
        
        top_k = top_k or self.config.top_k_final
        
        # === ÉTAPE 1: Query Processing ===
        
        # 1.1 Intent Classification
        intent_result = self.intent_classifier.classify(query)
        
        # 1.2 Entity Detection
        entity_result = self.entity_detector.detect(query)
        
        # 1.3 Query Normalization (prix, débit, etc.)
        normalized_query = self.query_normalizer.normalize(query)
        
        # === ÉTAPE 2 & 3: Hybrid Retrieval ===
        
        # Récupère les passages
        scored_passages = self.hybrid_ranker.search(
            query=normalized_query["expanded_query"],
            intent=intent_result.primary_intent,
            query_prices=normalized_query["prices"],
            query_speeds=normalized_query["speeds"],
            top_k=self.config.top_k_retrieval
        )
        
        total_retrieved = len(scored_passages)
        
        # === Entity Hard Filter ===
        if entity_result.apply_hard_filter and entity_result.detected_entities:
            target_entity = entity_result.detected_entities[0]
            scored_passages = [
                sp for sp in scored_passages
                if sp.passage.get("entity_code", "").upper() == target_entity.upper()
            ]
        
        total_after_filter = len(scored_passages)
        
        # === ÉTAPE 4: Signature Boost ===
        if self.config.enable_signature_boost and self.signature_booster:
            entity_filter = entity_result.detected_entities[0] if entity_result.detected_entities else None
            self.signature_booster.apply_boost_to_passages(
                scored_passages, 
                query,
                entity_code_filter=entity_filter
            )
        
        # === ÉTAPE 5 & 6: Cross-Encoder Rerank + Document Aggregation ===
        if self.config.use_cross_encoder and self.cross_encoder:
            final_docs = self.cross_encoder.rerank_with_aggregation(
                query=query,
                scored_passages=scored_passages,
                text_field="text",
                top_k_passages=self.config.top_k_rerank,
                top_k_docs=top_k
            )
        else:
            # Fallback: agrégation simple sans cross-encoder
            final_docs = self._simple_aggregation(scored_passages, top_k)
        
        # Prépare les passages finaux
        top_passages = [
            {
                "id": sp.passage.get("id"),
                "doc_id": sp.passage.get("doc_id"),
                "text": sp.passage.get("text"),
                "entity_code": sp.passage.get("entity_code"),
                "score": sp.final_score,
                "bm25_score": sp.bm25_score,
                "dense_score": sp.dense_score,
                "numeric_boost": sp.numeric_boost,
                "signature_boost": sp.signature_boost,
            }
            for sp in scored_passages[:20]
        ]
        
        return SearchResult(
            query=query,
            intent=intent_result.primary_intent.value,
            intent_confidence=intent_result.confidence,
            detected_entities=entity_result.detected_entities,
            entity_filter_applied=entity_result.apply_hard_filter,
            top_documents=final_docs,
            top_passages=top_passages,
            total_passages_retrieved=total_retrieved,
            total_passages_after_filter=total_after_filter
        )
    
    def _simple_aggregation(
        self, 
        scored_passages: List[ScoredPassage], 
        top_k: int
    ) -> List[Dict]:
        """Agrégation simple sans cross-encoder."""
        doc_scores = {}
        doc_passages = {}
        
        for sp in scored_passages:
            doc_id = sp.passage.get("doc_id", sp.passage.get("id"))
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = []
                doc_passages[doc_id] = []
            
            doc_scores[doc_id].append(sp.final_score)
            doc_passages[doc_id].append(sp.passage)
        
        # Score = max + mean(top3)
        results = []
        for doc_id, scores in doc_scores.items():
            max_score = max(scores)
            top3_mean = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))
            
            results.append({
                "doc_id": doc_id,
                "aggregated_score": max_score + 0.5 * top3_mean,
                "hybrid_score_max": max_score,
                "hybrid_score_mean_top3": top3_mean,
                "cross_encoder_score": 0,
                "best_passage": doc_passages[doc_id][0],
                "all_passages": doc_passages[doc_id],
            })
        
        results.sort(key=lambda x: x["aggregated_score"], reverse=True)
        return results[:top_k]
    
    def search_batch(
        self, 
        queries: List[str], 
        top_k: int = None
    ) -> List[SearchResult]:
        """
        Exécute plusieurs recherches.
        """
        return [self.search(q, top_k) for q in queries]
    
    def explain_search(self, query: str) -> Dict:
        """
        Exécute une recherche avec explication détaillée.
        Utile pour le debug et l'évaluation.
        """
        result = self.search(query)
        
        intent_details = self.intent_classifier.classify_with_explanation(query)
        entity_details = self.entity_detector.detect(query)
        normalized_query = self.query_normalizer.normalize(query)
        
        return {
            "query": query,
            "result": result.to_dict(),
            "intent_analysis": intent_details,
            "entity_analysis": {
                "detected": entity_details.detected_entities,
                "explicit": entity_details.is_explicit,
                "hard_filter": entity_details.apply_hard_filter,
                "confidence": entity_details.confidence,
                "patterns": entity_details.matched_patterns,
            },
            "query_normalization": normalized_query,
            "signature_matches": (
                self.signature_booster.find_matching_signatures(query)
                if self.signature_booster else []
            ),
        }
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Récupère un document par son ID."""
        for passage in self.passages:
            if passage.get("doc_id") == doc_id:
                return passage
        return None


# Factory function pour créer un pipeline configuré
def create_pipeline(
    documents_path: str,
    config: PipelineConfig = None,
    passages_cache_path: str = None
) -> RetrievalPipeline:
    """
    Crée et initialise un pipeline de retrieval.
    
    Args:
        documents_path: Chemin vers les documents JSON
        config: Configuration optionnelle
        passages_cache_path: Chemin pour sauvegarder/charger les passages
        
    Returns:
        Pipeline initialisé
    """
    config = config or PipelineConfig()
    
    if passages_cache_path is None:
        # Génère un chemin de cache par défaut
        base_dir = os.path.dirname(documents_path)
        passages_cache_path = os.path.join(base_dir, "passages_cache.json")
    
    pipeline = RetrievalPipeline(config)
    pipeline.initialize(
        documents_path=documents_path,
        passages_path=passages_cache_path
    )
    
    return pipeline


# Tests
if __name__ == "__main__":
    print("=== Test Pipeline ===\n")
    
    # Test avec des données mock
    mock_documents = [
        {
            "filename": "conv_P.docx",
            "establishment": "L'établissement P",
            "beneficiaries": "Cadres supérieurs et retraités",
            "internet_offers_table": [
                {
                    "category_segment": "Cadres Supérieurs",
                    "offer_type": "Fibre",
                    "speed": "1.5 Gbps",
                    "price": "Gratuit",
                    "benefits": "1 accès gratuit"
                },
                {
                    "category_segment": "Retraités",
                    "offer_type": "Fibre",
                    "speed": "1.5 Gbps",
                    "price": "1 100 DA",
                    "benefits": "Tarif réduit"
                }
            ],
            "required_documents_new": [
                "Attestation de travail",
                "Pièce d'identité"
            ],
            "notes": ["Un seul changement d'ONT autorisé"]
        }
    ]
    
    # Initialise avec config minimale (sans dense ni cross-encoder pour le test)
    config = PipelineConfig(
        use_dense_retrieval=False,
        use_cross_encoder=False
    )
    
    pipeline = RetrievalPipeline(config)
    pipeline.initialize(documents=mock_documents)
    
    # Test de recherche
    test_queries = [
        "Prix de la fibre pour les retraités établissement P",
        "Documents requis pour l'établissement P",
        "Offre gratuite cadres supérieurs",
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        result = pipeline.search(query, top_k=3)
        
        print(f"Intent: {result.intent} (conf: {result.intent_confidence:.2f})")
        print(f"Entities: {result.detected_entities}")
        print(f"Filter applied: {result.entity_filter_applied}")
        print(f"Passages: {result.total_passages_retrieved} → {result.total_passages_after_filter}")
        
        print("\nTop documents:")
        for i, doc in enumerate(result.top_documents[:3]):
            print(f"  {i+1}. {doc['doc_id']} (score: {doc['aggregated_score']:.3f})")
            if doc.get('best_passage'):
                print(f"     → {doc['best_passage'].get('text', '')[:60]}...")
