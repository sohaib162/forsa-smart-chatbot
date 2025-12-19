"""
Évaluation du Pipeline - Script pour mesurer Recall@K
Permet de valider les performances et d'ajuster les paramètres.
"""

import json
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from .pipeline import RetrievalPipeline, PipelineConfig, create_pipeline


@dataclass
class EvaluationSample:
    """Un échantillon de test avec query et document attendu."""
    query: str
    expected_doc_id: str
    expected_establishment: str = None
    category: str = None  # PRICE, SPEED, DOCUMENTS, etc.


@dataclass
class EvaluationResult:
    """Résultat de l'évaluation sur un échantillon."""
    sample: EvaluationSample
    retrieved_doc_ids: List[str]
    retrieved_ranks: Dict[str, int]  # doc_id -> rank
    is_correct_at_1: bool
    is_correct_at_3: bool
    is_correct_at_5: bool
    is_correct_at_10: bool
    first_correct_rank: Optional[int]
    latency_ms: float


@dataclass
class EvaluationMetrics:
    """Métriques agrégées de l'évaluation."""
    total_samples: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank
    avg_latency_ms: float
    
    # Par catégorie
    recall_by_category: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "📊 MÉTRIQUES D'ÉVALUATION",
            "=" * 50,
            f"Total samples: {self.total_samples}",
            f"",
            f"🎯 Recall@1:  {self.recall_at_1*100:.1f}%",
            f"📈 Recall@3:  {self.recall_at_3*100:.1f}%",
            f"📈 Recall@5:  {self.recall_at_5*100:.1f}%",
            f"📈 Recall@10: {self.recall_at_10*100:.1f}%",
            f"",
            f"📊 MRR: {self.mrr:.3f}",
            f"⏱️ Latency: {self.avg_latency_ms:.1f}ms",
        ]
        
        if self.recall_by_category:
            lines.append("")
            lines.append("📂 Par catégorie:")
            for cat, metrics in self.recall_by_category.items():
                lines.append(f"   {cat}:")
                lines.append(f"      R@1: {metrics.get('recall_at_1', 0)*100:.1f}%")
                lines.append(f"      R@5: {metrics.get('recall_at_5', 0)*100:.1f}%")
        
        lines.append("=" * 50)
        return "\n".join(lines)


class PipelineEvaluator:
    """
    Évaluateur du pipeline de retrieval.
    Calcule Recall@K et autres métriques.
    """
    
    def __init__(self, pipeline: RetrievalPipeline):
        self.pipeline = pipeline
    
    def evaluate_sample(
        self, 
        sample: EvaluationSample,
        top_k: int = 10
    ) -> EvaluationResult:
        """
        Évalue un échantillon unique.
        """
        start_time = time.time()
        
        # Exécute la recherche
        result = self.pipeline.search(sample.query, top_k=top_k)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extrait les doc_ids récupérés
        retrieved_doc_ids = []
        retrieved_ranks = {}
        
        for i, doc in enumerate(result.top_documents):
            doc_id = doc.get("doc_id", "")
            retrieved_doc_ids.append(doc_id)
            if doc_id not in retrieved_ranks:
                retrieved_ranks[doc_id] = i + 1  # 1-indexed
        
        # Calcule les métriques
        expected = sample.expected_doc_id
        first_correct_rank = retrieved_ranks.get(expected)
        
        is_correct_at_1 = first_correct_rank == 1 if first_correct_rank else False
        is_correct_at_3 = first_correct_rank is not None and first_correct_rank <= 3
        is_correct_at_5 = first_correct_rank is not None and first_correct_rank <= 5
        is_correct_at_10 = first_correct_rank is not None and first_correct_rank <= 10
        
        return EvaluationResult(
            sample=sample,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_ranks=retrieved_ranks,
            is_correct_at_1=is_correct_at_1,
            is_correct_at_3=is_correct_at_3,
            is_correct_at_5=is_correct_at_5,
            is_correct_at_10=is_correct_at_10,
            first_correct_rank=first_correct_rank,
            latency_ms=latency_ms
        )
    
    def evaluate(
        self, 
        samples: List[EvaluationSample],
        top_k: int = 10,
        verbose: bool = True
    ) -> Tuple[EvaluationMetrics, List[EvaluationResult]]:
        """
        Évalue le pipeline sur un ensemble d'échantillons.
        
        Returns:
            (métriques agrégées, résultats individuels)
        """
        results = []
        
        for i, sample in enumerate(samples):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Évaluation: {i+1}/{len(samples)}")
            
            result = self.evaluate_sample(sample, top_k)
            results.append(result)
        
        # Calcule les métriques agrégées
        metrics = self._compute_metrics(results)
        
        return metrics, results
    
    def _compute_metrics(self, results: List[EvaluationResult]) -> EvaluationMetrics:
        """Calcule les métriques à partir des résultats."""
        n = len(results)
        
        if n == 0:
            return EvaluationMetrics(
                total_samples=0,
                recall_at_1=0, recall_at_3=0, recall_at_5=0, recall_at_10=0,
                mrr=0, avg_latency_ms=0, recall_by_category={}
            )
        
        # Métriques globales
        recall_1 = sum(1 for r in results if r.is_correct_at_1) / n
        recall_3 = sum(1 for r in results if r.is_correct_at_3) / n
        recall_5 = sum(1 for r in results if r.is_correct_at_5) / n
        recall_10 = sum(1 for r in results if r.is_correct_at_10) / n
        
        # MRR
        mrr_sum = sum(
            1.0 / r.first_correct_rank if r.first_correct_rank else 0
            for r in results
        )
        mrr = mrr_sum / n
        
        # Latence moyenne
        avg_latency = sum(r.latency_ms for r in results) / n
        
        # Métriques par catégorie
        category_results = defaultdict(list)
        for r in results:
            cat = r.sample.category or "GENERAL"
            category_results[cat].append(r)
        
        recall_by_category = {}
        for cat, cat_results in category_results.items():
            cat_n = len(cat_results)
            recall_by_category[cat] = {
                "count": cat_n,
                "recall_at_1": sum(1 for r in cat_results if r.is_correct_at_1) / cat_n,
                "recall_at_5": sum(1 for r in cat_results if r.is_correct_at_5) / cat_n,
            }
        
        return EvaluationMetrics(
            total_samples=n,
            recall_at_1=recall_1,
            recall_at_3=recall_3,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            mrr=mrr,
            avg_latency_ms=avg_latency,
            recall_by_category=recall_by_category
        )
    
    def analyze_errors(
        self, 
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Analyse les erreurs pour identifier les patterns.
        """
        errors = [r for r in results if not r.is_correct_at_1]
        
        # Analyse par catégorie
        error_by_category = defaultdict(list)
        for e in errors:
            cat = e.sample.category or "GENERAL"
            error_by_category[cat].append(e)
        
        # Analyse des rangs
        rank_distribution = defaultdict(int)
        for e in errors:
            if e.first_correct_rank:
                rank_distribution[e.first_correct_rank] += 1
            else:
                rank_distribution["not_found"] += 1
        
        # Top erreurs
        top_errors = []
        for e in errors[:10]:
            top_errors.append({
                "query": e.sample.query,
                "expected": e.sample.expected_doc_id,
                "got_at_1": e.retrieved_doc_ids[0] if e.retrieved_doc_ids else None,
                "correct_rank": e.first_correct_rank,
                "category": e.sample.category,
            })
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(results) if results else 0,
            "errors_by_category": {k: len(v) for k, v in error_by_category.items()},
            "rank_distribution": dict(rank_distribution),
            "top_errors": top_errors,
        }


def create_test_samples_from_json(
    documents: List[Dict],
    samples_per_doc: int = 3
) -> List[EvaluationSample]:
    """
    Génère automatiquement des échantillons de test à partir des documents.
    """
    samples = []
    
    for doc in documents:
        doc_id = doc.get("filename", "unknown")
        establishment = doc.get("establishment", "")
        
        # Génère des queries basées sur le contenu
        
        # 1. Query sur les prix
        for offer in doc.get("internet_offers_table", [])[:2]:
            price = offer.get("price", "")
            speed = offer.get("speed", "")
            offer_type = offer.get("offer_type", "")
            
            if price and speed:
                query = f"Prix {offer_type} {speed} {establishment}"
                samples.append(EvaluationSample(
                    query=query,
                    expected_doc_id=doc_id,
                    expected_establishment=establishment,
                    category="PRICE"
                ))
        
        # 2. Query sur les bénéficiaires
        beneficiaries = doc.get("beneficiaries", "")
        if beneficiaries:
            query = f"Offres pour {beneficiaries[:30]} {establishment}"
            samples.append(EvaluationSample(
                query=query,
                expected_doc_id=doc_id,
                expected_establishment=establishment,
                category="BENEFICIARY"
            ))
        
        # 3. Query sur les documents
        docs_required = doc.get("required_documents_new", [])
        if docs_required:
            query = f"Documents requis pour {establishment}"
            samples.append(EvaluationSample(
                query=query,
                expected_doc_id=doc_id,
                expected_establishment=establishment,
                category="DOCUMENTS"
            ))
        
        # 4. Query générale
        query = f"Convention {establishment}"
        samples.append(EvaluationSample(
            query=query,
            expected_doc_id=doc_id,
            expected_establishment=establishment,
            category="GENERAL"
        ))
    
    return samples


def run_evaluation(
    documents_path: str,
    test_samples_path: str = None,
    config: PipelineConfig = None,
    output_path: str = None
) -> EvaluationMetrics:
    """
    Exécute une évaluation complète du pipeline.
    
    Args:
        documents_path: Chemin vers les documents JSON
        test_samples_path: Chemin vers les échantillons de test (optionnel)
        config: Configuration du pipeline
        output_path: Chemin pour sauvegarder les résultats
        
    Returns:
        Métriques d'évaluation
    """
    print("🚀 Démarrage de l'évaluation...\n")
    
    # 1. Crée le pipeline
    config = config or PipelineConfig()
    pipeline = create_pipeline(documents_path, config)
    
    # 2. Charge ou génère les échantillons de test
    if test_samples_path and os.path.exists(test_samples_path):
        print(f"📂 Chargement des échantillons depuis {test_samples_path}")
        with open(test_samples_path, 'r', encoding='utf-8') as f:
            samples_data = json.load(f)
        samples = [EvaluationSample(**s) for s in samples_data]
    else:
        print("⚙️ Génération automatique des échantillons de test...")
        with open(documents_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        samples = create_test_samples_from_json(documents)
    
    print(f"   → {len(samples)} échantillons de test\n")
    
    # 3. Évalue
    evaluator = PipelineEvaluator(pipeline)
    metrics, results = evaluator.evaluate(samples, verbose=True)
    
    # 4. Affiche les résultats
    print("\n" + str(metrics))
    
    # 5. Analyse des erreurs
    errors_analysis = evaluator.analyze_errors(results)
    print(f"\n🔍 Analyse des erreurs:")
    print(f"   Total erreurs: {errors_analysis['total_errors']}")
    print(f"   Par catégorie: {errors_analysis['errors_by_category']}")
    print(f"   Distribution des rangs: {errors_analysis['rank_distribution']}")
    
    if errors_analysis['top_errors']:
        print(f"\n   Top erreurs:")
        for err in errors_analysis['top_errors'][:5]:
            print(f"      - Query: {err['query'][:50]}...")
            print(f"        Expected: {err['expected']}, Got: {err['got_at_1']}, Correct rank: {err['correct_rank']}")
    
    # 6. Sauvegarde les résultats
    if output_path:
        output_data = {
            "metrics": metrics.to_dict(),
            "errors_analysis": errors_analysis,
            "config": asdict(config),
            "results": [
                {
                    "query": r.sample.query,
                    "expected": r.sample.expected_doc_id,
                    "category": r.sample.category,
                    "correct_at_1": r.is_correct_at_1,
                    "correct_at_5": r.is_correct_at_5,
                    "first_rank": r.first_correct_rank,
                    "retrieved": r.retrieved_doc_ids[:5],
                }
                for r in results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Résultats sauvegardés dans {output_path}")
    
    return metrics


# CLI
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <documents_path> [test_samples_path] [output_path]")
        print("\nExample:")
        print("  python evaluate.py data/conventions.json")
        print("  python evaluate.py data/conventions.json test_samples.json results.json")
        sys.exit(1)
    
    documents_path = sys.argv[1]
    test_samples_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Configuration optimale
    config = PipelineConfig(
        use_dense_retrieval=True,
        use_cross_encoder=True,
        apply_hard_entity_filter=True,
        enable_numeric_boost=True,
        enable_signature_boost=True,
        top_k_retrieval=50,
        top_k_rerank=30,
        top_k_final=10,
    )
    
    metrics = run_evaluation(
        documents_path=documents_path,
        test_samples_path=test_samples_path,
        config=config,
        output_path=output_path
    )
