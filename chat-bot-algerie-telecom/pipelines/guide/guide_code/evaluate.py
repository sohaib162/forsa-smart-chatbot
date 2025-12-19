"""
Evaluation Script for NGBSS Retrieval Pipeline
==============================================
Measures retrieval accuracy and performance on the test dataset.

Targets:
- Accuracy: >85% Recall@5
- Speed: <1000ms per query

Run: python evaluate.py
"""
import json
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))
from scripts.step2_sparse_index import BM25Index
from scripts.step4_query_pipeline import QueryPipeline, RetrievalResult

# =============================================================================
# Configuration
# =============================================================================

MAX_RETRIEVAL_TIME_MS = 1000  # 1 second target
TARGET_ACCURACY = 0.85  # 85% target

# =============================================================================
# Source Document to Guide ID Mapping
# =============================================================================

SOURCE_DOCUMENT_TO_GUIDE_ID = {
    # Exact matches from test file
    "Guide d'utilisation NGBSS Création enquête PSTN et la Gestion d'ordre": "creation_enquete_pstn_gestion_ordre",
    "Guide d'Utilisation NGBSS Création nouveau Pack IDOOM Fibre": "creation_nouveau_pack_idoom_fibre",
    "Encaissement des factures payées au niveau de bureau de Poste": "encaissement_factures_payees_bureau_poste",
    "Guide d'utilisation NGBSS - La Saisie de la facture détaillée": "saisie_facture_detaillee",
    "Guide d'Utilisation NGBSS << Inventaire »": "inventaire_ngbss",
    "Guide NGBSS Réactivation Abonné 4G LTE": "reactivation_abonne_4g_lte",
    "Guide d'Utilisation NGBSS Retour ressource": "retour_ressource",
    "Guide d'utilisation NGBSS - Gestion de Payement Arrangement -Échéancier-": "gestion_payement_arrangement_echeancier",
    "Guide d'utilisation NGBSS - Enregistrement et encaissement Facture complémentaire TVA 2%": "facture_complementaire_tva_2",
    "Guide d'Utilisation NGBSS - Ventes par lot (Version 2)": "ventes_par_lot_v2",
    "Guide d'utilisation NGBSS Edition facture duplicata": "edition_facture_duplicata",
    "Guide d'Utilisation NGBSS << Ligne temporaire >>": "ligne_temporaire",
    "Guide d'Utilisation NGBSS Recharge par Bon de commande - Complément-": "recharge_bon_commande_complement",
}

# Additional mappings for variations in test file
ADDITIONAL_MAPPINGS = {
    "enquête pstn": "creation_enquete_pstn_gestion_ordre",
    "gestion d'ordre": "creation_enquete_pstn_gestion_ordre",
    "idoom fibre": "creation_nouveau_pack_idoom_fibre",
    "ftth": "creation_nouveau_pack_idoom_fibre",
    "bureau de poste": "encaissement_factures_payees_bureau_poste",
    "encaissement externe": "encaissement_factures_payees_bureau_poste",
    "facture détaillée": "saisie_facture_detaillee",
    "fadet": "saisie_facture_detaillee",
    "inventaire": "inventaire_ngbss",
    "4g lte": "reactivation_abonne_4g_lte",
    "réactivation": "reactivation_abonne_4g_lte",
    "retour ressource": "retour_ressource",
    "remboursement": "retour_ressource",
    "échéancier": "gestion_payement_arrangement_echeancier",
    "arrangement": "gestion_payement_arrangement_echeancier",
    "aod": "gestion_payement_arrangement_echeancier",
    "tva 2%": "facture_complementaire_tva_2",
    "facture complémentaire": "facture_complementaire_tva_2",
    "ventes par lot": "ventes_par_lot_v2",
    "duplicata": "edition_facture_duplicata",
    "ligne temporaire": "ligne_temporaire",
    "bon de commande": "recharge_bon_commande_complement",
    "recharge": "recharge_bon_commande_complement",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvaluationResult:
    """Result of evaluating a single query"""
    question: str
    expected_guide_id: str
    expected_source: str
    retrieved_guide_ids: List[str]
    
    # Metrics
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    
    # Timing
    time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "expected": self.expected_source,
            "expected_guide_id": self.expected_guide_id,
            "retrieved": self.retrieved_guide_ids[:5],
            "hit@1": self.hit_at_1,
            "hit@3": self.hit_at_3,
            "hit@5": self.hit_at_5,
            "mrr": self.reciprocal_rank,
            "time_ms": self.time_ms
        }


@dataclass
class EvaluationSummary:
    """Summary of evaluation results"""
    total_queries: int
    
    # Accuracy metrics
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float  # Mean Reciprocal Rank
    
    # Timing metrics
    avg_time_ms: float
    max_time_ms: float
    min_time_ms: float
    queries_under_1s: int
    
    # Per-guide breakdown
    per_guide_accuracy: Dict[str, float]
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "EVALUATION SUMMARY",
            "=" * 60,
            f"Total queries: {self.total_queries}",
            "",
            "Accuracy Metrics:",
            f"  Recall@1: {self.recall_at_1:.1%}",
            f"  Recall@3: {self.recall_at_3:.1%}",
            f"  Recall@5: {self.recall_at_5:.1%}",
            f"  MRR:      {self.mrr:.4f}",
            "",
            "Performance Metrics:",
            f"  Avg time: {self.avg_time_ms:.1f}ms",
            f"  Max time: {self.max_time_ms:.1f}ms",
            f"  Min time: {self.min_time_ms:.1f}ms",
            f"  Under 1s: {self.queries_under_1s}/{self.total_queries} ({self.queries_under_1s/self.total_queries:.1%})",
            "",
            "Target Check:",
            f"  Accuracy target (>85%): {'✓ PASS' if self.recall_at_5 >= TARGET_ACCURACY else '✗ FAIL'} ({self.recall_at_5:.1%})",
            f"  Speed target (<1s):     {'✓ PASS' if self.queries_under_1s == self.total_queries else '✗ FAIL'}",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for matching"""
    text = text.lower()
    # Fix encoding issues
    replacements = {
        "ã©": "é", "ã¨": "è", "ã ": "à", "ã®": "î",
        "ã´": "ô", "ã¢": "â", "ã§": "ç", "ãª": "ê",
        "â«": "»", "â»": "»", "Â«": "«", "Â»": "»",
        "<<": "", ">>": "", "«": "", "»": "",
        "-": " ", "_": " ", "  ": " "
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def find_guide_id(source_document: str) -> str:
    """
    Find guide_id from source_document string.
    Handles variations in naming and encoding issues.
    """
    # Fix encoding first
    source_fixed = normalize_text(source_document)
    
    # Try exact match first
    for source, guide_id in SOURCE_DOCUMENT_TO_GUIDE_ID.items():
        if normalize_text(source) == source_fixed:
            return guide_id
    
    # Try partial matching
    for source, guide_id in SOURCE_DOCUMENT_TO_GUIDE_ID.items():
        normalized = normalize_text(source)
        if source_fixed in normalized or normalized in source_fixed:
            return guide_id
    
    # Try keyword matching
    for keyword, guide_id in ADDITIONAL_MAPPINGS.items():
        if keyword in source_fixed:
            return guide_id
    
    return "unknown"


def get_unique_guide_ids(results: List[RetrievalResult]) -> List[str]:
    """Get unique guide IDs from results, preserving order"""
    seen = set()
    unique = []
    for r in results:
        if r.guide_id not in seen:
            seen.add(r.guide_id)
            unique.append(r.guide_id)
    return unique


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_query(
    pipeline: QueryPipeline,
    question: str,
    expected_source: str,
    top_k: int = 10
) -> EvaluationResult:
    """Evaluate a single query"""
    expected_guide_id = find_guide_id(expected_source)
    
    # Run search
    start_time = time.time()
    results, timing = pipeline.search(question, top_k=top_k, doc_type="section")
    elapsed_ms = timing['total_ms']
    
    # Get unique retrieved guide IDs
    retrieved_ids = get_unique_guide_ids(results)
    
    # Calculate metrics
    hit_at_1 = expected_guide_id in retrieved_ids[:1]
    hit_at_3 = expected_guide_id in retrieved_ids[:3]
    hit_at_5 = expected_guide_id in retrieved_ids[:5]
    
    # Reciprocal rank
    try:
        rank = retrieved_ids.index(expected_guide_id) + 1
        rr = 1.0 / rank
    except ValueError:
        rr = 0.0
    
    return EvaluationResult(
        question=question,
        expected_guide_id=expected_guide_id,
        expected_source=expected_source,
        retrieved_guide_ids=retrieved_ids,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        reciprocal_rank=rr,
        time_ms=elapsed_ms
    )


def run_evaluation(
    test_file: str = "test_questions.json",
    verbose: bool = True
) -> Tuple[EvaluationSummary, List[EvaluationResult]]:
    """
    Run full evaluation on test dataset.
    
    Args:
        test_file: Path to test questions JSON
        verbose: Print progress
        
    Returns:
        Tuple of (summary, individual_results)
    """
    # Load test data
    test_path = Path(__file__).parent / test_file
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if verbose:
        print(f"Loaded {len(test_data)} test questions")
    
    # Initialize pipeline
    if verbose:
        print("Initializing pipeline...")
    
    pipeline = QueryPipeline(enable_reranking=True)  # Enable reranking for better accuracy
    pipeline.connect()
    
    if verbose:
        dense_status = "with dense" if getattr(pipeline, '_dense_available', False) else "BM25 only"
        print(f"Pipeline ready ({dense_status})")
    
    # Warm-up query to load models (not counted in timing)
    _ = pipeline.search("test warmup query", top_k=1)
    
    # Run evaluation
    results = []
    per_guide_hits = defaultdict(lambda: {"hits": 0, "total": 0})
    
    for i, item in enumerate(test_data):
        question = item["question"]
        expected = item["source_document"]
        
        result = evaluate_query(pipeline, question, expected)
        results.append(result)
        
        # Track per-guide accuracy
        per_guide_hits[result.expected_guide_id]["total"] += 1
        if result.hit_at_5:
            per_guide_hits[result.expected_guide_id]["hits"] += 1
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_data)} queries...")
    
    pipeline.close()
    
    # Calculate summary metrics
    total = len(results)
    times = [r.time_ms for r in results]
    
    per_guide_accuracy = {
        guide: data["hits"] / data["total"] if data["total"] > 0 else 0
        for guide, data in per_guide_hits.items()
    }
    
    summary = EvaluationSummary(
        total_queries=total,
        recall_at_1=sum(r.hit_at_1 for r in results) / total,
        recall_at_3=sum(r.hit_at_3 for r in results) / total,
        recall_at_5=sum(r.hit_at_5 for r in results) / total,
        mrr=sum(r.reciprocal_rank for r in results) / total,
        avg_time_ms=sum(times) / total,
        max_time_ms=max(times),
        min_time_ms=min(times),
        queries_under_1s=sum(1 for t in times if t < MAX_RETRIEVAL_TIME_MS),
        per_guide_accuracy=per_guide_accuracy
    )
    
    return summary, results


def print_failed_queries(results: List[EvaluationResult]):
    """Print queries that failed to retrieve the correct document"""
    failed = [r for r in results if not r.hit_at_5]
    
    if not failed:
        print("\n✓ No failed queries! All queries retrieved correct document in top 5.")
        return
    
    print(f"\n{'='*60}")
    print(f"FAILED QUERIES ({len(failed)} total)")
    print("="*60)
    
    for r in failed:
        print(f"\nQuestion: {r.question}")
        print(f"Expected: {r.expected_source}")
        print(f"Expected ID: {r.expected_guide_id}")
        print(f"Retrieved: {r.retrieved_guide_ids[:5]}")


def save_results(
    summary: EvaluationSummary,
    results: List[EvaluationResult],
    output_file: str = "evaluation_results.json"
):
    """Save evaluation results to JSON"""
    output = {
        "summary": {
            "total_queries": summary.total_queries,
            "recall_at_1": summary.recall_at_1,
            "recall_at_3": summary.recall_at_3,
            "recall_at_5": summary.recall_at_5,
            "mrr": summary.mrr,
            "avg_time_ms": summary.avg_time_ms,
            "max_time_ms": summary.max_time_ms,
            "queries_under_1s": summary.queries_under_1s,
            "per_guide_accuracy": summary.per_guide_accuracy,
        },
        "results": [r.to_dict() for r in results]
    }
    
    output_path = Path(__file__).parent / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate NGBSS Retrieval Pipeline")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--show-failed", action="store_true", help="Show failed queries")
    parser.add_argument("--show-all", action="store_true", help="Show all query results")
    args = parser.parse_args()
    
    print("=" * 60)
    print("NGBSS Retrieval Pipeline Evaluation")
    print("=" * 60)
    
    summary, results = run_evaluation(verbose=True)
    
    print()
    print(summary)
    
    if args.show_failed or args.show_all:
        print_failed_queries(results)
    
    if args.show_all:
        print(f"\n{'='*60}")
        print("ALL RESULTS")
        print("="*60)
        for r in results:
            status = "✓" if r.hit_at_5 else "✗"
            print(f"\n{status} Q: {r.question[:60]}...")
            print(f"  Expected: {r.expected_guide_id}")
            print(f"  Got: {r.retrieved_guide_ids[:3]}")
            print(f"  Time: {r.time_ms:.1f}ms")
    
    if args.save:
        save_results(summary, results)
    
    # Print per-guide breakdown
    print(f"\n{'='*60}")
    print("PER-GUIDE ACCURACY")
    print("="*60)
    for guide_id, accuracy in sorted(summary.per_guide_accuracy.items()):
        status = "✓" if accuracy >= 0.85 else "✗"
        print(f"  {status} {guide_id}: {accuracy:.0%}")
