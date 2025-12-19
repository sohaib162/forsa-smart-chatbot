#!/usr/bin/env python
"""
🚀 Script Principal - Optimal Retrieval Pipeline
Target: Recall@1 ≈ 85%

Usage:
    # Générer les passages
    python main.py generate --input data/conventions.json --output data/passages.json
    
    # Rechercher
    python main.py search --data data/conventions.json --query "Prix fibre établissement P"
    
    # Évaluer
    python main.py evaluate --data data/conventions.json --output results.json
    
    # Mode interactif
    python main.py interactive --data data/conventions.json
"""

import argparse
import json
import sys
import os

# Ajoute le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_pipeline import (
    PassageGenerator,
    RetrievalPipeline,
    PipelineConfig,
)
from retrieval_pipeline.evaluate import run_evaluation, create_test_samples_from_json


def cmd_generate(args):
    """Génère les passages factuels à partir des documents."""
    print(f"📂 Chargement des documents depuis {args.input}...")
    
    with open(args.input, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"   → {len(documents)} documents chargés")
    
    generator = PassageGenerator()
    passages = generator.generate_all_passages(documents)
    
    print(f"✅ {len(passages)} passages générés")
    
    # Sauvegarde
    output_path = args.output or "data/passages.json"
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([p.to_dict() for p in passages], f, ensure_ascii=False, indent=2)
    
    print(f"💾 Passages sauvegardés dans {output_path}")
    
    # Stats
    types_count = {}
    entity_count = {}
    for p in passages:
        t = p.passage_type
        e = p.entity_code
        types_count[t] = types_count.get(t, 0) + 1
        entity_count[e] = entity_count.get(e, 0) + 1
    
    print("\n📊 Statistiques:")
    print("   Par type:")
    for t, c in sorted(types_count.items()):
        print(f"      {t}: {c}")
    print("   Par établissement:")
    for e, c in sorted(entity_count.items()):
        print(f"      {e}: {c}")


def cmd_search(args):
    """Exécute une recherche unique."""
    # Configuration
    config = PipelineConfig(
        use_dense_retrieval=not args.no_dense,
        use_cross_encoder=not args.no_rerank,
        apply_hard_entity_filter=True,
        enable_numeric_boost=True,
        enable_signature_boost=True,
    )
    
    # Initialise le pipeline
    pipeline = RetrievalPipeline(config)
    
    passages_path = args.passages or "data/passages.json"
    if os.path.exists(passages_path):
        pipeline.initialize(passages_path=passages_path)
    else:
        pipeline.initialize(documents_path=args.data)
    
    # Recherche
    query = args.query
    result = pipeline.search(query, top_k=args.top_k)
    
    # Affiche les résultats
    print(f"\n{'='*60}")
    print(f"🔍 Query: {query}")
    print(f"{'='*60}")
    print(f"Intent: {result.intent} (conf: {result.intent_confidence:.2f})")
    print(f"Entities: {result.detected_entities}")
    print(f"Filter: {'✅ Applied' if result.entity_filter_applied else '❌ Not applied'}")
    print(f"Passages: {result.total_passages_retrieved} → {result.total_passages_after_filter}")
    
    print(f"\n📚 Top {args.top_k} Documents:")
    for i, doc in enumerate(result.top_documents):
        print(f"\n  {i+1}. {doc['doc_id']}")
        print(f"     Score: {doc['aggregated_score']:.4f}")
        if doc.get('cross_encoder_score'):
            print(f"     CE Score: {doc['cross_encoder_score']:.4f}")
        if doc.get('best_passage'):
            text = doc['best_passage'].get('text', '')[:100]
            print(f"     → {text}...")
    
    if args.verbose:
        print(f"\n📄 Top Passages:")
        for i, p in enumerate(result.top_passages[:5]):
            print(f"  {i+1}. [{p['entity_code']}] Score: {p['score']:.4f}")
            print(f"     {p['text'][:80]}...")


def cmd_evaluate(args):
    """Évalue le pipeline."""
    config = PipelineConfig(
        use_dense_retrieval=not args.no_dense,
        use_cross_encoder=not args.no_rerank,
        apply_hard_entity_filter=True,
        enable_numeric_boost=True,
        enable_signature_boost=True,
    )
    
    run_evaluation(
        documents_path=args.data,
        test_samples_path=args.samples,
        config=config,
        output_path=args.output
    )


def cmd_interactive(args):
    """Mode interactif."""
    config = PipelineConfig(
        use_dense_retrieval=not args.no_dense,
        use_cross_encoder=not args.no_rerank,
    )
    
    pipeline = RetrievalPipeline(config)
    
    passages_path = args.passages or "data/passages.json"
    if os.path.exists(passages_path):
        pipeline.initialize(passages_path=passages_path)
    else:
        pipeline.initialize(documents_path=args.data)
    
    print("\n" + "="*60)
    print("🔍 Mode Interactif - Retrieval Pipeline")
    print("="*60)
    print("Tapez une requête et appuyez sur Entrée.")
    print("Commandes spéciales:")
    print("  /explain <query> - Explication détaillée")
    print("  /quit           - Quitter")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("Query > ").strip()
            
            if not query:
                continue
            
            if query.lower() == "/quit":
                print("Au revoir! 👋")
                break
            
            if query.startswith("/explain "):
                query = query[9:]
                explanation = pipeline.explain_search(query)
                print(json.dumps(explanation, indent=2, ensure_ascii=False, default=str))
                continue
            
            result = pipeline.search(query, top_k=5)
            
            print(f"\n  Intent: {result.intent} | Entities: {result.detected_entities}")
            print(f"  Results:")
            for i, doc in enumerate(result.top_documents[:3]):
                print(f"    {i+1}. {doc['doc_id']} (score: {doc['aggregated_score']:.3f})")
                if doc.get('best_passage'):
                    print(f"       → {doc['best_passage'].get('text', '')[:60]}...")
            print()
            
        except KeyboardInterrupt:
            print("\nAu revoir! 👋")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")


def cmd_generate_samples(args):
    """Génère des échantillons de test."""
    with open(args.data, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    samples = create_test_samples_from_json(documents)
    
    output_path = args.output or "data/test_samples.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([
            {
                "query": s.query,
                "expected_doc_id": s.expected_doc_id,
                "expected_establishment": s.expected_establishment,
                "category": s.category
            }
            for s in samples
        ], f, ensure_ascii=False, indent=2)
    
    print(f"✅ {len(samples)} échantillons générés dans {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimal Retrieval Pipeline - Target Recall@1 ≈ 85%",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")
    
    # Generate
    gen_parser = subparsers.add_parser("generate", help="Génère les passages")
    gen_parser.add_argument("--input", "-i", required=True, help="Fichier JSON des documents")
    gen_parser.add_argument("--output", "-o", help="Fichier de sortie pour les passages")
    
    # Search
    search_parser = subparsers.add_parser("search", help="Recherche")
    search_parser.add_argument("--data", "-d", required=True, help="Fichier JSON des documents")
    search_parser.add_argument("--passages", "-p", help="Fichier des passages (optionnel)")
    search_parser.add_argument("--query", "-q", required=True, help="Requête de recherche")
    search_parser.add_argument("--top-k", "-k", type=int, default=5, help="Nombre de résultats")
    search_parser.add_argument("--no-dense", action="store_true", help="Désactive le retrieval dense")
    search_parser.add_argument("--no-rerank", action="store_true", help="Désactive le cross-encoder")
    search_parser.add_argument("--verbose", "-v", action="store_true", help="Affiche plus de détails")
    
    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Évalue le pipeline")
    eval_parser.add_argument("--data", "-d", required=True, help="Fichier JSON des documents")
    eval_parser.add_argument("--samples", "-s", help="Fichier des échantillons de test")
    eval_parser.add_argument("--output", "-o", help="Fichier de sortie des résultats")
    eval_parser.add_argument("--no-dense", action="store_true", help="Désactive le retrieval dense")
    eval_parser.add_argument("--no-rerank", action="store_true", help="Désactive le cross-encoder")
    
    # Interactive
    int_parser = subparsers.add_parser("interactive", help="Mode interactif")
    int_parser.add_argument("--data", "-d", required=True, help="Fichier JSON des documents")
    int_parser.add_argument("--passages", "-p", help="Fichier des passages (optionnel)")
    int_parser.add_argument("--no-dense", action="store_true", help="Désactive le retrieval dense")
    int_parser.add_argument("--no-rerank", action="store_true", help="Désactive le cross-encoder")
    
    # Generate samples
    samples_parser = subparsers.add_parser("generate-samples", help="Génère des échantillons de test")
    samples_parser.add_argument("--data", "-d", required=True, help="Fichier JSON des documents")
    samples_parser.add_argument("--output", "-o", help="Fichier de sortie")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "generate-samples":
        cmd_generate_samples(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
