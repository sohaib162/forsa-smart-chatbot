#!/usr/bin/env python3
"""
CLI entry point for the 3-layer retrieval pipeline.

Usage:
    # Interactive mode
    python main_pipeline.py --data-dir individual_docs

    # Single query mode
    python main_pipeline.py --data-dir individual_docs --query "votre question ici" --top-k 3

    # Disable dense layer for faster startup
    python main_pipeline.py --data-dir individual_docs --no-dense
"""

import argparse
import json
import sys
import logging
from pathlib import Path

from .pipeline import load_documents, RetrievalPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_result(result: dict, verbose: bool = False):
    """
    Pretty print a pipeline result, handling the new dynamic output structure.
    """
    print("\n" + "=" * 80)
    print("RETRIEVAL RESULT")
    print("=" * 80)

    print(f"\nQuery: {result['query']}")
    print(f"Layer used: {result['layer_used'].upper()}")

    best_summary = result.get('best_document_summary')
    retrieved_docs = result.get('retrieved_documents', [])
    
    # FIX: Check if best_document_summary exists before accessing its keys
    if best_summary: 
        doc_id = best_summary.get('document_id', 'N/A')
        print(f"\nBest match (Top-1):")
        print(f"  Document ID: {doc_id}")
        print(f"  Title (FR): {best_summary.get('title_fr', 'N/A')}")
        print(f"  Title (AR): {best_summary.get('title_ar', 'N/A')}")
        print(f"  Score: {best_summary.get('score', 0):.4f}")
    else:
        print("\nNo matching document found!")

    print(f"\nDynamically retrieved {len(retrieved_docs)} document(s):")
    for i, doc_data in enumerate(retrieved_docs, 1):
        summary = result['candidates_summary'][i-1]
        print(f"\n  --- Document {i} ---")
        print(f"  Score: {doc_data['score']:.4f} (from {doc_data['layer']})")
        print(f"  Doc ID: {summary.get('document_id', 'N/A')}")
        print(f"  Title (FR): {summary.get('title_fr', 'N/A')}")
        print(f"  Product family: {summary.get('product_family', 'N/A')}")
        
        if verbose:
            print("\n  FULL DOCUMENT JSON (first 1000 chars):")
            # We access the full JSON from doc_data['full_document_json']
            full_json_str = json.dumps(doc_data['full_document_json'], indent=2, ensure_ascii=False)
            print("-" * 25)
            print(full_json_str[:1000])
            if len(full_json_str) > 1000:
                print(f"\n... ({len(full_json_str) - 1000} more characters)")
            print("-" * 25)

    if verbose and result['llm_context']:
        print("\n" + "-" * 80)
        print("LLM CONTEXT (from Top-1 doc):")
        print("-" * 80)
        print(result['llm_context'][:1000])
        if len(result['llm_context']) > 1000:
            print(f"\n... ({len(result['llm_context']) - 1000} more characters)")

    print("\n" + "=" * 80 + "\n")


def interactive_mode(pipeline: RetrievalPipeline, top_k: int = 5):
    """
    Run the pipeline in interactive mode.
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE RETRIEVAL PIPELINE")
    print("=" * 80)
    print("\nEnter your queries (type 'quit', 'exit', or press Ctrl+C to exit)")
    print("Commands:")
    print("  - Type a query to search")
    print("  - 'verbose' or 'v' - toggle verbose output (shows full JSON summary)")
    print("  - 'top N' - set MAX number of results (e.g., 'top 3')")
    print("  - 'quit' or 'exit' - exit the program")
    print("=" * 80 + "\n")

    verbose = False
    current_top_k = top_k

    while True:
        try:
            query = input(f"Query (Max {current_top_k} docs): ").strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if query.lower() in ['verbose', 'v']:
                verbose = not verbose
                print(f"Verbose mode: {'ON' if verbose else 'OFF'} (shows full JSON start)")
                continue

            if query.lower().startswith('top '):
                try:
                    current_top_k = int(query.split()[1])
                    print(f"Max Top-K set to: {current_top_k}")
                    continue
                except (IndexError, ValueError):
                    print("Usage: top N (e.g., 'top 3')")
                    continue

            # Execute search
            result = pipeline.search(query, top_k=current_top_k)
            print_result(result, verbose=verbose)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"Error: {e}")


def single_query_mode(
    pipeline: RetrievalPipeline,
    query: str,
    top_k: int = 5,
    output_json: bool = False
):
    """
    Run a single query and print the result.
    """
    result = pipeline.search(query, top_k=top_k)

    if output_json:
        # Output the entire result dictionary, which now includes retrieved_documents
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_result(result, verbose=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="3-Layer Retrieval Pipeline for Algérie Télécom Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing JSON documents'
    )

    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Single query to execute (if not provided, enters interactive mode)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='MAX number of top results to return (default: 5)'
    )

    parser.add_argument(
        '--no-dense',
        action='store_true',
        help='Disable dense (embedding) layer for faster startup'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON (single query mode only)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load documents
    try:
        logger.info(f"Loading documents from: {args.data_dir}")
        docs = load_documents(args.data_dir)

        if not docs:
            logger.error("No documents loaded!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        sys.exit(1)

    # Initialize pipeline
    try:
        # We need to initialize the pipeline without passing --no-dense
        # to ensure the Cross-Encoder is checked if available.
        pipeline = RetrievalPipeline(
            docs=docs,
            use_dense=True 
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Run in appropriate mode
    if args.query:
        # Single query mode
        single_query_mode(
            pipeline=pipeline,
            query=args.query,
            top_k=args.top_k,
            output_json=args.json
        )
    else:
        # Interactive mode
        interactive_mode(pipeline=pipeline, top_k=args.top_k)


if __name__ == '__main__':
    main()