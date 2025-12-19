#!/usr/bin/env python3
"""
NGBSS Retrieval Pipeline - Main Runner
=======================================

Run individual steps or the complete pipeline.

Usage:
    # Run all steps
    python run_pipeline.py --all
    
    # Run individual steps
    python run_pipeline.py --step 1  # Data preparation
    python run_pipeline.py --step 2  # BM25 indexing
    python run_pipeline.py --step 3  # Dense indexing
    python run_pipeline.py --step 4  # Query pipeline demo
    
    # Interactive search
    python run_pipeline.py --search "TVA 2%"
    python run_pipeline.py --interactive
    
    # With document path
    python run_pipeline.py --search "facture" --docs-root /path/to/documents
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_step1():
    """Run data preparation"""
    from scripts.step1_data_preparation import main
    return main()


def run_step2():
    """Run BM25 indexing"""
    from scripts.step2_sparse_index import main
    return main()


def run_step3():
    """Run dense indexing"""
    from scripts.step3_dense_index import main
    return main()


def run_step4():
    """Run query pipeline demo"""
    from scripts.step4_query_pipeline import main
    return main()


def run_all():
    """Run all pipeline steps"""
    print("\n" + "=" * 70)
    print(" NGBSS RETRIEVAL PIPELINE - FULL BUILD")
    print("=" * 70)
    
    print("\n[1/4] Data Preparation...")
    run_step1()
    
    print("\n[2/4] BM25 Indexing...")
    run_step2()
    
    print("\n[3/4] Dense Indexing...")
    run_step3()
    
    print("\n[4/4] Pipeline Demo...")
    run_step4()
    
    print("\n" + "=" * 70)
    print(" PIPELINE BUILD COMPLETE!")
    print("=" * 70)
    print("\nYou can now use the retrieval API:")
    print("  from scripts.retrieval_api import NGBSSRetriever")
    print('  retriever = NGBSSRetriever(documents_root="/path/to/docs")')
    print('  results = retriever.search("your query")')


def run_search(query: str, docs_root: str = None, top_k: int = 5):
    """Run a search query"""
    from scripts.retrieval_api import NGBSSRetriever
    
    print(f"\n🔍 Searching: '{query}'")
    
    with NGBSSRetriever(documents_root=docs_root, enable_reranking=False) as retriever:
        # Use search_with_urls to get presigned URLs
        results, timing = retriever.search_with_urls(query, top_k=top_k)
        
        # Calculate retrieval time (total minus URL generation)
        url_time = timing.get('url_generation_ms', 0)
        retrieval_time = timing['total_ms'] - url_time
        
        print(f"⏱ Retrieval: {retrieval_time:.0f}ms | URLs: {url_time:.0f}ms | Total: {timing['total_ms']:.0f}ms")
        print(f"   ├─ BM25: {timing['bm25_ms']:.0f}ms")
        print(f"   ├─ Dense: {timing['dense_ms']:.0f}ms")
        print(f"   └─ Fusion: {timing['fusion_ms']:.0f}ms")
        
        if timing.get('filters_applied'):
            print(f"🏷 Filters: {timing['filters_applied']}")
        
        print(f"\n📄 Found {len(results)} results:\n")
        
        for r in results:
            print(f"{'─'*60}")
            print(f"#{r.rank} [{r.score:.3f}] {r.guide_title}")
            if r.section_title:
                print(f"   Section: {r.section_title}")
            print(f"   File: {r.filename}")
            print(f"   Tags: {', '.join(r.tags[:5])}")
            if r.url:
                print(f"   🔗 URL: {r.url}")
            elif r.s3_key:
                print(f"   S3 Key: {r.s3_key}")
            if docs_root and r.document_path:
                status = "✓ exists" if r.exists() else "✗ not found"
                print(f"   Document: {r.document_path} ({status})")


def run_rag_search(query: str, top_k: int = 3, output_file: str = None):
    """Run RAG search and return complete JSON for LLM"""
    from scripts.retrieval_api import NGBSSRetriever
    import json
    
    print(f"\n🤖 RAG Search: '{query}'")
    
    with NGBSSRetriever(enable_reranking=False) as retriever:
        result = retriever.search_for_rag(query, top_k=top_k)
        
        print(f"⏱ Retrieval: {result['retrieval_info']['retrieval_time_ms']:.0f}ms")
        print(f"📄 Found {result['retrieval_info']['total_results']} guides")
        
        # Pretty print summary
        for doc in result['retrieved_documents']:
            guide = doc['guide']
            print(f"\n  #{doc['rank']} [{doc['relevance_score']:.3f}] {guide['title']}")
            print(f"      Section: {doc['matched_section']}")
            print(f"      Sections: {len(guide['sections'])} | Tags: {', '.join(guide['tags'][:3])}")
            if guide.get('url'):
                print(f"      🔗 URL: {guide['url'][:80]}...")
        
        # Output JSON
        json_output = json.dumps(result, ensure_ascii=False, indent=2)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"\n✓ Saved RAG JSON to: {output_file}")
        else:
            print(f"\n{'='*60}")
            print("📋 RAG JSON OUTPUT (for DeepSeek):")
            print('='*60)
            print(json_output)
        
        return result


def run_interactive(docs_root: str = None):
    """Interactive search mode"""
    from scripts.retrieval_api import NGBSSRetriever
    
    print("\n" + "=" * 60)
    print(" NGBSS Interactive Search")
    print("=" * 60)
    print("\nCommands:")
    print("  /quit or /exit - Exit")
    print("  /guide <query> - Search guides")
    print("  /section <query> - Search sections (default)")
    print("  /step <query> - Search steps")
    print("  /top <n> - Set number of results")
    print("")
    
    retriever = NGBSSRetriever(documents_root=docs_root, enable_reranking=False)
    retriever.connect()
    
    top_k = 5
    doc_type = "section"
    
    try:
        while True:
            try:
                query = input("\n🔍 Query > ").strip()
            except EOFError:
                break
            
            if not query:
                continue
            
            if query.lower() in ['/quit', '/exit', '/q']:
                break
            
            if query.startswith('/top '):
                try:
                    top_k = int(query[5:].strip())
                    print(f"✓ Set top_k to {top_k}")
                except ValueError:
                    print("✗ Invalid number")
                continue
            
            if query.startswith('/guide '):
                doc_type = "guide"
                query = query[7:].strip()
            elif query.startswith('/section '):
                doc_type = "section"
                query = query[9:].strip()
            elif query.startswith('/step '):
                doc_type = "step"
                query = query[6:].strip()
            
            if query:
                # Use search_with_urls to get presigned URLs
                results, timing = retriever.search_with_urls(
                    query, top_k=top_k, doc_type=doc_type
                )
                
                # Calculate retrieval time (total minus URL generation)
                url_time = timing.get('url_generation_ms', 0)
                retrieval_time = timing['total_ms'] - url_time
                
                print(f"⏱ Retrieval: {retrieval_time:.0f}ms | URLs: {url_time:.0f}ms | Type: {doc_type}")
                
                for r in results:
                    print(f"\n  #{r.rank} [{r.score:.3f}] {r.guide_title}")
                    if r.section_title:
                        print(f"      └─ {r.section_title}")
                    print(f"      File: {r.filename}")
                    if r.url:
                        print(f"      🔗 {r.url}")
    
    finally:
        retriever.close()
    
    print("\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="NGBSS Retrieval Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all                    # Build everything
  python run_pipeline.py --step 1                 # Just data prep
  python run_pipeline.py --search "TVA 2%"        # Quick search
  python run_pipeline.py --rag "TVA 2%"           # RAG search (full JSON for LLM)
  python run_pipeline.py --interactive            # Interactive mode
        """
    )
    
    parser.add_argument(
        '--all', action='store_true',
        help='Run all pipeline steps'
    )
    parser.add_argument(
        '--step', type=int, choices=[1, 2, 3, 4],
        help='Run specific step: 1=data, 2=BM25, 3=dense, 4=demo'
    )
    parser.add_argument(
        '--search', type=str,
        help='Run a search query'
    )
    parser.add_argument(
        '--rag', type=str,
        help='RAG search - returns complete JSON with all metadata for LLM'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output file for RAG JSON (optional)'
    )
    parser.add_argument(
        '--interactive', '-i', action='store_true',
        help='Interactive search mode'
    )
    parser.add_argument(
        '--docs-root', type=str,
        help='Root directory for documents (PDF/DOCX)'
    )
    parser.add_argument(
        '--top-k', type=int, default=5,
        help='Number of results to return (default: 5)'
    )
    
    args = parser.parse_args()
    
    if args.all:
        run_all()
    elif args.step:
        steps = {1: run_step1, 2: run_step2, 3: run_step3, 4: run_step4}
        steps[args.step]()
    elif args.search:
        run_search(args.search, args.docs_root, args.top_k)
    elif args.rag:
        run_rag_search(args.rag, args.top_k, args.output)
    elif args.interactive:
        run_interactive(args.docs_root)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
