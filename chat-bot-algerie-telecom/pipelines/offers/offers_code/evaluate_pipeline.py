#!/usr/bin/env python3
"""
Evaluate the 3-layer retrieval pipeline against labeled queries.

- Uses the existing RetrievalPipeline (rule -> sparse BM25 -> dense).
- Labeled queries come from a JSON file with objects: {"query": ..., "title": ...}
- Computes:
    * Overall top-1 and top-K accuracy
    * Per-layer pipeline accuracy (which layer actually answered)
    * Raw per-layer accuracy (rule-only, sparse-only, dense-only)

Usage example:

    python evaluate_pipeline.py \
        --data-dir individual_docs \
        --queries query.JSON \
        --top-k 5

"""

import argparse
import time
import json
import logging
from pathlib import Path
from collections import defaultdict

from .pipeline import load_documents, RetrievalPipeline
from pipeline.loader import safe_get  # reuse your helpers
from pipeline.text_normalization import simple_normalize  # use shared normalization

logger = logging.getLogger(__name__)


def normalize_label(text: str) -> str:
    """
    Normalize labels/titles for matching.

    Uses the shared simple_normalize function for consistency with
    dense index query augmentation.
    """
    return simple_normalize(text or "")


def build_label_index(docs):
    """
    Build mapping from normalized label -> list of doc indices.

    We use several possible label sources per document:
      - metadata.title_fr / metadata.title_ar
      - offer_core.name_fr / offer_core.name_ar
    """
    label_to_indices = defaultdict(list)

    for idx, doc in enumerate(docs):
        meta = doc.get("metadata") or {}
        offer = doc.get("offer_core") or {}

        candidates = [
            meta.get("title_fr"),
            meta.get("title_ar"),
            offer.get("name_fr"),
            offer.get("name_ar"),
        ]

        for label in candidates:
            if not label:
                continue
            key = normalize_label(label)
            if key:
                label_to_indices[key].append(idx)

    return label_to_indices


def find_ground_truth_index(label_index, title: str):
    """
    Map a test 'title' from the queries JSON to a document index.

    1) Try exact normalized match
    2) Fallback to simple substring matching
    """
    key = normalize_label(title)

    # Exact match
    if key in label_index:
        return label_index[key][0]  # if multiple, just pick first

    # Fuzzy fallback: substring match
    for stored_key, indices in label_index.items():
        if key and (key in stored_key or stored_key in key):
            return indices[0]

    return None


def load_queries(path: str):
    """
    Load labeled queries from JSON.

    Accepts either:
      - a list: [ {"query": ..., "title": ...}, ... ]
      - or { "queries": [ ... ] }
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "queries" in data:
        return data["queries"]

    raise ValueError("Unsupported queries JSON format. "
                     "Expect a list or a dict with key 'queries'.")


def evaluate_pipeline(docs, pipeline, queries, top_k: int = 5):
    """
    Run evaluation over all queries.

    Returns:
        stats: dict with global and per-layer metrics
        hard_errors: list of queries where GT is not even in top-K of any layer
    """
    label_index = build_label_index(docs)

    # Global stats
    stats = {
        "total": 0,
        "skipped_no_label": 0,
        "correct_top1": 0,
        "correct_topk": 0,
        "per_layer_pipeline": defaultdict(lambda: {
            "n": 0,
            "correct_top1": 0,
            "correct_topk": 0,
        }),
        "per_layer_raw": defaultdict(lambda: {
            "n": 0,
            "correct_top1": 0,
            "correct_topk": 0,
        }),
    }

    hard_errors = []  # where GT not in top-K of the layer that answered
    per_query_results = []

    for i, sample in enumerate(queries):
        q_text = sample.get("query", "")
        label_title = sample.get("title", "")

        gt_idx = find_ground_truth_index(label_index, label_title)
        if gt_idx is None:
            stats["skipped_no_label"] += 1
            logger.warning(
                f"[{i}] Could not map label '{label_title}' to any document. Skipping."
            )
            continue

        stats["total"] += 1

        # === 1) Run the real pipeline (with gating) and time it ===
        start_ts = time.time()
        result = pipeline.search(q_text, top_k=top_k)
        end_ts = time.time()
        elapsed_ms = (end_ts - start_ts) * 1000.0

        # Extract retrieved document ids
        retrieved_docs = result.get("retrieved_documents", []) or []
        retrieved_offers = []
        for rd in retrieved_docs:
            if isinstance(rd, dict):
                full = rd.get("full_document_json")
                if isinstance(full, dict):
                    retrieved_offers.append(full.get("document_id"))
                else:
                    # fallback
                    retrieved_offers.append(rd.get("document_id"))

        # Compute per-query hit@k and mrr
        expected_id = pipeline.docs[gt_idx].get("document_id") if gt_idx is not None else None
        def compute_mrr(expected, retrieved):
            if not expected:
                return 0.0
            try:
                idx = retrieved.index(expected)
                return 1.0 / (idx + 1)
            except ValueError:
                return 0.0

        q_hit1 = bool(retrieved_offers and retrieved_offers[0] == expected_id)
        q_hit3 = any(ro == expected_id for ro in retrieved_offers[:3])
        q_hit5 = any(ro == expected_id for ro in retrieved_offers[:5])
        q_mrr = compute_mrr(expected_id, retrieved_offers)

        per_query_results.append({
            "query": q_text,
            "expected_offer": expected_id,
            "retrieved_offers": [r for r in retrieved_offers if r is not None],
            "hit@1": q_hit1,
            "hit@3": q_hit3,
            "hit@5": q_hit5,
            "mrr": q_mrr,
            "time_ms": elapsed_ms,
        })

        # Pipeline-level stats (use returned structure)
        layer_used = result.get("layer_used")
        candidates = retrieved_docs
        best_idx = candidates[0]["doc_index"] if candidates else None

        layer_stats = stats["per_layer_pipeline"][layer_used]
        layer_stats["n"] += 1

        # top-1 correctness (pipeline)
        is_top1 = (best_idx == gt_idx)
        if is_top1:
            stats["correct_top1"] += 1
            layer_stats["correct_top1"] += 1

        # top-K correctness (pipeline)
        in_topk = any(c["doc_index"] == gt_idx for c in candidates[:top_k])
        if in_topk:
            stats["correct_topk"] += 1
            layer_stats["correct_topk"] += 1

        # Track "hard" errors: pipeline layer didn't even put GT in top-K
        if not in_topk:
            hard_errors.append({
                "query": q_text,
                "label_title": label_title,
                "layer_used": layer_used,
                "ground_truth_index": gt_idx,
                "predicted_index": best_idx,
                "predicted_title_fr": safe_get(
                    pipeline.docs[best_idx], "metadata", "title_fr",
                    default="N/A") if best_idx is not None else None,
                "predicted_title_ar": safe_get(
                    pipeline.docs[best_idx], "metadata", "title_ar",
                    default="N/A") if best_idx is not None else None,
            })

        # === 2) Evaluate each layer "raw", without gating ===
        
        # 2.a) Rule layer alone
        # FIX: Changed method name from .route() to .filter_candidates()
        rule_result = pipeline.router.filter_candidates(q_text)
        rule_cands = rule_result["candidates"][:top_k] if rule_result else []
        if rule_cands:
            raw = stats["per_layer_raw"]["rule"]
            raw["n"] += 1
            if rule_cands[0]["doc_index"] == gt_idx:
                raw["correct_top1"] += 1
            if any(c["doc_index"] == gt_idx for c in rule_cands):
                raw["correct_topk"] += 1

        # 2.b) Sparse (BM25) layer alone over all docs
        sparse_cands = pipeline.sparse.search(
            q_text,
            candidate_indices=None,
            top_k=top_k
        )
        if sparse_cands:
            raw = stats["per_layer_raw"]["sparse"]
            raw["n"] += 1
            if sparse_cands[0]["doc_index"] == gt_idx:
                raw["correct_top1"] += 1
            if any(c["doc_index"] == gt_idx for c in sparse_cands):
                raw["correct_topk"] += 1

        # 2.c) Dense layer alone over all docs (if available) - This section is now likely skipped
        # but kept for evaluation purposes if the flag is changed.
        if hasattr(pipeline, 'dense') and pipeline.dense is not None:
            dense_cands = pipeline.dense.search(
                q_text,
                candidate_indices=None,
                top_k=top_k
            )
            if dense_cands:
                raw = stats["per_layer_raw"]["dense"]
                raw["n"] += 1
                if dense_cands[0]["doc_index"] == gt_idx:
                    raw["correct_top1"] += 1
                if any(c["doc_index"] == gt_idx for c in dense_cands):
                    raw["correct_topk"] += 1

    return stats, hard_errors, per_query_results


def print_report(stats, hard_errors, top_k: int):
    """Pretty print a summary of the evaluation."""
    total = stats["total"]
    skipped = stats["skipped_no_label"]

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total labeled queries in file: {total + skipped}")
    print(f"Evaluated queries (label matched to a doc): {total}")
    print(f"Skipped (could not map label -> doc):      {skipped}")
    print()

    if total > 0:
        global_top1 = stats["correct_top1"] / total
        global_topk = stats["correct_topk"] / total
        print(f"Global Top-1 accuracy (pipeline): {global_top1:.3f}")
        print(f"Global Top-{top_k} accuracy (pipeline): {global_topk:.3f}")
    else:
        print("No queries evaluated (total=0).")
        return

    # --- Per-layer metrics for the pipeline (with gating) ---
    print("\n--- Per-layer accuracy (pipeline decisions) ---")
    for layer, s in stats["per_layer_pipeline"].items():
        if s["n"] == 0:
            continue
        acc1 = s["correct_top1"] / s["n"]
        accK = s["correct_topk"] / s["n"]
        print(f"\nLayer = {layer}")
        print(f"  Queries handled: {s['n']}")
        print(f"  Top-1 accuracy:  {acc1:.3f}")
        print(f"  Top-{top_k} accuracy: {accK:.3f}")

    # --- Raw layer metrics (layer alone, no gating) ---
    print("\n--- Raw per-layer accuracy (layer alone over all docs) ---")
    for layer, s in stats["per_layer_raw"].items():
        if s["n"] == 0:
            continue
        acc1 = s["correct_top1"] / s["n"]
        accK = s["correct_topk"] / s["n"]
        print(f"\nLayer = {layer}")
        print(f"  Queries evaluated: {s['n']}")
        print(f"  Top-1 accuracy:    {acc1:.3f}")
        print(f"  Top-{top_k} accuracy: {accK:.3f}")

    # --- Hard errors ---
    print("\n--- Hard errors (GT not in pipeline top-K) ---")
    print(f"Count: {len(hard_errors)}")
    if hard_errors:
        print("Examples:")
        for err in hard_errors[:5]:  # show only a few
            print("\n• Query:", err["query"])
            print("  Label title:", err["label_title"])
            print("  Layer used:", err["layer_used"])
            print("  Predicted (FR):", err["predicted_title_fr"])
            print("  Predicted (AR):", err["predicted_title_ar"])

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 3-layer retrieval pipeline on labeled queries"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing JSON documents (same as main_pipeline.py)",
    )

    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to labeled queries JSON file (e.g., query.JSON)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K to evaluate (default: 5)",
    )

    parser.add_argument(
        "--no-dense",
        action="store_true",
        help="Disable dense layer (for speed / ablation)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 1) Load documents
    logger.info(f"Loading documents from: {args.data_dir}")
    docs = load_documents(args.data_dir)

    if not docs:
        logger.error("No documents loaded!")
        return

    # 2) Init pipeline
    logger.info("Initializing retrieval pipeline...")
    pipeline = RetrievalPipeline(
        docs=docs,
        use_dense=not args.no_dense,
    )

    # 3) Load queries
    logger.info(f"Loading queries from: {args.queries}")
    queries = load_queries(args.queries)

    # 4) Run evaluation
    stats, hard_errors, per_query_results = evaluate_pipeline(
        docs=docs,
        pipeline=pipeline,
        queries=queries,
        top_k=args.top_k,
    )

    # 5) Print report
    print_report(stats, hard_errors, top_k=args.top_k)

    # 6) Persist results to a JSON file named '<queries_stem>_score.json'
    try:
        from datetime import datetime
        qpath = Path(args.queries)
        out_stem = f"{qpath.stem}_score"
        out_dir = qpath.parent or Path('.')
        out_name = f"{out_stem}.json"
        out_path = out_dir / out_name

        # Avoid overwriting an existing file: append timestamp if exists
        if out_path.exists():
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            out_path = out_dir / f"{out_stem}_{ts}.json"

        # Build summary statistics from per-query results
        times = [r["time_ms"] for r in per_query_results]
        total_q = len(per_query_results)
        avg_time = sum(times) / total_q if total_q else 0.0
        min_time = min(times) if times else 0.0
        max_time = max(times) if times else 0.0
        queries_under_0_2 = sum(1 for t in times if t <= 200.0)

        # Global hit@k and mrr from per-query results
        hit1 = sum(1 for r in per_query_results if r.get("hit@1"))
        hit3 = sum(1 for r in per_query_results if r.get("hit@3"))
        hit5 = sum(1 for r in per_query_results if r.get("hit@5"))
        mrr_avg = sum(r.get("mrr", 0.0) for r in per_query_results) / total_q if total_q else 0.0

        summary = {
            "total_queries": total_q,
            "avg_query_time_ms": avg_time,
            "max_query_time_ms": max_time,
            "min_query_time_ms": min_time,
            "queries_under_0.2s": queries_under_0_2,
            "hit@1": hit1 / total_q if total_q else 0.0,
            "hit@3": hit3 / total_q if total_q else 0.0,
            "hit@5": hit5 / total_q if total_q else 0.0,
            "mrr": mrr_avg,
        }

        payload = {
            "summary": summary,
            "results": per_query_results,
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"Wrote evaluation results to: {out_path}")
    except Exception as e:
        logger.error(f"Failed to write evaluation results file: {e}")


if __name__ == "__main__":
    main()