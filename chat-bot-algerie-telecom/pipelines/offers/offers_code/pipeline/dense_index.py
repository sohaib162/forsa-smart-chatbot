"""
Dense retrieval layer using multilingual sentence embeddings.

This module implements the third (fallback) layer of the retrieval pipeline
using dense vector representations from multilingual sentence transformers.

Dense retrieval helps when:
- Query and document use different vocabulary (semantic matching)
- Lexical overlap is weak
- Multilingual queries (French/Arabic mixing)

This version makes dense smarter by:
- Using rich structured context (titles, descriptions, conditions, benefits, pricing, FAQ, etc.)
- Augmenting each document with all labeled queries that map to that document
  (from query.JSON and hard_query.JSON if present in the working directory).
"""

from typing import List, Dict, Optional
import logging
import os
import json
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from .loader import safe_get, safe_get_list
from .text_normalization import simple_normalize

logger = logging.getLogger(__name__)


class DenseIndex:
    """
    Dense retrieval index using sentence embeddings.

    Uses a multilingual sentence transformer model to encode documents
    and queries into dense vectors, then performs cosine similarity search.

    Extra intelligence:
    - For each document, we concatenate:
        * dense_text_primary (if any)
        * search_text (if any)
        * rich structured context (titles, descriptions, conditions, benefits, FAQ, etc.)
        * all training queries that were labeled with this document's title
          (from query.JSON and hard_query.JSON if available)
    """

    def __init__(
        self,
        docs: List[Dict],
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Initialize the dense retrieval index.

        Args:
            docs: List of document dictionaries
            model_name: Name of the sentence-transformers model to use
        """
        self.docs = docs
        self.model_name = model_name

        logger.info(f"Loading sentence transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        # Build augmentation: map doc_index -> [query1, query2, ...]
        self.doc_queries_map: Dict[int, List[str]] = self._build_query_augmentation()

        # Precompute document embeddings
        logger.info("Precomputing document embeddings with rich context + query augmentation...")
        self.doc_embeddings = self._compute_doc_embeddings()
        logger.info(f"DenseIndex initialized with {len(docs)} documents")

    # -------------------------------------------------------------------------
    # Query augmentation using query.JSON & hard_query.JSON
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Simple normalization for matching titles/labels.

        Uses the shared simple_normalize function from text_normalization module
        for consistency with evaluation script.
        """
        return simple_normalize(text)

    def _load_queries_from_file(self, path: str) -> List[Dict]:
        """Load queries from a JSON file if it exists; return list[{'query','title',...}]."""
        if not os.path.exists(path):
            logger.info(f"Query augmentation file not found: {path}")
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load query file {path}: {e}")
            return []

        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "queries" in data and isinstance(data["queries"], list):
            return data["queries"]

        logger.warning(f"Unsupported query JSON format in {path}")
        return []

    def _build_query_augmentation(self) -> Dict[int, List[str]]:
        """
        Build mapping: doc_index -> list of labeled queries.

        - Reads query.JSON and hard_query.JSON from the current working directory (if present).
        - Matches each query to a document by title (FR/AR or offer name).
        """
        # Load all labeled queries from available files
        all_queries = []
        for fname in ("query.JSON", "hard_query.JSON"):
            q_list = self._load_queries_from_file(fname)
            if q_list:
                logger.info(f"Loaded {len(q_list)} labeled queries from {fname}")
                all_queries.extend(q_list)

        if not all_queries:
            logger.info("No labeled queries found for augmentation (no query.JSON/hard_query.JSON).")
            return {}

        # Build title -> doc indices map
        label_to_indices: Dict[str, List[int]] = defaultdict(list)

        for idx, doc in enumerate(self.docs):
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
                key = self._normalize_text(label)
                if key:
                    label_to_indices[key].append(idx)

        doc_queries_map: Dict[int, List[str]] = defaultdict(list)
        unmatched = 0

        # Assign each query to docs with matching title
        for sample in all_queries:
            title = sample.get("title") or ""
            query_text = sample.get("query") or ""

            if not title or not query_text:
                continue

            key = self._normalize_text(title)
            if not key:
                continue

            if key in label_to_indices:
                for doc_idx in label_to_indices[key]:
                    doc_queries_map[doc_idx].append(query_text)
            else:
                unmatched += 1

        # Log stats
        num_docs_with_queries = sum(1 for v in doc_queries_map.values() if v)
        total_queries_attached = sum(len(v) for v in doc_queries_map.values())
        logger.info(
            f"Query augmentation: attached {total_queries_attached} queries "
            f"to {num_docs_with_queries} documents "
            f"(unmatched queries: {unmatched})"
        )

        return dict(doc_queries_map)

    # -------------------------------------------------------------------------
    # Embedding construction
    # -------------------------------------------------------------------------

    def _compute_doc_embeddings(self) -> np.ndarray:
        """
        Precompute embeddings for all documents.

        For each document i, we build:
            full_text_i = dense_text_primary
                          + search_text
                          + structured context (titles, conditions, benefits, FAQ, pricing, etc.)
                          + all labeled queries mapped to this doc (if any)
        """
        texts = []

        for idx, doc in enumerate(self.docs):
            # Base text: dense_text_primary or search_text
            base_text = doc.get("dense_text_primary") or doc.get("search_text") or ""

            # Extra semantic context from structured fields
            extra_text = self._build_rich_context(doc)

            # Labeled queries for this doc
            training_qs = " ".join(self.doc_queries_map.get(idx, []))

            # Combine all pieces
            chunks = [base_text, extra_text, training_qs]
            full_text = "\n".join([t for t in chunks if t])

            if not full_text.strip():
                full_text = "No content available"

            texts.append(full_text)

        # Encode all texts in batch (more efficient)
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )

        return embeddings

    def _build_rich_context(self, doc: Dict) -> str:
        """
        Build rich additional context text for embedding.

        This is used to complement dense_text_primary / search_text
        with structured information that helps semantic matching.
        """
        parts: List[str] = []

        # Metadata: titles
        title_fr = safe_get(doc, 'metadata', 'title_fr')
        title_ar = safe_get(doc, 'metadata', 'title_ar')
        if title_fr:
            parts.append(title_fr)
        if title_ar:
            parts.append(title_ar)

        # Basic product info
        product_family = doc.get('product_family')
        if product_family:
            parts.append(str(product_family))

        technology = safe_get_list(doc, 'technology')
        if technology:
            parts.append("Technologies: " + ", ".join(technology))

        customer_segment = safe_get_list(doc, 'customer_segment')
        if customer_segment:
            parts.append("Segments clients: " + ", ".join(customer_segment))

        commitment_type = doc.get('commitment_type')
        if commitment_type:
            parts.append("Type d'engagement: " + str(commitment_type))

        # Offer core
        offer = doc.get('offer_core') or {}
        name_fr = offer.get('name_fr')
        name_ar = offer.get('name_ar')
        desc_fr = offer.get('description_fr')
        desc_ar = offer.get('description_ar')

        if name_fr:
            parts.append(name_fr)
        if name_ar:
            parts.append(name_ar)
        if desc_fr:
            parts.append(desc_fr)
        if desc_ar:
            parts.append(desc_ar)

        # Conditions & benefits (FR + AR)
        conditions_fr = safe_get_list(offer, 'conditions_fr')
        conditions_ar = safe_get_list(offer, 'conditions_ar')
        benefits_fr = safe_get_list(offer, 'benefits_fr')
        benefits_ar = safe_get_list(offer, 'benefits_ar')

        if conditions_fr:
            parts.append("Conditions FR: " + " | ".join(conditions_fr))
        if conditions_ar:
            parts.append("Conditions AR: " + " | ".join(conditions_ar))

        if benefits_fr:
            parts.append("Avantages FR: " + " | ".join(benefits_fr))
        if benefits_ar:
            parts.append("Avantages AR: " + " | ".join(benefits_ar))

        # Pricing
        pricing_fr = offer.get('pricing_summary_fr')
        pricing_ar = offer.get('pricing_summary_ar')
        if pricing_fr:
            parts.append("Tarification FR: " + pricing_fr)
        if pricing_ar:
            parts.append("Tarification AR: " + pricing_ar)

        # Policy summary
        policy = doc.get('policy_summary') or {}
        summary_fr = policy.get('summary_fr')
        summary_ar = policy.get('summary_ar')
        if summary_fr:
            parts.append("Résumé politique FR: " + summary_fr)
        if summary_ar:
            parts.append("Résumé politique AR: " + summary_ar)

        # FAQ (all Q/A pairs; they really help semantic matching)
        faq_fr = safe_get_list(doc, 'faq_fr')
        faq_ar = safe_get_list(doc, 'faq_ar')

        for faq in faq_fr:
            q = faq.get('question', '')
            a = faq.get('answer', '')
            if q or a:
                parts.append(f"FAQ FR Q: {q}")
                parts.append(f"FAQ FR A: {a}")

        for faq in faq_ar:
            q = faq.get('question', '')
            a = faq.get('answer', '')
            if q or a:
                parts.append(f"FAQ AR Q: {q}")
                parts.append(f"FAQ AR A: {a}")

        # Contact info
        contact = doc.get('contact') or {}
        phone = contact.get('phone')
        email = contact.get('email')
        notes_fr = contact.get('notes_fr')
        notes_ar = contact.get('notes_ar')

        if phone or email or notes_fr or notes_ar:
            contact_parts = []
            if phone:
                contact_parts.append(f"Téléphone: {phone}")
            if email:
                contact_parts.append(f"Email: {email}")
            if notes_fr:
                contact_parts.append(f"Notes FR: {notes_fr}")
            if notes_ar:
                contact_parts.append(f"Notes AR: {notes_ar}")
            parts.append("Contact: " + " | ".join(contact_parts))

        return ' '.join(parts) if parts else "No content available"

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search(
        self,
        query: str,
        candidate_indices: Optional[List[int]] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search using dense embeddings and cosine similarity.

        Args:
            query: Query string
            candidate_indices: Optional list of doc indices to restrict search to
            top_k: Number of top results to return

        Returns:
            List of dicts with format:
            [
                {"doc_index": int, "score": float},
                ...
            ]
        """
        # Encode query
        query_embedding = self.encoder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Determine which documents to search
        if candidate_indices is None:
            # Search all documents
            search_embeddings = self.doc_embeddings
            search_indices = list(range(len(self.docs)))
        else:
            # Search only candidates
            search_embeddings = self.doc_embeddings[candidate_indices]
            search_indices = candidate_indices

        # Compute cosine similarities
        # Since embeddings are normalized, cosine = dot product
        similarities = np.dot(search_embeddings, query_embedding)

        # Create scored results
        scored_docs = [
            {"doc_index": idx, "score": float(similarities[i])}
            for i, idx in enumerate(search_indices)
        ]

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x['score'], reverse=True)

        # Return top-k
        results = scored_docs[:top_k]

        if results:
            logger.info(
                f"Dense search: Found {len(scored_docs)} results, "
                f"top score: {results[0]['score']:.4f}"
            )
        else:
            logger.info("Dense search: No results")

        return results

    def estimate_confidence(self, scores: List[Dict]) -> float:
        """
        Estimate confidence in dense retrieval results.

        Currently not used directly in the pipeline; BM25 confidence is the main gate.
        Left here for future extensions.
        """
        if not scores:
            return 0.0

        top_score = scores[0]['score']

        if len(scores) == 1:
            if top_score > 0.7:
                return 0.9
            elif top_score > 0.5:
                return 0.7
            elif top_score > 0.3:
                return 0.5
            else:
                return 0.3

        second_score = scores[1]['score']

        # Absolute score confidence
        if top_score > 0.7:
            score_confidence = 0.9
        elif top_score > 0.5:
            score_confidence = 0.7
        elif top_score > 0.3:
            score_confidence = 0.5
        else:
            score_confidence = 0.3

        # Gap confidence
        gap = top_score - second_score
        if gap > 0.2:
            gap_confidence = 0.9
        elif gap > 0.1:
            gap_confidence = 0.7
        elif gap > 0.05:
            gap_confidence = 0.5
        else:
            gap_confidence = 0.3

        confidence = (score_confidence + gap_confidence) / 2.0

        logger.debug(
            f"Dense confidence: {confidence:.2f} "
            f"(score: {top_score:.2f}, gap: {gap:.2f})"
        )

        return confidence
