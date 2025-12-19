"""
Sparse retrieval layer using BM25.
"""

from typing import List, Dict, Optional
import logging
from rank_bm25 import BM25Okapi

from .loader import safe_get_list, safe_get
from .text_normalization import normalize_text_multilingual, tokenize_multilingual
from .bilingual_synonyms import expand_query_with_synonyms

logger = logging.getLogger(__name__)


class SparseIndex:
    """
    BM25-based sparse retrieval index with multilingual support and score fusion.
    """

    def __init__(self, docs: List[Dict], keyword_boost: int = 3, use_query_expansion: bool = True):
        """
        Initialize the BM25 sparse index.
        """
        self.docs = docs
        self.keyword_boost = keyword_boost
        self.use_query_expansion = use_query_expansion

        self.corpus_texts = self._build_corpus_texts()

        self.tokenized_corpus = [
            self._tokenize(text) for text in self.corpus_texts
        ]

        # BM25 index with b=0.5 (to reduce document length penalty)
        self.bm25 = BM25Okapi(self.tokenized_corpus, b=0.5)

        logger.info(f"SparseIndex initialized with {len(docs)} documents, "
                   f"query_expansion={'enabled' if use_query_expansion else 'disabled'} (BM25 b=0.5)")

    def _build_corpus_texts(self) -> List[str]:
        """
        Build rich corpus texts for BM25 indexing with multilingual support.
        """
        corpus_texts = []

        # --- High Boost Factor for Differentiating Terms ---
        HIGH_BOOST_FACTOR = 20
        # --------------------------------------------------

        for doc in self.docs:
            parts = []
            
            # Highly Boost Differentiating Metadata
            discriminating_terms = []

            pf = doc.get('product_family')
            if pf:
                discriminating_terms.append(pf.replace('_', ' ')) 
            dt = doc.get('doc_type')
            if dt:
                discriminating_terms.append(dt.replace('_', ' '))

            segments = safe_get_list(doc, 'customer_segment')
            for seg in segments:
                discriminating_terms.append(seg.replace('_', ' '))

            techs = safe_get_list(doc, 'technology')
            for tech in techs:
                discriminating_terms.append(tech.replace('_', ' '))
                
            comm = doc.get('commitment_type')
            if comm:
                discriminating_terms.append(comm.replace('_', ' '))

            if discriminating_terms:
                boosted_text = ' '.join(discriminating_terms)
                parts.append(boosted_text * HIGH_BOOST_FACTOR) 

            # Primary search texts
            search_text = doc.get('search_text', '')
            dense_text = doc.get('dense_text_primary', '')

            if search_text:
                parts.append(search_text)
            if dense_text and dense_text != search_text:
                parts.append(dense_text)

            # Titles (FR + AR) - heavily boost for better matching
            title_fr = safe_get(doc, 'metadata', 'title_fr')
            title_ar = safe_get(doc, 'metadata', 'title_ar')
            if title_fr:
                parts.append(title_fr * 3)
            if title_ar:
                parts.append(title_ar * 3)

            # Offer core (FR + AR)
            offer = doc.get('offer_core') or {}
            name_fr = offer.get('name_fr')
            name_ar = offer.get('name_ar')
            desc_fr = offer.get('description_fr')
            desc_ar = offer.get('description_ar')

            if name_fr:
                parts.append(name_fr * 3)
            if name_ar:
                parts.append(name_ar * 3)
            if desc_fr:
                parts.append(desc_fr)
            if desc_ar:
                parts.append(desc_ar)

            # Conditions (FR + AR)
            conditions_fr = safe_get_list(offer, 'conditions_fr')
            conditions_ar = safe_get_list(offer, 'conditions_ar')
            if conditions_fr:
                parts.append(' '.join(conditions_fr))
            if conditions_ar:
                parts.append(' '.join(conditions_ar))

            # Benefits (FR + AR)
            benefits_fr = safe_get_list(offer, 'benefits_fr')
            benefits_ar = safe_get_list(offer, 'benefits_ar')
            if benefits_fr:
                parts.append(' '.join(benefits_fr))
            if benefits_ar:
                parts.append(' '.join(benefits_ar))

            # Pricing (FR + AR)
            pricing_fr = offer.get('pricing_summary_fr')
            pricing_ar = offer.get('pricing_summary_ar')
            if pricing_fr:
                parts.append(pricing_fr)
            if pricing_ar:
                parts.append(pricing_ar)

            # FAQ (FR + AR) - questions and answers
            faq_fr = safe_get_list(doc, 'faq_fr')
            faq_ar = safe_get_list(doc, 'faq_ar')

            for faq in faq_fr:
                q = faq.get('question', '')
                a = faq.get('answer', '')
                if q:
                    parts.append(q)
                if a:
                    parts.append(a)

            for faq in faq_ar:
                q = faq.get('question', '')
                a = faq.get('answer', '')
                if q:
                    parts.append(q)
                if a:
                    parts.append(a)

            # Keywords (boosted)
            keywords = safe_get_list(doc, 'keywords')
            keywords_fr = safe_get_list(doc, 'keywords_fr')
            keywords_ar = safe_get_list(doc, 'keywords_ar')

            all_keywords = keywords + keywords_fr + keywords_ar

            if all_keywords:
                # Repeat keywords to boost their importance
                keyword_text = ' '.join(all_keywords) * self.keyword_boost
                parts.append(keyword_text)

            # Fallback if no content
            if not parts:
                parts.append(self._build_fallback_text(doc))

            corpus_text = ' '.join(parts)
            corpus_texts.append(corpus_text)

        return corpus_texts

    def _build_fallback_text(self, doc: Dict) -> str:
        """
        Build search text from document fields if search_text is missing.
        """
        parts = []

        # Titles
        if doc.get('metadata'):
            title_fr = doc['metadata'].get('title_fr')
            title_ar = doc['metadata'].get('title_ar')
            if title_fr:
                parts.append(title_fr)
            if title_ar:
                parts.append(title_ar)

        # Offer core
        if doc.get('offer_core'):
            offer = doc['offer_core']
            for field in ['name_fr', 'name_ar', 'description_fr', 'description_ar']:
                value = offer.get(field)
                if value:
                    parts.append(value)

        # Policy summary
        if doc.get('policy_summary'):
            policy = doc['policy_summary']
            for field in ['summary_fr', 'summary_ar']:
                value = policy.get(field)
                if value:
                    parts.append(value)

        return ' '.join(parts)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 with multilingual support.
        """
        # Normalize text (handles both FR and AR)
        normalized = normalize_text_multilingual(text)

        # Tokenize (preserves both Latin and Arabic tokens)
        tokens = tokenize_multilingual(normalized)

        return tokens

    def search(
        self,
        query: str,
        candidate_indices: Optional[List[int]] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search the BM25 index (Legacy - kept for raw evaluation).
        """
        # Tokenize query with multilingual support
        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.warning("Empty query after tokenization")
            return []

        # Optionally expand query with bilingual synonyms
        if self.use_query_expansion:
            expanded_tokens = expand_query_with_synonyms(query_tokens, max_expansions=2)
            query_tokens = expanded_tokens

        # Get BM25 scores
        if candidate_indices is None:
            scores = self.bm25.get_scores(query_tokens)
            indices = list(range(len(self.docs)))
        else:
            scores = self.bm25.get_scores(query_tokens)
            indices = candidate_indices

        # Create (index, score) pairs for candidates
        scored_docs = [
            {"doc_index": idx, "score": float(scores[idx])}
            for idx in indices
        ]

        scored_docs.sort(key=lambda x: x['score'], reverse=True)

        results = scored_docs[:top_k]

        if results:
            logger.info(f"BM25 (Raw) search: Found {len(scored_docs)} results, "
                       f"top score: {results[0]['score']:.4f}")
        else:
            logger.info("BM25 (Raw) search: No results")

        return results
    
    def search_fused(
        self,
        query: str,
        rule_scores: Dict[int, float], # Rule score map from RuleRouter
        candidate_indices: Optional[List[int]] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search the BM25 index and fuse results with the Rule score for final ranking.
        """
        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.warning("Empty query after tokenization in Fused Search")
            return []

        if self.use_query_expansion:
            expanded_tokens = expand_query_with_synonyms(query_tokens, max_expansions=2)
            query_tokens = expanded_tokens

        # 1. Get Raw BM25 scores for all documents (or candidates)
        if candidate_indices is None:
            bm25_raw_scores = self.bm25.get_scores(query_tokens)
            indices = list(range(len(self.docs)))
        else:
            bm25_raw_scores = self.bm25.get_scores(query_tokens)
            indices = candidate_indices

        # Filter indices to only include candidates from the rule layer (if provided)
        if rule_scores:
            indices = [idx for idx in indices if idx in rule_scores]
        
        if not indices:
            logger.warning("Fused Search: No candidates remaining after rule filter.")
            return []

        # 2. Prepare scores for normalization
        bm25_scores = {idx: bm25_raw_scores[idx] for idx in indices}
        rule_scores_filtered = {idx: rule_scores.get(idx, 0.0) for idx in indices}

        # 3. Normalize scores (Min-Max)
        bm25_vals = list(bm25_scores.values())
        bm25_max = max(bm25_vals) if bm25_vals else 1.0
        bm25_min = min(bm25_vals) if bm25_vals else 0.0
        bm25_range = max(bm25_max - bm25_min, 1e-6)

        rule_vals = list(rule_scores_filtered.values())
        rule_max = max(rule_vals) if rule_vals else 1.0
        rule_min = min(rule_vals) if rule_vals else 0.0
        rule_range = max(rule_max - rule_min, 1e-6)

        # 4. Apply Fusion Formula
        # Rule score is now strongly favored (1.25x) to break close ties decisively.
        # This is a crucial reduction from 1.5 to balance the Locataire/Gamers issue.
        W_BM25_NORM = 1.0 
        W_RULE_NORM = 1.25 # <--- UPDATED: Reduced multiplier to 1.25

        combined = []
        for doc_idx in indices:
            bm25_raw = bm25_scores.get(doc_idx, 0.0)
            rule_raw = rule_scores_filtered.get(doc_idx, 0.0)

            # Normalize
            bm25_norm = (bm25_raw - bm25_min) / bm25_range
            rule_norm = (rule_raw - rule_min) / rule_range
            
            # Final Fused Score
            combined_score = (W_BM25_NORM * bm25_norm) + (W_RULE_NORM * rule_norm)
            
            # Add a strong penalty if the Rule score was zero, as a sanity check. 
            if rule_raw == 0.0:
                combined_score *= 0.7 
            
            combined.append({
                "doc_index": doc_idx,
                "score": float(combined_score)
            })

        # 5. Sort and return
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        results = combined[:top_k]
        
        if results:
            logger.info(f"Sparse Fused Search: Found {len(combined)} results, "
                       f"top score: {results[0]['score']:.4f}")
        
        return results