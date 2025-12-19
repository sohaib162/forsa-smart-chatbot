"""
Main retrieval pipeline orchestrator.

This module implements the 2-layer retrieval pipeline:
1. Rule-based routing (metadata matching)  -> used as a filter
2. Sparse retrieval (BM25 + Rule Score Fusion) -> final decision maker (with dynamic Top-N)
"""

from typing import List, Dict
import logging

from .loader import safe_get, safe_get_list

# Import all layers
from .rule_router import RuleRouter
from .sparse_index import SparseIndex

# Cross-Encoder abandoned; replaced with placeholder
CROSS_ENCODER_AVAILABLE = False
CrossEncoderRanker = None


logger = logging.getLogger(__name__)


def build_llm_context(doc: Dict, query: str = "") -> str:
    """
    Build a compact context string from a document for the chatbot LLM.
    (This function remains unchanged)
    """
    parts = []

    # Document ID and type
    doc_id = doc.get('document_id', 'Unknown')
    doc_type = doc.get('doc_type', 'document')
    parts.append(f"[Document ID: {doc_id}]")
    parts.append(f"[Type: {doc_type}]")
    parts.append("")

    # Title
    title_fr = safe_get(doc, 'metadata', 'title_fr')
    title_ar = safe_get(doc, 'metadata', 'title_ar')

    if title_fr:
        parts.append(f"Titre: {title_fr}")
    if title_ar:
        parts.append(f"العنوان: {title_ar}")
    if title_fr or title_ar:
        parts.append("")

    # Metadata
    company = safe_get(doc, 'metadata', 'company')
    version = safe_get(doc, 'metadata', 'version')
    effective_date = safe_get(doc, 'metadata', 'effective_date')

    if company or version or effective_date:
        parts.append("Informations:")
        if company:
            parts.append(f"  - Entreprise: {company}")
        if version:
            parts.append(f"  - Version: {version}")
        if effective_date:
            parts.append(f"  - Date d'effet: {effective_date}")
        parts.append("")

    # Product information
    product_family = doc.get('product_family')
    technology = safe_get_list(doc, 'technology')
    customer_segment = safe_get_list(doc, 'customer_segment')
    commitment_type = doc.get('commitment_type')

    if product_family or technology or customer_segment or commitment_type:
        parts.append("Caractéristiques:")
        if product_family:
            parts.append(f"  - Famille de produit: {product_family}")
        if technology:
            parts.append(f"  - Technologies: {', '.join(technology)}")
        if customer_segment:
            parts.append(f"  - Segments clients: {', '.join(customer_segment)}")
        if commitment_type:
            parts.append(f"  - Type d'engagement: {commitment_type}")
        parts.append("")

    # Offer core (if available)
    if doc.get('offer_core'):
        offer = doc['offer_core']

        name_fr = offer.get('name_fr')
        name_ar = offer.get('name_ar')

        if name_fr or name_ar:
            parts.append("Nom de l'offre:")
            if name_fr:
                parts.append(f"  FR: {name_fr}")
            if name_ar:
                parts.append(f"  AR: {name_ar}")
            parts.append("")

        desc_fr = offer.get('description_fr')
        desc_ar = offer.get('description_ar')

        if desc_fr:
            parts.append("Description (FR):")
            parts.append(f"  {desc_fr}")
            parts.append("")
        if desc_ar:
            parts.append("Description (AR):")
            parts.append(f"  {desc_ar}")
            parts.append("")

        # Conditions
        conditions_fr = safe_get_list(offer, 'conditions_fr')
        conditions_ar = safe_get_list(offer, 'conditions_ar')

        if conditions_fr:
            parts.append("Conditions (FR):")
            for i, cond in enumerate(conditions_fr, 1):
                parts.append(f"  {i}. {cond}")
            parts.append("")

        if conditions_ar:
            parts.append("Conditions (AR):")
            for i, cond in enumerate(conditions_ar, 1):
                parts.append(f"  {i}. {cond}")
            parts.append("")

        # Benefits
        benefits_fr = safe_get_list(offer, 'benefits_fr')
        benefits_ar = safe_get_list(offer, 'benefits_ar')

        if benefits_fr:
            parts.append("Avantages (FR):")
            for i, benefit in enumerate(benefits_fr, 1):
                parts.append(f"  {i}. {benefit}")
            parts.append("")

        if benefits_ar:
            parts.append("Avantages (AR):")
            for i, benefit in enumerate(benefits_ar, 1):
                parts.append(f"  {i}. {benefit}")
            parts.append("")

        # Pricing
        pricing_fr = offer.get('pricing_summary_fr')
        pricing_ar = offer.get('pricing_summary_ar')

        if pricing_fr:
            parts.append("Tarification (FR):")
            parts.append(f"  {pricing_fr}")
            parts.append("")

        if pricing_ar:
            parts.append("Tarification (AR):")
            parts.append(f"  {pricing_ar}")
            parts.append("")

    # Policy summary (for policy documents)
    if doc.get('policy_summary'):
        policy = doc['policy_summary']
        summary_fr = policy.get('summary_fr')
        summary_ar = policy.get('summary_ar')

        if summary_fr:
            parts.append("Résumé de la politique (FR):")
            parts.append(f"  {summary_fr}")
            parts.append("")
        if summary_ar:
            parts.append("Résumé de la politique (AR):")
            parts.append(f"  {summary_ar}")
            parts.append("")

    # FAQ (include top 3 most relevant)
    faq_fr = safe_get_list(doc, 'faq_fr')
    faq_ar = safe_get_list(doc, 'faq_ar')

    if faq_fr:
        parts.append("Questions Fréquentes (FR):")
        for i, faq in enumerate(faq_fr[:3], 1):
            q = faq.get('question', '')
            a = faq.get('answer', '')
            if q and a:
                parts.append(f"  Q{i}: {q}")
                parts.append(f"  R{i}: {a}")
        parts.append("")

    if faq_ar:
        parts.append("Questions Fréquentes (AR):")
        for i, faq in enumerate(faq_ar[:3], 1):
            q = faq.get('question', '')
            a = faq.get('answer', '')
            if q and a:
                parts.append(f"  س{i}: {q}")
                parts.append(f"  ج{i}: {a}")
        parts.append("")

    # Contact information
    if doc.get('contact'):
        contact = doc['contact']
        phone = contact.get('phone')
        email = contact.get('email')
        notes_fr = contact.get('notes_fr')
        notes_ar = contact.get('notes_ar')

        if phone or email or notes_fr or notes_ar:
            parts.append("Contact:")
            if phone:
                parts.append(f"  - Téléphone: {phone}")
            if email:
                parts.append(f"  - Email: {email}")
            if notes_fr:
                parts.append(f"  - Notes: {notes_fr}")
            if notes_ar:
                parts.append(f"  - ملاحظات: {notes_ar}")
            parts.append("")

    return "\n".join(parts)


class RetrievalPipeline:
    """
    2-layer retrieval pipeline orchestrator: Rule-based Filtering -> Sparse Fusion Ranking.
    The output includes the dynamic Top-N based on score margins and the full JSON.
    """

    def __init__(
        self,
        docs: List[Dict],
        use_dense: bool = False 
    ):
        """
        Initialize the retrieval pipeline.
        """
        self.docs = docs
        
        # Initialize layers
        logger.info("Building rule-based router (for filtering)...")
        self.router = RuleRouter(docs) 

        logger.info("Building sparse (BM25) index...")
        self.sparse = SparseIndex(docs)
        
        # Cross-Encoder abandoned
        self.cross_encoder = None 

        logger.info("Retrieval pipeline initialized successfully (Sparse Fusion only).")

    def search(self, query: str, top_k: int = 5) -> Dict:
        """
        Execute the 2-layer retrieval pipeline: Rule Filter -> Sparse Fusion.
        The top_k here refers to the MAX number of results.
        """
        logger.info(f"Processing query: {query[:100]}...")

        # ---------------- Stage 1: Rule-based Filtering ----------------
        logger.info("Stage 1: Rule-based filtering")
        rule_result = self.router.filter_candidates(query)
        rule_candidates = rule_result['candidates']
        
        rule_scores = {c['doc_index']: c['score'] for c in rule_candidates}
        candidate_indices = [c['doc_index'] for c in rule_candidates]

        if not candidate_indices:
            candidate_indices = None
            logger.warning("Rule router found no candidates: Sparse search over all documents")

        # ---------------- Stage 2: Sparse Fusion (Final Ranking) ----------------
        # Use Sparse Fusion to generate candidates. 
        CANDIDATE_POOL_SIZE = max(top_k * 2, 5) 
        
        logger.info(f"Stage 2: Sparse Fusion for Final Ranking (Candidate Pool Size: {CANDIDATE_POOL_SIZE})")

        final_candidates = self.sparse.search_fused(
            query=query,
            rule_scores=rule_scores,
            candidate_indices=candidate_indices,
            top_k=CANDIDATE_POOL_SIZE
        )
        
        layer_used = "sparse_fused"
        
        if not final_candidates:
            logger.warning("Sparse Fusion found no candidates.")
            return self._build_result(query=query, layer_used=layer_used, candidates=[], top_k=top_k)

        # --- Dynamic Top-N selection is now handled inside _build_result ---
        return self._build_result(
            query=query,
            layer_used=layer_used, 
            candidates=final_candidates, # Pass all candidates to allow dynamic comparison
            top_k=top_k # Pass max allowed K
        )

    def _build_result(
        self,
        query: str,
        layer_used: str,
        candidates: List[Dict],
        top_k: int
    ) -> Dict:
        """
        Build the final result dictionary.

        Dynamically selects Top-N based on score margin.
        The output includes the *entire* document JSON.
        """
        if not candidates:
            logger.warning("No candidates found!")
            return {
                "query": query,
                "layer_used": layer_used,
                "best_document_summary": None,
                "candidates_summary": [],
                "llm_context": "No relevant document found.",
                "retrieved_documents": []
            }
        
        # --- Dynamic Top-N Selection Logic ---
        final_top_k = 1 # Start by returning at least the Top-1
        
        if len(candidates) >= 2:
            score1 = candidates[0]['score']
            score2 = candidates[1]['score']
            
            # Since the scores are normalized (0 to 1) in Sparse Fusion,
            # we use a small margin (0.05 for 5% relative closeness)
            # Relative margin: (Score1 - Score2) / Score1. Low score1 values can be noisy, so we cap max K at 3.
            
            MARGIN = 0.05
            
            if score1 > 0.0 and (score1 - score2) / score1 < MARGIN:
                final_top_k = 2 # Scores 1 and 2 are too close
                
                if len(candidates) >= 3:
                    score3 = candidates[2]['score']
                    
                    if (score2 > 0.0) and (score2 - score3) / score2 < MARGIN:
                        final_top_k = 3 # Scores 2 and 3 are also too close
                        
        # Ensure final_top_k doesn't exceed the actual number of candidates or the requested max_k
        final_top_k = min(final_top_k, len(candidates), top_k)
        
        logger.info(f"Dynamic Top-N: Returned {final_top_k} results based on score margins.")
        
        selected_candidates = candidates[:final_top_k]
        best_index = selected_candidates[0]['doc_index']
        best_doc = self.docs[best_index]
        
        # --- Prepare Output for Full JSON Retrieval ---
        
        retrieved_documents = []
        candidate_info_summary = []

        for cand in selected_candidates:
            idx = cand['doc_index']
            doc = self.docs[idx]
            
            # Retrieve the full document JSON
            retrieved_documents.append({
                "doc_index": idx,
                "score": cand['score'],
                "layer": layer_used,
                "title_fr": safe_get(doc, 'metadata', 'title_fr'),
                "title_ar": safe_get(doc, 'metadata', 'title_ar'),
                "full_document_json": doc # <--- Includes entire document JSON
            })
            
            # Also maintain a summary of the candidates for logging/easy view
            candidate_info_summary.append({
                "doc_index": idx,
                "score": cand['score'],
                "document_id": doc.get('document_id'),
                "title_fr": safe_get(doc, 'metadata', 'title_fr'),
                "title_ar": safe_get(doc, 'metadata', 'title_ar'),
                "product_family": doc.get('product_family'),
                "doc_type": doc.get('doc_type')
            })


        # Build LLM context from the TOP-1 document
        llm_context = build_llm_context(best_doc, query)

        result = {
            "query": query,
            "layer_used": layer_used,
            "best_document_summary": candidate_info_summary[0] if candidate_info_summary else None, 
            "candidates_summary": candidate_info_summary, # Summaries of all returned candidates
            "llm_context": llm_context,
            "retrieved_documents": retrieved_documents # <--- The full JSON objects
        }

        logger.info(
            f"Pipeline complete: layer={layer_used}, "
            f"best_doc={best_doc.get('document_id')}, "
            f"returned_N={final_top_k}"
        )

        return result