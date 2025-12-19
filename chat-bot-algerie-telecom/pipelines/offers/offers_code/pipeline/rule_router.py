"""
Rule-based routing layer for document retrieval.

This module implements the first layer of the retrieval pipeline using
metadata-based routing. It now acts purely as a filter, passing all
scored candidates to the Sparse layer for final ranking fusion.
"""

import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import logging

from .loader import safe_get, safe_get_list
from .text_normalization import normalize_text_multilingual, tokenize_multilingual, detect_language
from .bilingual_synonyms import find_cross_language_matches, get_synonyms

logger = logging.getLogger(__name__)


class RuleRouter:
    """
    Multilingual rule-based document router using metadata and tags.

    Now acts as an intelligent filter and scorer for the Sparse Fusion layer.
    """

    def __init__(self, docs: List[Dict]):
        """
        Initialize the rule-based router.
        """
        self.docs = docs

        # Build routing index with multilingual tokens
        self.routing_index = self._build_routing_index()

        logger.info(f"RuleRouter initialized with {len(docs)} documents (multilingual)")

    def _build_routing_index(self) -> List[Dict[str, any]]:
        """
        Build a multilingual routing index for each document.
        """
        routing_index = []

        for idx, doc in enumerate(self.docs):
            tokens = set()
            raw_tokens = set()

            # Doc type
            doc_type = doc.get('doc_type')
            if doc_type:
                norm = normalize_text_multilingual(doc_type)
                tokens.add(norm)
                raw_tokens.add(norm)

            # Product family
            product_family = doc.get('product_family')
            if product_family:
                norm = normalize_text_multilingual(product_family)
                tokens.add(norm)
                raw_tokens.add(norm)
                tokens.update(norm.split('_'))

            # Technology
            technologies = safe_get_list(doc, 'technology')
            for tech in technologies:
                if tech:
                    norm = normalize_text_multilingual(tech)
                    tokens.add(norm)
                    raw_tokens.add(norm)
                    tokens.update(norm.split('_'))

            # Customer segment
            segments = safe_get_list(doc, 'customer_segment')
            for segment in segments:
                if segment:
                    norm = normalize_text_multilingual(segment)
                    tokens.add(norm)
                    raw_tokens.add(norm)
                    tokens.update(norm.split('_'))

            # Commitment type
            commitment = doc.get('commitment_type')
            if commitment:
                norm = normalize_text_multilingual(commitment)
                tokens.add(norm)
                raw_tokens.add(norm)
                if commitment == "no_commitment":
                    tokens.update(["sans", "engagement", "sans engagement", "بدون", "التزام"])

            # Usage focus
            usage_focus = safe_get_list(doc, 'usage_focus')
            for focus in usage_focus:
                if focus:
                    norm = normalize_text_multilingual(focus)
                    tokens.add(norm)
                    tokens.update(norm.split('_'))

            # Keywords
            keywords_fr = safe_get_list(doc, 'keywords_fr')
            keywords_ar = safe_get_list(doc, 'keywords_ar')
            keywords = safe_get_list(doc, 'keywords')

            for keyword in keywords_fr + keywords_ar + keywords:
                if keyword:
                    norm_kw = normalize_text_multilingual(keyword)
                    tokens.add(norm_kw)
                    raw_tokens.add(norm_kw)
                    word_tokens = tokenize_multilingual(norm_kw)
                    tokens.update(word_tokens)

            # Titles
            title_fr = safe_get(doc, 'metadata', 'title_fr')
            title_ar = safe_get(doc, 'metadata', 'title_ar')

            if title_fr:
                norm_title = normalize_text_multilingual(title_fr)
                words = tokenize_multilingual(norm_title)
                words = [w for w in words if len(w) > 3]
                tokens.update(words)
                raw_tokens.update(words)

            if title_ar:
                norm_title = normalize_text_multilingual(title_ar)
                words = tokenize_multilingual(norm_title)
                words = [w for w in words if len(w) > 2]
                tokens.update(words)
                raw_tokens.update(words)

            # Expand tokens with bilingual synonyms
            synonym_tokens = set()
            for token in list(raw_tokens)[:50]:
                syns = get_synonyms(token, max_synonyms=3)
                synonym_tokens.update(syns)

            tokens.update(synonym_tokens)

            # Store routing metadata
            routing_index.append({
                'doc_index': idx,
                'tokens': tokens,
                'doc_type': doc_type,
                'product_family': product_family,
                'customer_segment': segments,
                'technology': technologies,
                'commitment_type': commitment
            })

        return routing_index

    # --- Intent/Segment detection methods (unchanged, high boosts retained) ---

    def _detect_paiement_electronique(self, query_str: str, query_tokens: Set[str]) -> Tuple[bool, float]:
        payment_terms_fr = {'paiement', 'payment', 'payement', 'carte', 'eddahabia', 'cib', 'edahabia', 'bancaire', 'électronique', 'avantage'}
        payment_terms_ar = {'دفع', 'تسديد', 'إلكتروني', 'بطاقة', 'الذهبية', 'رقمي'}
        fr_matches = len(query_tokens.intersection(payment_terms_fr))
        ar_matches = len(query_tokens.intersection(payment_terms_ar))
        total_matches = fr_matches + ar_matches
        if total_matches >= 2:
            return True, 25.0
        elif total_matches == 1:
            return False, 8.0
        return False, 0.0

    def _detect_ont_equipment(self, query_str: str, query_tokens: Set[str]) -> Tuple[bool, float]:
        ont_terms_fr = {'ont', 'modem', 'routeur', 'wifi', 'wi-fi', 'wifi 6', 'xgs-pon', 'xgs', 'gpon', 'préférentiel', 'preferentiel', 'tarif'}
        ont_terms_ar = {'اونت', 'مودم', 'راوتر', 'واي فاي', 'وايفاي', 'تفضيلي', 'سعر'}
        fr_matches = len(query_tokens.intersection(ont_terms_fr))
        ar_matches = len(query_tokens.intersection(ont_terms_ar))
        wifi6_pattern = any(term in query_str for term in ['wifi 6', 'واي فاي 6', 'xgs-pon', 'xgs'])
        ont_pattern = any(term in query_str for term in ['ont', 'اونت'])
        preferentiel_pattern = any(term in query_str for term in ['préférentiel', 'preferentiel', 'تفضيلي'])
        if ont_pattern and (wifi6_pattern or preferentiel_pattern):
            return True, 30.0
        elif (fr_matches + ar_matches) >= 2:
            return True, 22.0
        elif ont_pattern or wifi6_pattern:
            return False, 12.0
        return False, 0.0

    def _detect_ecoles_primaires(self, query_str: str, query_tokens: Set[str]) -> Tuple[bool, float]:
        school_terms_fr = {'école', 'ecole', 'écoles', 'ecoles', 'primaire', 'primaires', 'scolaire', 'établissement', 'etablissement'}
        school_terms_ar = {'مدرسة', 'مدارس', 'ابتدائي', 'ابتدائية', 'مدرسي', 'تعليمي', 'مؤسسة'}
        fr_matches = len(query_tokens.intersection(school_terms_fr))
        ar_matches = len(query_tokens.intersection(school_terms_ar))
        school_pattern = any(term in query_str for term in ['école', 'ecole', 'écoles', 'ecoles', 'مدرسة', 'مدارس'])
        primaire_pattern = any(term in query_str for term in ['primaire', 'ابتدائي', 'ابتدائية'])
        if school_pattern and primaire_pattern:
            return True, 30.0
        elif school_pattern or (fr_matches + ar_matches) >= 2:
            return True, 22.0
        return False, 0.0

    def _detect_4g_lte_sans_engagement(self, query_str: str, query_tokens: Set[str]) -> Tuple[bool, float]:
        lte_terms_fr = {'4g', 'lte', '4g lte', 'sans engagement', 'sans', 'engagement'}
        lte_terms_ar = {'4جي', 'لتي', 'الجيل الرابع', 'بدون التزام', 'بدون', 'التزام'}
        fr_matches = len(query_tokens.intersection(lte_terms_fr))
        ar_matches = len(query_tokens.intersection(lte_terms_ar))
        lte_pattern = any(term in query_str for term in ['4g', 'lte', '4g lte', 'لتي', '4جي', 'الجيل الرابع'])
        no_commitment_pattern = any(term in query_str for term in ['sans engagement', 'بدون التزام'])
        if lte_pattern and no_commitment_pattern:
            return True, 32.0
        elif lte_pattern and any(term in query_tokens for term in ['sans', 'بدون']):
            return True, 25.0
        elif lte_pattern:
            return False, 15.0
        return False, 0.0

    def _detect_parrainage(self, query_str: str, query_tokens: Set[str]) -> Tuple[bool, float]:
        parrainage_terms_fr = {'parrainage', 'parrainer', 'parrain', 'sponsoring', 'référer', 'filleul'}
        parrainage_terms_ar = {'إحالة', 'رعاية', 'راعي', 'مُحال'}
        fr_matches = len(query_tokens.intersection(parrainage_terms_fr))
        ar_matches = len(query_tokens.intersection(parrainage_terms_ar))
        if (fr_matches + ar_matches) >= 1:
            return True, 30.0
        return False, 0.0

    def _detect_segment_business(self, query_str: str, query_tokens: Set[str]) -> float:
        business_terms_fr = {'business', 'entreprise', 'professionnel', 'société', 'pme', 'medium business', 'moyen', 'pmi', 'tpe'}
        business_terms_ar = {'أعمال', 'محترف', 'مهني', 'شركة', 'مؤسسة', 'صغيرة', 'متوسطة'}
        fr_matches = len(query_tokens.intersection(business_terms_fr))
        ar_matches = len(query_tokens.intersection(business_terms_ar))
        medium_pattern = any(term in query_str for term in ['medium business', 'medium', 'moyen', 'متوسطة', 'صغيرة'])
        if medium_pattern and (fr_matches + ar_matches) >= 1:
            return 20.0
        elif (fr_matches + ar_matches) >= 2:
            return 15.0
        elif (fr_matches + ar_matches) >= 1:
            return 8.0
        return 0.0

    def _detect_segment_locataire(self, query_str: str, query_tokens: Set[str]) -> float:
        locataire_terms_fr = {'locataire', 'resident', 'résident', 'location', 'sans facture', 'sans justificatif'}
        locataire_terms_ar = {'مستأجر', 'ساكن', 'كاري', 'بدون فاتورة', 'بدون إثبات'}
        fr_matches = len(query_tokens.intersection(locataire_terms_fr))
        ar_matches = len(query_tokens.intersection(locataire_terms_ar))
        if (fr_matches + ar_matches) >= 2:
            return 20.0
        if (fr_matches + ar_matches) >= 1:
            return 15.0
        return 0.0

    def _detect_segment_residentiel(self, query_str: str, query_tokens: Set[str]) -> float:
        residentiel_terms_fr = {'résidentiel', 'residentiel', 'particulier', 'domicile', 'maison', 'appartement'}
        residentiel_terms_ar = {'سكني', 'منزلي', 'خواص'}
        fr_matches = len(query_tokens.intersection(residentiel_terms_fr))
        ar_matches = len(query_tokens.intersection(residentiel_terms_ar))
        if (fr_matches + ar_matches) >= 2:
            return 15.0
        if (fr_matches + ar_matches) >= 1:
            return 10.0
        return 0.0

    def _detect_segment_gamers(self, query_str: str, query_tokens: Set[str]) -> float:
        gamer_terms_fr = {'gamer', 'gamers', 'gaming', 'jeux', 'joueur', 'ps5', 'xbox'}
        gamer_terms_ar = {'جيمر', 'ألعاب', 'لاعب', 'بلايستيشن', 'اكس بوكس'}
        fr_matches = len(query_tokens.intersection(gamer_terms_fr))
        ar_matches = len(query_tokens.intersection(gamer_terms_ar))
        if (fr_matches + ar_matches) >= 1:
            return 22.0
        return 0.0
    
    # --- Score Document Logic (remains unchanged, focuses on high metadata score) ---

    def _score_document(self, doc_routing: Dict, query_tokens: Set[str], query_str: str) -> Tuple[float, Dict]:
        """
        Score a document based on how many routing tokens match the query.
        """
        score = 0.0
        doc_tokens = doc_routing['tokens']
        intent_signals = {}

        # Detect intents
        has_paiement, paiement_boost = self._detect_paiement_electronique(query_str, query_tokens)
        has_ont, ont_boost = self._detect_ont_equipment(query_str, query_tokens)
        has_ecoles, ecoles_boost = self._detect_ecoles_primaires(query_str, query_tokens)
        has_4g, lte_boost = self._detect_4g_lte_sans_engagement(query_str, query_tokens)
        has_parrainage, parrainage_boost = self._detect_parrainage(query_str, query_tokens)

        # Detect segments
        business_boost = self._detect_segment_business(query_str, query_tokens)
        locataire_boost = self._detect_segment_locataire(query_str, query_tokens)
        residentiel_boost = self._detect_segment_residentiel(query_str, query_tokens)
        gamers_boost = self._detect_segment_gamers(query_str, query_tokens)

        intent_signals = {
            'has_paiement': has_paiement, 'has_ont': has_ont, 'has_ecoles': has_ecoles,
            'has_4g_lte': has_4g, 'has_parrainage': has_parrainage,
            'segment_business': business_boost > 0, 'segment_locataire': locataire_boost > 0,
            'segment_residentiel': residentiel_boost > 0, 'segment_gamers': gamers_boost > 0,
        }

        # Count matching tokens (direct matches)
        matching_tokens = query_tokens.intersection(doc_tokens)
        score += len(matching_tokens) * 1.5

        # Cross-language matching using synonyms
        query_tokens_list = list(query_tokens)
        doc_tokens_list = list(doc_tokens)
        cross_matches = find_cross_language_matches(query_tokens_list, doc_tokens_list)
        score += cross_matches * 1.0

        # Apply strong intent boosting
        product_family = doc_routing['product_family'] or ''
        customer_segments = doc_routing['customer_segment'] or []
        doc_type = doc_routing['doc_type'] or ''

        # Paiement électronique intent
        if has_paiement:
            if 'payment' in product_family.lower() or doc_type == 'payment_benefits':
                score += paiement_boost
            else:
                score -= 10.0
        elif paiement_boost > 0:
            if 'payment' in product_family.lower():
                score += paiement_boost

        # ONT equipment intent
        if has_ont:
            if 'ont' in product_family.lower() or 'wifi_6' in product_family.lower() or 'xgs-pon' in product_family.lower():
                score += ont_boost
            else:
                score -= 12.0
        elif ont_boost > 0:
            if 'ont' in product_family.lower():
                score += ont_boost

        # Écoles primaires intent
        if has_ecoles:
            if any('school' in seg for seg in customer_segments) or any('primary' in seg for seg in customer_segments):
                score += ecoles_boost
            else:
                score -= 20.0
        elif ecoles_boost > 0:
            if any('school' in seg for seg in customer_segments):
                score += ecoles_boost

        # 4G LTE sans engagement intent
        if has_4g:
            if '4g_lte' in product_family.lower() and doc_routing['commitment_type'] == 'no_commitment':
                score += lte_boost
            elif '4g_lte' in product_family.lower():
                score += lte_boost * 0.7
            else:
                score -= 15.0
        elif lte_boost > 0:
            if '4g_lte' in product_family.lower():
                score += lte_boost

        # Parrainage intent
        if has_parrainage:
            if 'referral' in product_family.lower() or 'parrainage' in product_family.lower():
                score += parrainage_boost
            else:
                score -= 20.0

        # Apply segment boosting with mutual exclusivity
        # Business segment
        if business_boost > 0:
            if any('business' in seg for seg in customer_segments) or any('enterprise' in seg for seg in customer_segments):
                score += business_boost
                if any('locataire' in seg for seg in customer_segments) and not ('moohtarif' in product_family.lower() or 'medium_business' in product_family.lower()):
                    score -= 10.0
                if any('residential' in seg for seg in customer_segments) and not any('business' in seg for seg in customer_segments):
                    score -= 12.0
            else:
                score -= 10.0

        # Locataire segment
        if locataire_boost > 0:
            if any('locataire' in seg for seg in customer_segments):
                score += locataire_boost
                if any('business' in seg for seg in customer_segments) and not ('moohtarif' in product_family.lower() or 'medium_business' in product_family.lower()):
                    score -= 8.0
                if gamers_boost == 0 and 'gamer' in product_family.lower():
                    score -= 10.0
            else:
                score -= 12.0

        # Gamers segment
        if gamers_boost > 0:
            if 'gamer' in product_family.lower():
                score += gamers_boost
                if locataire_boost > 0 and any('locataire' in seg for seg in customer_segments):
                    score += 8.0
            else:
                score -= 15.0

        # Residential segment (less exclusive)
        if residentiel_boost > 0:
            if any('residential' in seg for seg in customer_segments):
                score += residentiel_boost
                if any('business' in seg for seg in customer_segments):
                    score -= 10.0

        # Boost for doc_type match (with multilingual normalization)
        if doc_routing['doc_type']:
            doc_type_norm = normalize_text_multilingual(doc_routing['doc_type'])
            if doc_type_norm in query_tokens:
                score += 5.0
            doc_type_syns = set(get_synonyms(doc_type_norm))
            if query_tokens.intersection(doc_type_syns):
                score += 3.0

        # Boost for product_family match
        if doc_routing['product_family']:
            pf_norm = normalize_text_multilingual(doc_routing['product_family'])
            if pf_norm in query_tokens:
                score += 5.0
            pf_syns = set(get_synonyms(pf_norm))
            if query_tokens.intersection(pf_syns):
                score += 3.0

        # Boost for customer segment match
        for segment in doc_routing['customer_segment']:
            seg_norm = normalize_text_multilingual(segment)
            if seg_norm in query_tokens:
                score += 3.0
            seg_syns = set(get_synonyms(seg_norm))
            if query_tokens.intersection(seg_syns):
                score += 2.0

        # Boost for technology match
        for tech in doc_routing['technology']:
            tech_norm = normalize_text_multilingual(tech)
            if tech_norm in query_tokens:
                score += 2.0
            tech_syns = set(get_synonyms(tech_norm))
            if query_tokens.intersection(tech_syns):
                score += 1.5

        # Tax / fiscal pattern
        if any(term in query_str for term in ['timbre', 'fiscal', 'taxe', 'طابع', 'جبائي', 'قانون', 'المالية', 'ضريبة']):
            if doc_routing['doc_type'] == 'tax_policy':
                score += 15.0
            if doc_routing['product_family'] == 'tax_stamp':
                score += 15.0

        # Modernisation / migration pattern
        if any(term in query_str for term in ['modernisation', 'migration', 'basculement', 'عصرنة', 'تحديث', 'تحويل']):
            if doc_routing['product_family'] and 'modernisation' in doc_routing['product_family']:
                score += 10.0

        # 4G LTE to Fibre migration / modernisation pattern
        if any(term in query_str for term in ['4g', 'lte', 'adsl', 'vdsl']) and any(term in query_str for term in ['vers', 'fibre', 'modernisation', 'migration', 'الألياف']):
            if doc_routing['product_family'] and 'modernisation' in str(doc_routing['product_family']).lower():
                score += 12.0

        # Interruption / service quality issues
        if any(term in query_str for term in ['interruption', 'coupure', 'ralenti', 'lent', 'انقطاع', 'بطء']):
            if doc_routing['doc_type'] in ['technical', 'service_quality']:
                score += 8.0

        return score, intent_signals


    def filter_candidates(self, query: str) -> Dict:
        """
        Routes a query to candidates, returning a large pool of scored documents.
        """
        # Normalize and tokenize query with multilingual support
        norm_query = normalize_text_multilingual(query)
        query_tokens = set(tokenize_multilingual(norm_query))
        query_tokens.add(norm_query)

        # Expand query tokens with synonyms
        expanded_tokens = set()
        for token in list(query_tokens)[:20]:
            syns = get_synonyms(token, max_synonyms=3)
            expanded_tokens.update(syns)
        all_query_tokens = query_tokens.union(expanded_tokens)

        # Score all documents
        scores = []
        for doc_routing in self.routing_index:
            score, _ = self._score_document(doc_routing, all_query_tokens, norm_query)
            if score > 0: # Only include docs with non-zero score
                scores.append({
                    'doc_index': doc_routing['doc_index'],
                    'score': score
                })

        # Sort by score (descending)
        scores.sort(key=lambda x: x['score'], reverse=True)

        # Return a large pool of candidates (e.g., top 15)
        CANDIDATE_POOL_SIZE = 15
        
        if scores:
            logger.info(f"Rule filtering complete: returned {len(scores)} scored candidates (Top 15 kept)")

        return {
            'candidates': scores[:CANDIDATE_POOL_SIZE],
        }