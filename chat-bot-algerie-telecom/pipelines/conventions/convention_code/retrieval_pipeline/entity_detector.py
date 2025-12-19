"""
Entity Detector - ÉTAPE 1.2
Détection des établissements et application du HARD FILTER.
Si un établissement est explicitement mentionné → filtrage dur, pas d'injection.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass


def normalize_accents(text: str) -> str:
    """Remove accents from text for accent-insensitive matching."""
    # Normalize to NFD form (separates base characters from diacritics)
    nfkd_form = unicodedata.normalize('NFD', text)
    # Remove all combining characters (accents)
    return ''.join(c for c in nfkd_form if not unicodedata.combining(c))


@dataclass
class EntityDetectionResult:
    """Résultat de la détection d'entités."""
    detected_entities: List[str]  # Codes des entités détectées
    is_explicit: bool  # True si mention explicite dans la query
    confidence: float
    matched_patterns: List[str]
    apply_hard_filter: bool  # True si on doit appliquer un filtre dur


class EntityDetector:
    """
    Détecte les établissements mentionnés dans une requête.
    Applique un HARD FILTER quand l'établissement est explicite.
    """
    
    # Mapping complet des établissements
    # Format: code -> [patterns de reconnaissance]
    ENTITY_PATTERNS = {
        "P": [
            r"établissement\s+P\b",
            r"etab\s*=?\s*P\b",
            r"\bP\b.*établissement",
            r"convention.*\bP\b",
            r"\bl['']établissement\s+P\b",
        ],
        "V": [
            r"établissement\s+V\b",
            r"etab\s*=?\s*V\b",
            r"\bV\b.*établissement",
            r"\bl['']établissement\s+V\b",
        ],
        "F": [
            r"établissement\s+F\b",
            r"etab\s*=?\s*F\b",
            r"\bl['']établissement\s+F\b",
            r"\bÉtablissement\s+F\b",
        ],
        "A": [
            r"établissement\s+A\b",
            r"etab\s*=?\s*A\b",
            r"\bl['']établissement\s+A\b",
        ],
        "N": [
            r"établissement\s+N\b",
            r"etab\s*=?\s*N\b",
            r"\bl['']établissement\s+N\b",
        ],
        "O": [
            r"établissement\s+O\b",
            r"etab\s*=?\s*O\b",
            r"\bl['']établissement\s+O\b",
        ],
        "I": [
            r"établissement\s+I\b",
            r"etab\s*=?\s*I\b",
            r"\bl['']établissement\s+I\b",
            r"\bÉtablissement\s+I\b",
        ],
        "AD": [
            r"établissement\s+AD\b",
            r"etab\s*=?\s*AD\b",
            r"\bl['']établissement\s+AD\b",
        ],
        "AC": [
            r"établissement\s+AC\b",
            r"etab\s*=?\s*AC\b",
            r"\bl['']établissement\s+AC\b",
        ],
        "AY": [
            r"établissement\s+AY\b",
            r"etab\s*=?\s*AY\b",
            r"\bl['']établissement\s+AY\b",
        ],
        "E": [
            r"établissement\s+E\b",
            r"etab\s*=?\s*E\b",
            r"\bl['']établissement\s+E\b",
        ],
        "H": [
            r"établissement\s+H\b",
            r"etab\s*=?\s*H\b",
            r"\bl['']établissement\s+H\b",
        ],
        "J": [
            r"établissement\s+J\b",
            r"etab\s*=?\s*J\b",
            r"\bl['']établissement\s+J\b",
        ],
        "K": [
            r"établissement\s+K\b",
            r"etab\s*=?\s*K\b",
            r"\bl['']établissement\s+K\b",
        ],
        "L": [
            r"établissement\s+L\b",
            r"etab\s*=?\s*L\b",
            r"\bl['']établissement\s+L\b",
        ],
        "M": [
            r"établissement\s+M\b",
            r"etab\s*=?\s*M\b",
            r"\bl['']établissement\s+M\b",
        ],
        "Q": [
            r"établissement\s+Q\b",
            r"etab\s*=?\s*Q\b",
            r"\bl['']établissement\s+Q\b",
        ],
        "R": [
            r"établissement\s+R\b",
            r"etab\s*=?\s*R\b",
            r"\bl['']établissement\s+R\b",
        ],
        "S": [
            r"établissement\s+S\b",
            r"etab\s*=?\s*S\b",
            r"\bl['']établissement\s+S\b",
        ],
        "T": [
            r"établissement\s+T\b",
            r"etab\s*=?\s*T\b",
            r"\bl['']établissement\s+T\b",
        ],
        "U": [
            r"établissement\s+U\b",
            r"etab\s*=?\s*U\b",
            r"\bl['']établissement\s+U\b",
        ],
        "W": [
            r"établissement\s+W\b",
            r"etab\s*=?\s*W\b",
            r"\bl['']établissement\s+W\b",
        ],
        "X": [
            r"établissement\s+X\b",
            r"etab\s*=?\s*X\b",
            r"\bl['']établissement\s+X\b",
        ],
    }
    
    # Patterns génériques qui indiquent une mention d'établissement
    GENERIC_ENTITY_PATTERNS = [
        r"pour\s+(?:l[''])?établissement\s+(\w+)",
        r"de\s+(?:l[''])?établissement\s+(\w+)",
        r"convention\s+(?:de\s+)?(?:l[''])?établissement\s+(\w+)",
        r"(?:l[''])?établissement\s+(\w+)",
        r"\[Etab\s*=\s*(\w+)\]",
    ]
    
    # Patterns d'exclusivité (indiquent que SEUL cet établissement est concerné)
    EXCLUSIVITY_PATTERNS = [
        r"uniquement\s+(?:pour\s+)?(?:l[''])?établissement",
        r"seulement\s+(?:pour\s+)?(?:l[''])?établissement",
        r"exclusivement\s+(?:pour\s+)?(?:l[''])?établissement",
        r"spécifiquement\s+(?:pour\s+)?(?:l[''])?établissement",
        r"pour\s+(?:l[''])?établissement\s+\w+\s+uniquement",
    ]
    
    def __init__(self, apply_hard_filter_by_default: bool = True):
        """
        Args:
            apply_hard_filter_by_default: Si True, applique le hard filter
                                          dès qu'un établissement est détecté.
        """
        self.apply_hard_filter_by_default = apply_hard_filter_by_default
        
        # Compile tous les patterns
        self.compiled_patterns = {
            code: [re.compile(p, re.IGNORECASE) for p in patterns]
            for code, patterns in self.ENTITY_PATTERNS.items()
        }
        
        self.compiled_generic = [
            re.compile(p, re.IGNORECASE) for p in self.GENERIC_ENTITY_PATTERNS
        ]
        
        self.compiled_exclusivity = [
            re.compile(p, re.IGNORECASE) for p in self.EXCLUSIVITY_PATTERNS
        ]
    
    def detect(self, query: str) -> EntityDetectionResult:
        """
        Détecte les établissements mentionnés dans une requête.
        
        Args:
            query: La requête utilisateur
            
        Returns:
            EntityDetectionResult avec les entités détectées
        """
        detected_entities: Set[str] = set()
        matched_patterns: List[str] = []
        
        # Normalize query for accent-insensitive matching
        normalized_query = normalize_accents(query)
        
        # 1. Recherche avec les patterns spécifiques par code
        for code, patterns in self.ENTITY_PATTERNS.items():
            for pattern_str in patterns:
                # Normalize pattern for accent-insensitive matching
                normalized_pattern = normalize_accents(pattern_str)
                pattern = re.compile(normalized_pattern, re.IGNORECASE)
                if pattern.search(normalized_query):
                    detected_entities.add(code)
                    matched_patterns.append(pattern_str)
                    break  # Un seul match suffit par code
        
        # 2. Recherche avec les patterns génériques
        for pattern_str in self.GENERIC_ENTITY_PATTERNS:
            normalized_pattern = normalize_accents(pattern_str)
            pattern = re.compile(normalized_pattern, re.IGNORECASE)
            matches = pattern.findall(normalized_query)
            for match in matches:
                # Vérifie si le match correspond à un code connu
                code = match.upper()
                if code in self.ENTITY_PATTERNS:
                    detected_entities.add(code)
                    matched_patterns.append(f"generic: {match}")
        
        # 3. Détermine si c'est une mention explicite
        is_explicit = len(detected_entities) > 0 and len(matched_patterns) > 0
        
        # 4. Vérifie les patterns d'exclusivité (also with accent normalization)
        has_exclusivity = any(
            re.compile(normalize_accents(pattern_str), re.IGNORECASE).search(normalized_query) 
            for pattern_str in self.EXCLUSIVITY_PATTERNS
        )
        
        # 5. Calcule la confiance
        if len(detected_entities) == 0:
            confidence = 0.0
        elif len(detected_entities) == 1:
            confidence = 0.95 if has_exclusivity else 0.85
        else:
            # Plusieurs entités détectées → moins sûr
            confidence = 0.6
        
        # 6. Détermine si on applique le hard filter
        apply_hard_filter = (
            is_explicit and 
            len(detected_entities) == 1 and
            self.apply_hard_filter_by_default
        )
        
        return EntityDetectionResult(
            detected_entities=list(detected_entities),
            is_explicit=is_explicit,
            confidence=confidence,
            matched_patterns=matched_patterns,
            apply_hard_filter=apply_hard_filter
        )
    
    def filter_passages(
        self, 
        passages: List[Dict], 
        entity_result: EntityDetectionResult
    ) -> List[Dict]:
        """
        Applique le hard filter sur les passages si nécessaire.
        
        Args:
            passages: Liste de passages (dicts avec 'entity_code')
            entity_result: Résultat de la détection
            
        Returns:
            Liste filtrée de passages
        """
        if not entity_result.apply_hard_filter:
            return passages
        
        if not entity_result.detected_entities:
            return passages
        
        # Hard filter: ne garde que les passages de l'entité détectée
        target_entity = entity_result.detected_entities[0]
        
        filtered = [
            p for p in passages 
            if p.get("entity_code", "").upper() == target_entity.upper()
        ]
        
        return filtered
    
    def get_entity_boost(
        self, 
        passage_entity: str, 
        query_entities: List[str]
    ) -> float:
        """
        Calcule un boost de score basé sur la correspondance d'entité.
        
        Returns:
            Multiplicateur de score (1.0 = pas de boost)
        """
        if not query_entities:
            return 1.0
        
        passage_entity = passage_entity.upper()
        query_entities = [e.upper() for e in query_entities]
        
        if passage_entity in query_entities:
            return 2.0  # Boost significatif
        else:
            return 0.5  # Pénalité si l'entité ne match pas
    
    @staticmethod
    def get_all_entity_codes() -> List[str]:
        """Retourne tous les codes d'établissements connus."""
        return list(EntityDetector.ENTITY_PATTERNS.keys())


class EntityFilter:
    """
    Filtre les résultats basé sur l'entité détectée.
    Utilisé dans le pipeline principal.
    """
    
    def __init__(self):
        self.detector = EntityDetector()
    
    def apply(
        self, 
        query: str, 
        passages: List[Dict],
        force_filter: bool = False
    ) -> Tuple[List[Dict], EntityDetectionResult]:
        """
        Détecte l'entité et filtre les passages si nécessaire.
        
        Args:
            query: Requête utilisateur
            passages: Liste de passages à filtrer
            force_filter: Si True, force le filtre même sans détection explicite
            
        Returns:
            (passages_filtrés, résultat_détection)
        """
        result = self.detector.detect(query)
        
        if force_filter and result.detected_entities:
            result.apply_hard_filter = True
        
        filtered_passages = self.detector.filter_passages(passages, result)
        
        return filtered_passages, result


# Tests
if __name__ == "__main__":
    detector = EntityDetector()
    
    test_queries = [
        "Quel est le prix pour l'établissement P ?",
        "Convention de l'établissement V",
        "Offres disponibles pour l'établissement AD",
        "Documents requis établissement AC",
        "Tarif fibre 1.5 Gbps",  # Pas d'entité
        "Offres pour les retraités de l'établissement F",
        "Établissement I: quels documents ?",
        "[Etab=N] Tarif ADSL",
    ]
    
    print("=== Test Entity Detector ===\n")
    
    for query in test_queries:
        result = detector.detect(query)
        
        print(f"Query: {query}")
        print(f"  → Entities: {result.detected_entities}")
        print(f"  → Explicit: {result.is_explicit}")
        print(f"  → Hard Filter: {result.apply_hard_filter}")
        print(f"  → Confidence: {result.confidence:.2f}")
        print()
    
    # Test de filtrage
    print("=== Test Filtering ===\n")
    
    mock_passages = [
        {"id": "1", "entity_code": "P", "text": "Passage établissement P"},
        {"id": "2", "entity_code": "V", "text": "Passage établissement V"},
        {"id": "3", "entity_code": "P", "text": "Autre passage P"},
        {"id": "4", "entity_code": "A", "text": "Passage établissement A"},
    ]
    
    query = "Offres pour l'établissement P"
    result = detector.detect(query)
    filtered = detector.filter_passages(mock_passages, result)
    
    print(f"Query: {query}")
    print(f"Original: {len(mock_passages)} passages")
    print(f"Filtered: {len(filtered)} passages")
    for p in filtered:
        print(f"  - {p['id']}: {p['entity_code']}")
