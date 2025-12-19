"""
Intent Classifier - ÉTAPE 1.1
Classification de la requête en UNE catégorie dominante.
Utilisé pour le GATING et le choix des poids hybrid scoring.
"""

import re
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class Intent(Enum):
    """Les 5 intents possibles pour le gating."""
    PRICE = "PRICE"
    SPEED = "SPEED"
    DOCUMENTS = "DOCUMENTS"
    BENEFICIARY = "BENEFICIARY"
    GENERAL = "GENERAL"


@dataclass
class IntentResult:
    """Résultat de la classification d'intent."""
    primary_intent: Intent
    confidence: float
    all_scores: Dict[Intent, float]
    detected_triggers: List[str]


class IntentClassifier:
    """
    Classifie une requête en une catégorie unique.
    Règles explicites (pas de ML) pour garantir la transparence.
    """
    
    # Déclencheurs par intent (triés par priorité)
    TRIGGERS = {
        Intent.PRICE: [
            # Termes exacts de prix
            r'\bprix\b', r'\btarif\b', r'\bcoût\b', r'\bcout\b',
            r'\bda\b', r'\bdinars?\b', r'\bgratuit\b',
            r'\bremise\b', r'\bréduction\b', r'\breduction\b',
            r'\bcombien\b.*(?:coûte|coute|paye|payer)',
            r'\bmoins cher\b', r'\bplus cher\b',
            r'\boffre.*(?:à|a)\s*\d+', r'\d+\s*(?:da|DA)',
            # Prix spécifiques
            r'\b800\b', r'\b1100\b', r'\b1\s*100\b', r'\b1200\b',
            r'\b1300\b', r'\b1500\b', r'\b2000\b', r'\b3500\b',
        ],
        Intent.SPEED: [
            # Termes de débit
            r'\bdébit\b', r'\bdebit\b', r'\bvitesse\b',
            r'\bmbps\b', r'\bgbps\b', r'\bmbit\b', r'\bgbit\b',
            r'\brapide\b', r'\blent\b', r'\bconnexion\b',
            # Débits spécifiques
            r'\b20\s*mbps\b', r'\b50\s*mbps\b', r'\b100\s*mbps\b',
            r'\b300\s*mbps\b', r'\b500\s*mbps\b',
            r'\b1\.?5?\s*gbps\b', r'\b1\.?2\s*gbps\b', r'\b1\s*gbps\b',
            # Types d'offres liés au débit
            r'\bfibre\b', r'\badsl\b', r'\bvdsl\b', r'\bftth\b',
        ],
        Intent.DOCUMENTS: [
            # Documents administratifs
            r'\bdocuments?\b', r'\bdossier\b', r'\bpièces?\b',
            r'\bpapiers?\b', r'\bjustificatifs?\b',
            r'\battestation\b', r'\bcertificat\b',
            r'\bcarte\s*(?:nationale|professionnelle|identité)\b',
            r'\bcni\b', r'\bidentité\b',
            # Actions documentaires
            r'\bfournir\b', r'\bprésenter\b', r'\bapporter\b',
            r'\bbesoin\s*de\s*quoi\b', r'\bquels?\s*documents?\b',
            r'\bqu\'est-ce\s*qu\'il\s*faut\b',
            # Documents spécifiques au contexte
            r'\bbon\s*d\'ouverture\b', r'\baction\s*sociale\b',
            r'\battesté\b', r'\bsigné\b',
        ],
        Intent.BENEFICIARY: [
            # Types de bénéficiaires
            r'\bretraités?\b', r'\bretraite\b',
            r'\bcadres?\s*supérieurs?\b', r'\bcadres?\s*superieurs?\b',
            r'\bpersonnel\b', r'\bemployés?\b', r'\bsalariés?\b',
            r'\bactifs?\b', r'\ben\s*activité\b',
            r'\bayants?\s*droit\b', r'\bfamille\b',
            # Questions sur l'éligibilité
            r'\béligible\b', r'\beligible\b', r'\béligibilité\b',
            r'\bqui\s*peut\b', r'\bpour\s*qui\b',
            r'\bbénéficiaires?\b', r'\bbeneficiaires?\b',
            # Catégories spécifiques
            r'\btous\s*les\s*employés\b', r'\btout\s*le\s*personnel\b',
        ],
        Intent.GENERAL: [
            # Requêtes générales
            r'\bconvention\b', r'\baccord\b', r'\bpartenariat\b',
            r'\bétablissement\b', r'\betablissement\b',
            r'\boffres?\b', r'\bservices?\b',
            r'\binformation\b', r'\brenseignement\b',
            r'\bcomment\b', r'\bpourquoi\b',
        ]
    }
    
    # Poids pour résoudre les conflits (plus haut = plus prioritaire)
    INTENT_PRIORITY = {
        Intent.PRICE: 5,
        Intent.SPEED: 4,
        Intent.DOCUMENTS: 3,
        Intent.BENEFICIARY: 2,
        Intent.GENERAL: 1,
    }
    
    # Patterns de négation à ignorer
    NEGATION_PATTERNS = [
        r'pas\s+de\s+prix', r'sans\s+prix',
        r'pas\s+de\s+documents?', r'sans\s+documents?',
    ]
    
    def __init__(self, use_priority_weights: bool = True):
        """
        Args:
            use_priority_weights: Si True, utilise les poids de priorité
                                  pour résoudre les conflits d'intent.
        """
        self.use_priority_weights = use_priority_weights
        
        # Compile les patterns
        self.compiled_triggers = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.TRIGGERS.items()
        }
        self.compiled_negations = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.NEGATION_PATTERNS
        ]
    
    def _count_triggers(self, query: str, intent: Intent) -> Tuple[int, List[str]]:
        """Compte le nombre de triggers détectés pour un intent."""
        count = 0
        detected = []
        
        for pattern in self.compiled_triggers[intent]:
            matches = pattern.findall(query)
            if matches:
                count += len(matches)
                detected.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
                
        return count, detected
    
    def _has_negation(self, query: str) -> bool:
        """Vérifie si la requête contient une négation pertinente."""
        for pattern in self.compiled_negations:
            if pattern.search(query):
                return True
        return False
    
    def _compute_scores(self, query: str) -> Dict[Intent, float]:
        """Calcule les scores bruts pour chaque intent."""
        scores = {}
        
        for intent in Intent:
            count, _ = self._count_triggers(query, intent)
            
            # Score de base = nombre de triggers
            base_score = count
            
            # Bonus de priorité si activé
            if self.use_priority_weights and count > 0:
                priority_bonus = self.INTENT_PRIORITY[intent] * 0.1
                base_score += priority_bonus
            
            scores[intent] = base_score
            
        return scores
    
    def classify(self, query: str) -> IntentResult:
        """
        Classifie une requête en un intent dominant.
        
        Args:
            query: La requête utilisateur
            
        Returns:
            IntentResult avec l'intent primaire et les scores
        """
        # Calcule les scores
        scores = self._compute_scores(query)
        
        # Trouve l'intent dominant
        max_score = max(scores.values())
        
        if max_score == 0:
            # Aucun trigger → GENERAL par défaut
            primary_intent = Intent.GENERAL
            confidence = 0.5
            detected_triggers = []
        else:
            # Trouve tous les intents avec le score max
            top_intents = [intent for intent, score in scores.items() if score == max_score]
            
            if len(top_intents) == 1:
                primary_intent = top_intents[0]
            else:
                # En cas d'égalité, utilise la priorité
                primary_intent = max(top_intents, key=lambda i: self.INTENT_PRIORITY[i])
            
            # Calcule la confiance (normalise par le total)
            total = sum(scores.values())
            confidence = max_score / total if total > 0 else 0.5
            
            # Récupère les triggers détectés
            _, detected_triggers = self._count_triggers(query, primary_intent)
        
        # Normalise les scores pour le retour
        total = sum(scores.values()) or 1
        normalized_scores = {intent: score / total for intent, score in scores.items()}
        
        return IntentResult(
            primary_intent=primary_intent,
            confidence=confidence,
            all_scores=normalized_scores,
            detected_triggers=detected_triggers
        )
    
    def classify_with_explanation(self, query: str) -> Dict:
        """
        Classifie avec une explication détaillée.
        Utile pour le debug et l'évaluation.
        """
        result = self.classify(query)
        
        # Collecte tous les triggers pour chaque intent
        all_triggers = {}
        for intent in Intent:
            _, triggers = self._count_triggers(query, intent)
            all_triggers[intent.value] = triggers
        
        return {
            "query": query,
            "primary_intent": result.primary_intent.value,
            "confidence": result.confidence,
            "scores": {k.value: v for k, v in result.all_scores.items()},
            "triggers_by_intent": all_triggers,
            "detected_triggers": result.detected_triggers,
        }


# Hybrid scoring weights par intent (ÉTAPE 3.1)
HYBRID_WEIGHTS = {
    Intent.PRICE: {"dense": 0.2, "sparse": 0.8},
    Intent.SPEED: {"dense": 0.3, "sparse": 0.7},
    Intent.DOCUMENTS: {"dense": 0.1, "sparse": 0.9},
    Intent.BENEFICIARY: {"dense": 0.6, "sparse": 0.4},
    Intent.GENERAL: {"dense": 0.7, "sparse": 0.3},
}


def get_hybrid_weights(intent: Intent) -> Tuple[float, float]:
    """
    Retourne les poids (α dense, β sparse) pour un intent donné.
    
    Returns:
        (dense_weight, sparse_weight)
    """
    weights = HYBRID_WEIGHTS.get(intent, HYBRID_WEIGHTS[Intent.GENERAL])
    return weights["dense"], weights["sparse"]


# Tests
if __name__ == "__main__":
    classifier = IntentClassifier()
    
    test_queries = [
        # PRICE
        "Quel est le prix de l'offre fibre pour les retraités ?",
        "Combien coûte l'abonnement 1.5 Gbps ?",
        "Tarif de l'ADSL à 800 DA",
        "Offre gratuite pour cadres supérieurs",
        
        # SPEED
        "Quelle est la vitesse de l'offre fibre ?",
        "Je veux 1.5 Gbps pour l'établissement P",
        "Débit 100 Mbps VDSL",
        
        # DOCUMENTS
        "Quels documents fournir pour souscrire ?",
        "J'ai besoin de quels papiers ?",
        "Attestation de travail requise ?",
        "Bon d'ouverture de droit action sociale",
        
        # BENEFICIARY
        "Les retraités ont-ils droit à une réduction ?",
        "Offres pour les cadres supérieurs",
        "Personnel actif éligible ?",
        "Qui peut bénéficier de la convention ?",
        
        # GENERAL
        "Convention établissement P",
        "Quelles sont les offres disponibles ?",
    ]
    
    print("=== Test Intent Classifier ===\n")
    
    for query in test_queries:
        result = classifier.classify(query)
        weights = get_hybrid_weights(result.primary_intent)
        
        print(f"Query: {query}")
        print(f"  → Intent: {result.primary_intent.value} (conf: {result.confidence:.2f})")
        print(f"  → Weights: dense={weights[0]}, sparse={weights[1]}")
        print(f"  → Triggers: {result.detected_triggers[:3]}")
        print()
