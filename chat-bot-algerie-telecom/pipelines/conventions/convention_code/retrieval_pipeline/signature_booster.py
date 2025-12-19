"""
Signature Booster - ÉTAPE 4
Boosting industrialisé basé sur les tokens de signature par établissement.
Remplace les règles manuelles par un dictionnaire automatique.
"""

import re
import math
from typing import List, Dict, Set, Any, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class SignatureMatch:
    """Représente un match de signature."""
    token: str
    idf_score: float
    entity_code: str


class SignatureBooster:
    """
    Booste les passages basés sur les tokens de signature.
    Construit automatiquement le dictionnaire de signatures par établissement.
    """
    
    # Tokens de signature communs (catégories de bénéficiaires)
    BASE_SIGNATURE_TOKENS = {
        "ayants droit",
        "action sociale",
        "cadres supérieurs",
        "retraités",
        "personnel actif",
        "bon d'ouverture de droit",
        "attestation de travail",
        "carte professionnelle",
        "responsable habilité",
        "ressources humaines",
    }
    
    def __init__(self):
        # Signature tokens par établissement
        self.entity_signatures: Dict[str, Set[str]] = defaultdict(set)
        
        # IDF des tokens (calculé sur tout le corpus)
        self.token_idf: Dict[str, float] = {}
        
        # Nombre total de documents
        self.total_docs = 0
        
        # Document frequency par token
        self.doc_freq: Dict[str, int] = defaultdict(int)
    
    def _tokenize_for_signature(self, text: str) -> List[str]:
        """
        Tokenize le texte en n-grams pour la détection de signatures.
        Retourne unigrams, bigrams et trigrams.
        """
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        tokens = []
        
        # Unigrams
        tokens.extend(words)
        
        # Bigrams
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]} {words[i+1]}")
        
        # Trigrams
        for i in range(len(words) - 2):
            tokens.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        return tokens
    
    def build_signatures(self, passages: List[Dict]):
        """
        Construit le dictionnaire de signatures à partir des passages.
        
        Args:
            passages: Liste de passages avec 'entity_code' et 'text'
        """
        self.total_docs = len(passages)
        
        # Compte les occurrences par établissement et calcule les doc frequencies
        entity_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for passage in passages:
            entity_code = passage.get("entity_code", "UNK")
            text = passage.get("text", "")
            
            tokens = self._tokenize_for_signature(text)
            unique_tokens = set(tokens)
            
            # Compte les tokens pour cet établissement
            for token in tokens:
                entity_token_counts[entity_code][token] += 1
            
            # Document frequency (pour IDF)
            for token in unique_tokens:
                self.doc_freq[token] += 1
            
            # Ajoute les tokens de signature existants
            if passage.get("signature_tokens"):
                for st in passage["signature_tokens"]:
                    self.entity_signatures[entity_code].add(st.lower())
        
        # Calcule l'IDF pour chaque token
        for token, df in self.doc_freq.items():
            self.token_idf[token] = math.log((self.total_docs + 1) / (df + 1)) + 1
        
        # Identifie les tokens de signature discriminants par établissement
        self._identify_discriminant_tokens(entity_token_counts)
    
    def _identify_discriminant_tokens(
        self, 
        entity_token_counts: Dict[str, Dict[str, int]]
    ):
        """
        Identifie les tokens qui sont discriminants pour chaque établissement.
        Un token est discriminant s'il apparaît fréquemment dans un établissement
        mais rarement dans les autres.
        """
        # Calcule la fréquence totale de chaque token
        global_token_counts: Dict[str, int] = defaultdict(int)
        for entity_counts in entity_token_counts.values():
            for token, count in entity_counts.items():
                global_token_counts[token] += count
        
        # Pour chaque établissement, trouve les tokens discriminants
        for entity_code, token_counts in entity_token_counts.items():
            for token, count in token_counts.items():
                # Skip les tokens très courts
                if len(token) < 4:
                    continue
                
                # Calcule le ratio local/global
                global_count = global_token_counts[token]
                if global_count < 3:
                    continue  # Trop rare pour être fiable
                
                local_ratio = count / sum(token_counts.values())
                global_ratio = count / global_count
                
                # Un token est discriminant s'il a une haute concentration locale
                if global_ratio > 0.6 and local_ratio > 0.01:
                    self.entity_signatures[entity_code].add(token)
    
    def add_base_signatures(self):
        """Ajoute les tokens de signature de base à tous les établissements."""
        # Ces tokens sont recherchés dans les requêtes
        for entity in self.entity_signatures.keys():
            # Pas besoin d'ajouter - on les utilise pour la recherche dans les requêtes
            pass
    
    def compute_boost(
        self, 
        passage: Dict, 
        query: str,
        entity_code_filter: Optional[str] = None
    ) -> float:
        """
        Calcule le boost de signature pour un passage.
        
        Args:
            passage: Le passage à évaluer
            query: La requête utilisateur
            entity_code_filter: Si spécifié, boost uniquement si l'entité match
            
        Returns:
            Multiplicateur de score (1.0 = pas de boost)
        """
        boost = 0.0
        
        passage_entity = passage.get("entity_code", "UNK")
        passage_text = passage.get("text", "").lower()
        passage_signatures = set(s.lower() for s in passage.get("signature_tokens", []))
        
        query_lower = query.lower()
        query_tokens = self._tokenize_for_signature(query)
        
        # 1. Boost basé sur les tokens de signature du passage
        matched_tokens = []
        for token in passage_signatures:
            if token in query_lower:
                idf = self.token_idf.get(token, 1.0)
                boost += idf * 0.1
                matched_tokens.append(token)
        
        # 2. Boost basé sur les tokens de signature de l'établissement
        entity_signatures = self.entity_signatures.get(passage_entity, set())
        for token in entity_signatures:
            if token in query_lower and token not in matched_tokens:
                idf = self.token_idf.get(token, 1.0)
                boost += idf * 0.05
        
        # 3. Boost supplémentaire si les base signatures matchent
        for base_token in self.BASE_SIGNATURE_TOKENS:
            if base_token in query_lower and base_token in passage_text:
                idf = self.token_idf.get(base_token, 1.0)
                boost += idf * 0.1
        
        # 4. Pénalité si l'entité ne match pas (et qu'un filtre est actif)
        if entity_code_filter and passage_entity != entity_code_filter:
            boost *= 0.5
        
        # Convertit en multiplicateur (1.0 + boost)
        return 1.0 + min(boost, 1.0)  # Cap à 2x max
    
    def apply_boost_to_passages(
        self,
        scored_passages: List[Any],
        query: str,
        entity_code_filter: Optional[str] = None
    ) -> List[Any]:
        """
        Applique le boost de signature à une liste de passages scorés.
        
        Args:
            scored_passages: Liste de ScoredPassage
            query: Requête utilisateur
            entity_code_filter: Filtre d'entité optionnel
            
        Returns:
            Liste mise à jour avec les boosts appliqués
        """
        for sp in scored_passages:
            boost = self.compute_boost(sp.passage, query, entity_code_filter)
            sp.signature_boost = boost
            sp.final_score *= boost
        
        # Re-trie par score final
        scored_passages.sort(key=lambda x: x.final_score, reverse=True)
        
        return scored_passages
    
    def get_entity_signature_summary(self) -> Dict[str, List[str]]:
        """Retourne un résumé des signatures par établissement."""
        return {
            entity: sorted(list(tokens))[:10]  # Top 10 tokens
            for entity, tokens in self.entity_signatures.items()
        }
    
    def find_matching_signatures(self, query: str) -> List[SignatureMatch]:
        """
        Trouve tous les tokens de signature qui matchent dans la requête.
        """
        query_lower = query.lower()
        matches = []
        
        for entity, signatures in self.entity_signatures.items():
            for token in signatures:
                if token in query_lower:
                    matches.append(SignatureMatch(
                        token=token,
                        idf_score=self.token_idf.get(token, 1.0),
                        entity_code=entity
                    ))
        
        for base_token in self.BASE_SIGNATURE_TOKENS:
            if base_token in query_lower:
                matches.append(SignatureMatch(
                    token=base_token,
                    idf_score=self.token_idf.get(base_token, 1.0),
                    entity_code="BASE"
                ))
        
        return matches


# Tests
if __name__ == "__main__":
    test_passages = [
        {
            "id": "1",
            "entity_code": "P",
            "text": "Offre pour les cadres supérieurs de l'établissement P",
            "signature_tokens": ["cadres supérieurs"]
        },
        {
            "id": "2",
            "entity_code": "AC",
            "text": "Bon d'ouverture de droit signé par l'action sociale",
            "signature_tokens": ["action sociale", "bon d'ouverture de droit"]
        },
        {
            "id": "3",
            "entity_code": "P",
            "text": "Retraités avec attestation de travail",
            "signature_tokens": ["retraités"]
        },
        {
            "id": "4",
            "entity_code": "V",
            "text": "Personnel actif uniquement",
            "signature_tokens": ["personnel actif"]
        },
    ]
    
    print("=== Test Signature Booster ===\n")
    
    booster = SignatureBooster()
    booster.build_signatures(test_passages)
    
    # Affiche les signatures détectées
    print("Signatures par établissement:")
    for entity, sigs in booster.get_entity_signature_summary().items():
        print(f"  {entity}: {sigs[:5]}")
    
    # Test de boost
    test_queries = [
        "Offre pour cadres supérieurs",
        "Documents action sociale",
        "Tarif pour les retraités",
        "Personnel actif établissement V",
    ]
    
    print("\nBoosts calculés:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        for p in test_passages:
            boost = booster.compute_boost(p, query)
            if boost > 1.0:
                print(f"  [{p['entity_code']}] Boost: {boost:.2f} - {p['text'][:40]}")
        
        matches = booster.find_matching_signatures(query)
        if matches:
            print(f"  Matches: {[m.token for m in matches]}")
