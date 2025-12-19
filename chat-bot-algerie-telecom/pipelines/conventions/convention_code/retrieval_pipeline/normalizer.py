"""
Normalizer - ÉTAPE 0.2
Normalisation structurée des champs: prix, débit, bénéficiaires, etc.
🚨 Ne jamais matcher prix/débit en texte brut
"""

import re
from typing import Optional, Dict, Any, List, Tuple


def parse_price(price_str: str) -> Optional[int]:
    """
    Parse une chaîne de prix et retourne la valeur en DA.
    
    Examples:
        "1 100 DA" -> 1100
        "1100 DA/Mois" -> 1100
        "Gratuit (0 DA)" -> 0
        "3 500 Da" -> 3500
    """
    if not price_str:
        return None
        
    price_str = price_str.lower()
    
    # Cas gratuit
    if "gratuit" in price_str:
        return 0
    
    # Nettoie la chaîne
    # Supprime les indicateurs de période
    price_str = re.sub(r'[/]*(mois|month|mensuel|an|année).*', '', price_str, flags=re.IGNORECASE)
    
    # Extrait tous les chiffres
    # Pattern: capture les nombres avec espaces (ex: "1 100" ou "3 500")
    numbers = re.findall(r'[\d\s]+', price_str)
    
    for num_str in numbers:
        # Nettoie les espaces et convertit
        clean_num = num_str.replace(' ', '').strip()
        if clean_num:
            try:
                return int(clean_num)
            except ValueError:
                continue
                
    return None


def parse_speed(speed_str: str) -> Optional[float]:
    """
    Parse une chaîne de débit et retourne la valeur en Mbps.
    
    Examples:
        "1.5 Gbps" -> 1500.0
        "100 Mbps" -> 100.0
        "20 Mbps" -> 20.0
        "1.2 Gbps" -> 1200.0
        "1 Gbps" -> 1000.0
    """
    if not speed_str:
        return None
        
    speed_str = speed_str.lower()
    
    # Extrait le nombre (avec décimales potentielles)
    match = re.search(r'([\d.,]+)', speed_str)
    if not match:
        return None
        
    try:
        # Gère les virgules françaises
        value = float(match.group(1).replace(',', '.'))
    except ValueError:
        return None
    
    # Convertit en Mbps si nécessaire
    if 'gbps' in speed_str or 'gbit' in speed_str or 'gb/s' in speed_str:
        return value * 1000
    elif 'mbps' in speed_str or 'mbit' in speed_str or 'mb/s' in speed_str:
        return value
    elif 'kbps' in speed_str or 'kbit' in speed_str or 'kb/s' in speed_str:
        return value / 1000
    else:
        # Par défaut, assume Mbps pour les valeurs < 100, Gbps au-dessus
        if value < 10:
            return value * 1000  # Probablement Gbps
        return value


def normalize_beneficiary(beneficiary_str: str) -> str:
    """
    Normalise une chaîne de bénéficiaires en catégorie standard.
    
    Returns:
        "cadres_superieurs" | "retraites" | "actifs" | "ayants_droit" | "tous" | "autre"
    """
    if not beneficiary_str:
        return "autre"
        
    text = beneficiary_str.lower()
    
    # Cadres supérieurs
    if "cadres supérieurs" in text or "cadres superieurs" in text or "cadre supérieur" in text:
        return "cadres_superieurs"
    
    # Retraités (mais pas "personnel et retraités")
    if "retraité" in text and "personnel" not in text and "actif" not in text:
        return "retraites"
    
    # Personnel actif uniquement
    if ("actif" in text or "en activité" in text or "activité" in text) and "retraité" not in text:
        return "actifs"
    
    # Ayants droit
    if "ayant" in text and "droit" in text:
        return "ayants_droit"
    
    # Tous bénéficiaires (actifs + retraités)
    if "tous" in text or ("personnel" in text and "retraité" in text):
        return "tous"
    
    # Personnel générique
    if "personnel" in text or "employé" in text or "salarié" in text:
        return "actifs"
    
    return "autre"


def normalize_offer_type(offer_str: str) -> str:
    """
    Normalise le type d'offre.
    
    Returns:
        "FIBRE" | "VDSL" | "ADSL" | "FIXE" | "AUTRE"
    """
    if not offer_str:
        return "AUTRE"
        
    text = offer_str.upper()
    
    if "FIBRE" in text or "FTTH" in text:
        return "FIBRE"
    elif "VDSL" in text:
        return "VDSL"
    elif "ADSL" in text:
        return "ADSL"
    elif "FIXE" in text or "TELEPHON" in text:
        return "FIXE"
    
    return "AUTRE"


class Normalizer:
    """Classe utilitaire pour normaliser les données des passages."""
    
    # Mapping des bénéficiaires pour la recherche
    BENEFICIARY_SYNONYMS = {
        "cadres_superieurs": [
            "cadres supérieurs", "cadres superieurs", "cadre supérieur",
            "dirigeants", "directeurs", "responsables"
        ],
        "retraites": [
            "retraités", "retraites", "retraité", "en retraite",
            "anciens employés", "pension"
        ],
        "actifs": [
            "personnel actif", "actifs", "en activité", "employés",
            "salariés", "travailleurs", "personnel en activité"
        ],
        "ayants_droit": [
            "ayants droit", "ayant droit", "action sociale",
            "bénéficiaires indirects", "famille"
        ],
        "tous": [
            "tous", "tout le personnel", "tous bénéficiaires",
            "tous les employés", "personnel et retraités"
        ]
    }
    
    # Mapping des prix communs
    COMMON_PRICES = [0, 800, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1680, 
                    1890, 2000, 2100, 2400, 2520, 2600, 3000, 3500, 3600]
    
    # Mapping des débits communs (en Mbps)
    COMMON_SPEEDS = [15, 20, 30, 50, 60, 100, 120, 240, 300, 480, 500, 600, 
                    1000, 1200, 1500]
    
    @staticmethod
    def extract_numeric_values(text: str) -> Dict[str, Any]:
        """
        Extrait toutes les valeurs numériques d'une requête.
        
        Returns:
            {
                "prices": [list of prices in DA],
                "speeds": [list of speeds in Mbps],
                "raw_numbers": [all numbers found]
            }
        """
        result = {
            "prices": [],
            "speeds": [],
            "raw_numbers": []
        }
        
        # Pattern pour les prix (nombre suivi de DA/Da/da)
        price_patterns = re.findall(r'([\d\s]+)\s*(?:da|DA|Da)', text)
        for p in price_patterns:
            price = parse_price(p + " DA")
            if price is not None:
                result["prices"].append(price)
        
        # Pattern pour les débits
        speed_patterns = re.findall(r'([\d.,]+)\s*(gbps|mbps|Gbps|Mbps|gbit|mbit)', text, re.IGNORECASE)
        for value, unit in speed_patterns:
            speed = parse_speed(f"{value} {unit}")
            if speed is not None:
                result["speeds"].append(speed)
        
        # Tous les nombres
        all_numbers = re.findall(r'\d+', text)
        result["raw_numbers"] = [int(n) for n in all_numbers]
        
        return result
    
    @staticmethod
    def normalize_query_beneficiary(query: str) -> Optional[str]:
        """Détecte et normalise le bénéficiaire dans une requête."""
        query_lower = query.lower()
        
        for category, synonyms in Normalizer.BENEFICIARY_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in query_lower:
                    return category
                    
        return None
    
    @staticmethod
    def find_closest_price(price: int, tolerance: float = 0.1) -> int:
        """
        Trouve le prix le plus proche dans la liste des prix communs.
        Utile pour corriger les erreurs de frappe.
        """
        if price in Normalizer.COMMON_PRICES:
            return price
            
        # Cherche le prix le plus proche avec une tolérance
        min_diff = float('inf')
        closest = price
        
        for common_price in Normalizer.COMMON_PRICES:
            diff = abs(common_price - price)
            if diff < min_diff and diff <= price * tolerance:
                min_diff = diff
                closest = common_price
                
        return closest
    
    @staticmethod
    def find_closest_speed(speed: float, tolerance: float = 0.1) -> float:
        """
        Trouve le débit le plus proche dans la liste des débits communs.
        """
        if speed in Normalizer.COMMON_SPEEDS:
            return speed
            
        min_diff = float('inf')
        closest = speed
        
        for common_speed in Normalizer.COMMON_SPEEDS:
            diff = abs(common_speed - speed)
            if diff < min_diff and diff <= speed * tolerance:
                min_diff = diff
                closest = common_speed
                
        return closest
    
    @staticmethod
    def expand_query_with_synonyms(query: str) -> str:
        """
        Étend la requête avec des synonymes pour améliorer le recall.
        """
        expanded = query
        
        # Ajoute des synonymes de bénéficiaires
        for category, synonyms in Normalizer.BENEFICIARY_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in query.lower():
                    # Ajoute les autres synonymes
                    additions = [s for s in synonyms[:3] if s not in query.lower()]
                    if additions:
                        expanded += " " + " ".join(additions)
                    break
        
        return expanded
    
    @staticmethod
    def normalize_passage_for_index(passage_dict: Dict) -> Dict:
        """
        Prépare un passage pour l'indexation.
        Ajoute des champs dérivés pour faciliter la recherche.
        """
        normalized = passage_dict.copy()
        
        # Ajoute des champs de recherche
        search_fields = []
        
        # Texte principal
        search_fields.append(passage_dict.get("text", ""))
        
        # Établissement (plusieurs formes)
        entity = passage_dict.get("entity_code", "")
        establishment = passage_dict.get("establishment", "")
        search_fields.append(entity)
        search_fields.append(establishment)
        
        # Bénéficiaires avec synonymes
        beneficiary = passage_dict.get("beneficiary", "")
        if beneficiary and beneficiary in Normalizer.BENEFICIARY_SYNONYMS:
            search_fields.extend(Normalizer.BENEFICIARY_SYNONYMS[beneficiary])
        
        # Prix en texte
        price = passage_dict.get("price_value")
        if price is not None:
            search_fields.append(f"{price} DA")
            search_fields.append(f"{price}DA")
        
        # Débit en texte
        speed = passage_dict.get("speed_mbps")
        if speed is not None:
            if speed >= 1000:
                search_fields.append(f"{speed/1000} Gbps")
            search_fields.append(f"{speed} Mbps")
        
        normalized["search_text"] = " ".join(str(s) for s in search_fields if s)
        
        return normalized


class QueryNormalizer:
    """Normalise une requête utilisateur pour le pipeline."""
    
    def __init__(self):
        self.normalizer = Normalizer()
    
    def normalize(self, query: str) -> Dict[str, Any]:
        """
        Normalise une requête et extrait les informations structurées.
        
        Returns:
            {
                "original_query": str,
                "expanded_query": str,
                "prices": List[int],
                "speeds": List[float],
                "beneficiary": Optional[str],
                "entities": List[str],  # Sera rempli par EntityDetector
                "numeric_values": Dict
            }
        """
        result = {
            "original_query": query,
            "expanded_query": Normalizer.expand_query_with_synonyms(query),
            "beneficiary": Normalizer.normalize_query_beneficiary(query),
            "entities": [],  # Sera rempli plus tard
        }
        
        # Extrait les valeurs numériques
        numeric = Normalizer.extract_numeric_values(query)
        result["prices"] = numeric["prices"]
        result["speeds"] = numeric["speeds"]
        result["numeric_values"] = numeric
        
        return result


# Tests unitaires
if __name__ == "__main__":
    # Test parse_price
    test_prices = [
        ("1 100 DA", 1100),
        ("800 DA/Mois", 800),
        ("Gratuit (0 DA)", 0),
        ("3 500 Da", 3500),
        ("2 100 Da/Mois", 2100),
        ("1100DA", 1100),
    ]
    
    print("=== Test parse_price ===")
    for input_str, expected in test_prices:
        result = parse_price(input_str)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_str}' -> {result} (expected: {expected})")
    
    # Test parse_speed
    test_speeds = [
        ("1.5 Gbps", 1500.0),
        ("100 Mbps", 100.0),
        ("20 Mbps", 20.0),
        ("1.2 Gbps", 1200.0),
        ("1 Gbps", 1000.0),
        ("300 Mbps", 300.0),
    ]
    
    print("\n=== Test parse_speed ===")
    for input_str, expected in test_speeds:
        result = parse_speed(input_str)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_str}' -> {result} (expected: {expected})")
    
    # Test normalize_beneficiary
    test_benef = [
        ("Cadres Supérieurs", "cadres_superieurs"),
        ("Personnel et Retraités", "tous"),
        ("Personnel Actif", "actifs"),
        ("Retraités", "retraites"),
        ("Tous bénéficiaires", "tous"),
    ]
    
    print("\n=== Test normalize_beneficiary ===")
    for input_str, expected in test_benef:
        result = normalize_beneficiary(input_str)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_str}' -> {result} (expected: {expected})")
