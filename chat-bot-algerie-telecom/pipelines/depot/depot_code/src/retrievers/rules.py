from typing import List, Set
from ..models.product_doc import ProductDoc

def normalize(text: str) -> str:
    """Normalize text to lowercase for matching"""
    return text.lower().strip()

# STRICT product patterns - only SPECIFIC product names
PRODUCT_PATTERNS = {
    "buzz": ["buzz 6", "buzz6", "buzz 6 pro", "buzz 6 flip", "buzz 6 prime", "buzz6pro"],
    "zte": ["zte", "blade a35", "blade a55", "nubia v60", "zte blade"],
    "ibox": ["ibox"],
    "ekoteb": ["ekoteb"],
    "classateck": ["classateck"],
    "dorouscom": ["dorouscom"],
    "moalim": ["moalim"],
    "twin_box": ["twin box", "twinbox"],
    "cache_modem": ["cache modem", "caches modems"],
    "idoom_market": ["idoom market", "idom market"],
}

def rule_based_filter(query: str, docs: List[ProductDoc]) -> List[ProductDoc]:
    """
    Layer 1: Match ONLY specific product names.
    
    THIS LAYER IS VERY STRICT:
    - "buzz 6 pro" → MATCH ✅
    - "ibox cloud" → MATCH ✅
    - "smartphone pas cher" → NO MATCH ❌ (generic, goes to Layer 2)
    - "téléphone android" → NO MATCH ❌ (generic, goes to Layer 2)
    
    Returns:
        List of matching ProductDoc objects, or EMPTY LIST if no specific match
    """
    q = normalize(query)
    
    # Step 1: Check if query mentions ANY specific product name
    matched_keys: Set[str] = set()
    for key, patterns in PRODUCT_PATTERNS.items():
        for pattern in patterns:
            if pattern in q:
                matched_keys.add(key)
                break  # Found a match for this key
    
    # If NO specific product name found, return empty (cascade to Layer 2)
    if not matched_keys:
        return []
    
    # Step 2: Find documents that match the specific products
    candidates: List[ProductDoc] = []
    for doc in docs:
        # Build searchable text from product name and keywords
        doc_text = normalize(doc.product_name + " " + " ".join(doc.keywords))
        
        # Check if this doc matches any of the matched product keys
        for key in matched_keys:
            patterns = PRODUCT_PATTERNS[key]
            if any(pattern in doc_text for pattern in patterns):
                candidates.append(doc)
                break  # Don't add same doc twice
    
    return candidates