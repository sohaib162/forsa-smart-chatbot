"""
Bilingual synonyms dictionary for French-Arabic query expansion.

This module provides a mapping of business terms and concepts between
French and Arabic for better cross-language matching in the retrieval pipeline.

The synonyms are used in:
- Rule router: to match Arabic queries against French metadata and vice versa
- Sparse index: for query expansion to improve BM25 recall
"""

# Bilingual synonym dictionary
# Format: Each entry maps a normalized term to its equivalents in both languages
BILINGUAL_SYNONYMS = {
    # Technology terms
    "fibre": ["fibre optique", "fibre", "الألياف البصرية", "الألياف", "فيبر"],
    "الألياف": ["الألياف البصرية", "fibre", "fibre optique", "فيبر"],
    "fibre optique": ["fibre", "الألياف البصرية", "الألياف", "فيبر"],

    "adsl": ["adsl", "اي دي اس ال", "ايديسال"],
    "vdsl": ["vdsl", "في دي اس ال", "فيديسال"],

    "4g": ["4g lte", "4g", "لتي", "الجيل الرابع", "4جي"],
    "lte": ["4g lte", "lte", "لتي", "4جي"],
    "4g lte": ["4g", "lte", "لتي", "الجيل الرابع"],

    # Speed/bandwidth terms
    "débit": ["débit", "vitesse", "تدفق", "سرعة", "ديبي"],
    "تدفق": ["تدفق", "سرعة", "débit", "vitesse"],
    "vitesse": ["vitesse", "débit", "سرعة", "تدفق"],

    "mbps": ["mbps", "mega", "ميغابت", "ميقا"],
    "gbps": ["gbps", "giga", "جيغابت", "غيغا", "جيقا"],

    # Customer segments
    "résidentiel": ["résidentiel", "particulier", "خواص", "سكني", "منزلي"],
    "خواص": ["خواص", "résidentiel", "particulier", "سكني"],
    "particulier": ["particulier", "résidentiel", "خواص", "سكني"],

    "professionnel": ["professionnel", "entreprise", "محترف", "مهني", "شركة"],
    "محترف": ["محترف", "مهني", "professionnel", "entreprise"],
    "entreprise": ["entreprise", "société", "professionnel", "شركة", "مؤسسة"],

    "locataire": ["locataire", "مستأجر", "resident"],
    "مستأجر": ["مستأجر", "locataire", "resident"],

    "école": ["école", "écoles", "primaire", "scolaire", "établissement", "مدرسة", "مدارس", "ابتدائي"],
    "مدرسة": ["مدرسة", "مدارس", "ابتدائي", "école", "écoles", "primaire", "établissement"],
    "primaire": ["primaire", "école primaire", "écoles primaires", "ابتدائي", "مدرسة ابتدائية", "établissement scolaire"],
    "établissement": ["établissement", "établissement scolaire", "école", "مؤسسة", "مدرسة"],
    "scolaire": ["scolaire", "école", "éducatif", "مدرسي", "تعليمي"],

    # Services
    "modernisation": ["modernisation", "migration", "évolution", "عصرنة", "تحديث", "تطوير"],
    "عصرنة": ["عصرنة", "تحديث", "modernisation", "migration"],
    "migration": ["migration", "modernisation", "basculement", "تحويل", "عصرنة"],

    "basculement": ["basculement", "migration", "changement", "تحويل", "نقل"],
    "تحويل": ["تحويل", "نقل", "basculement", "migration"],

    "installation": ["installation", "activation", "تركيب", "تنصيب", "تثبيت"],
    "تركيب": ["تركيب", "تنصيب", "installation", "activation"],

    # Equipment
    "ont": ["ont", "modem fibre", "modem optique", "اونت", "مودم الألياف"],
    "modem": ["modem", "routeur", "مودم", "راوتر"],
    "مودم": ["مودم", "راوتر", "modem", "routeur"],
    "routeur": ["routeur", "modem", "راوتر", "مودم"],

    "wifi": ["wifi", "wi-fi", "wifi 6", "واي فاي", "وايفاي"],
    "واي فاي": ["واي فاي", "وايفاي", "wifi", "wi-fi"],

    # Billing/pricing
    "tarif": ["tarif", "prix", "abonnement", "سعر", "تسعيرة", "تعريفة"],
    "سعر": ["سعر", "تسعيرة", "tarif", "prix"],
    "prix": ["prix", "tarif", "coût", "سعر", "تكلفة"],

    "abonnement": ["abonnement", "forfait", "اشتراك", "عرض"],
    "اشتراك": ["اشتراك", "عرض", "abonnement", "forfait"],
    "forfait": ["forfait", "abonnement", "offre", "عرض", "اشتراك"],

    # Contract terms
    "engagement": ["engagement", "contrat", "durée", "التزام", "عقد"],
    "التزام": ["التزام", "عقد", "engagement", "contrat"],
    "sans engagement": ["sans engagement", "بدون التزام", "no commitment"],
    "durée": ["durée", "période", "مدة", "فترة"],
    "mois": ["mois", "mensuel", "شهر", "شهري"],

    # Data/credit
    "crédit": ["crédit", "solde", "balance", "رصيد", "ذمة"],
    "رصيد": ["رصيد", "crédit", "solde", "balance"],
    "solde": ["solde", "crédit", "balance", "رصيد"],
    "balance": ["balance", "solde", "crédit", "رصيد"],

    "données": ["données", "data", "volume", "بيانات", "حجم"],
    "بيانات": ["بيانات", "données", "data"],

    # Taxes
    "timbre": ["timbre fiscal", "timbre", "taxe", "طابع", "طابع جبائي"],
    "طابع": ["طابع", "طابع جبائي", "timbre", "taxe"],
    "fiscal": ["fiscal", "taxe", "جبائي", "ضريبة"],
    "جبائي": ["جبائي", "ضريبة", "fiscal", "taxe"],

    # Offers/products
    "offre": ["offre", "promotion", "formule", "عرض", "عروض"],
    "عرض": ["عرض", "عروض", "offre", "promotion", "formule"],

    # Volume/data limits
    "illimité": ["illimité", "illimitée", "sans limite", "unlimited", "غير محدود", "بلا حدود"],
    "غير محدود": ["غير محدود", "illimité", "sans limite"],
    "volume": ["volume", "données", "data", "quota", "حجم", "بيانات"],
    "quota": ["quota", "limite", "plafond", "حد", "سقف"],

    "parrainage": ["parrainage", "sponsoring", "إحالة", "رعاية"],
    "إحالة": ["إحالة", "parrainage"],

    "gamer": ["gamer", "gamers", "gaming", "jeux", "جيمر", "ألعاب"],
    "gaming": ["gaming", "gamer", "jeux", "ألعاب", "جيمر"],

    # Payment
    "paiement": ["paiement", "payment", "payement", "دفع", "تسديد"],
    "دفع": ["دفع", "تسديد", "paiement", "payment"],
    "électronique": ["électronique", "numérique", "digital", "إلكتروني", "رقمي"],
    "إلكتروني": ["إلكتروني", "رقمي", "électronique", "digital"],

    # Actions/operations
    "upgrade": ["upgrade", "amélioration", "augmentation", "ترقية", "تحسين"],
    "ترقية": ["ترقية", "تحسين", "upgrade", "amélioration"],
    "downgrade": ["downgrade", "réduction", "خفض", "تخفيض"],

    # Service issues
    "interruption": ["interruption", "coupure", "panne", "انقطاع", "عطل"],
    "انقطاع": ["انقطاع", "عطل", "interruption", "coupure"],
    "problème": ["problème", "souci", "مشكلة", "عطل"],
    "مشكلة": ["مشكلة", "عطل", "problème", "souci"],

    # Location/transfer
    "déménagement": ["déménagement", "transfert", "changement adresse", "نقل", "تحويل عنوان"],
    "نقل": ["نقل", "déménagement", "transfert"],
    "transférer": ["transférer", "نقل", "تحويل", "transférer"],

    # Customer service
    "assistance": ["assistance", "support", "aide", "مساعدة", "دعم"],
    "مساعدة": ["مساعدة", "دعم", "assistance", "support"],
    "réclamation": ["réclamation", "plainte", "شكوى", "شكاية"],
    "شكوى": ["شكوى", "شكاية", "réclamation", "plainte"],

    # Coverage
    "zone": ["zone", "couverture", "région", "منطقة", "تغطية"],
    "منطقة": ["منطقة", "zone", "région"],
    "couverture": ["couverture", "éligibilité", "zone", "تغطية", "أهلية"],
    "تغطية": ["تغطية", "أهلية", "couverture", "zone"],
    "éligibilité": ["éligibilité", "eligibilité", "أهلية", "صلاحية"],
    "أهلية": ["أهلية", "éligibilité", "couverture"],
}


def get_synonyms(term: str, max_synonyms: int = 5) -> list:
    """
    Get synonyms for a given term (French or Arabic).

    Args:
        term: Normalized term to look up
        max_synonyms: Maximum number of synonyms to return

    Returns:
        List of synonym terms (including the original term)
    """
    term = term.lower().strip()

    if term in BILINGUAL_SYNONYMS:
        synonyms = [term] + BILINGUAL_SYNONYMS[term]
        return synonyms[:max_synonyms]

    return [term]


def expand_query_with_synonyms(tokens: list, max_expansions: int = 3) -> list:
    """
    Expand query tokens with bilingual synonyms.

    This function takes a list of query tokens and expands each token
    with its synonyms, useful for improving recall in BM25 search.

    Args:
        tokens: List of normalized query tokens
        max_expansions: Maximum number of synonyms to add per token

    Returns:
        Expanded list of tokens (original + synonyms)
    """
    expanded = []

    for token in tokens:
        # Add original token
        expanded.append(token)

        # Add synonyms
        synonyms = get_synonyms(token, max_synonyms=max_expansions + 1)
        for syn in synonyms[1:max_expansions + 1]:  # Skip first (original)
            if syn not in expanded:
                expanded.append(syn)

    return expanded


def find_cross_language_matches(query_tokens: list, doc_tokens: list) -> int:
    """
    Count cross-language matches between query and document using synonyms.

    This is useful for rule-based routing to detect when an Arabic query
    matches French metadata or vice versa.

    Args:
        query_tokens: List of query tokens
        doc_tokens: List of document tokens

    Returns:
        Number of cross-language matches found
    """
    matches = 0

    for q_token in query_tokens:
        q_syns = set(get_synonyms(q_token, max_synonyms=10))

        for d_token in doc_tokens:
            if d_token in q_syns:
                matches += 1
                break  # Count each query token match only once

    return matches
