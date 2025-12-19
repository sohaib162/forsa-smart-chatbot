# src/models/product_doc.py
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class ProductDoc:
    id: int
    raw: Dict[str, Any]
    text: str
    keywords: List[str] = field(default_factory=list)
    product_name: str = ""
    category: str = ""
    provider: str = ""

def _safe_get(d: Dict[str, Any], key: str, default=None):
    value = d.get(key, default)
    return value if value is not None else default

def build_text(d: Dict[str, Any]) -> str:
    """
    Builds a large text field from the product data in JSON.
    This text will be used for sparse and dense search.
    """
    keywords = d.get("keywords", [])
    metadata = d.get("metadata", {}) or {}
    pi = d.get("product_info", {}) or {}
    cd = d.get("commercial_details", {}) or {}
    hw = d.get("hardware_details", {}) or {}
    cons = d.get("constraints", {}) or {}
    logi = d.get("logistics_policy", {}) or {}
    ent = d.get("entitlement_details", {}) or {}
    spec = d.get("specifications", {}) or {}
    af = d.get("after_sales_service", {}) or {}

    def as_str(x):
        return "" if x is None else str(x)

    parts = [
        f"Titre document: {as_str(metadata.get('document_title'))}",
        f"Nom produit: {as_str(pi.get('name'))}",
        f"Catégorie: {as_str(pi.get('category'))}",
        f"Type produit: {as_str(pi.get('product_type'))}",
        f"Fournisseur: {as_str(pi.get('provider'))}",
        "",
        "Mots-clés:",
        ", ".join(keywords),
        "",
        "Détails commerciaux:",
        f"Prix précédent: {as_str(cd.get('previous_price_dzd'))}",
        f"Prix courant: {as_str(cd.get('pricing'))}",
        f"Options de prix: {as_str(cd.get('pricing_options'))}",
        f"Canaux de vente: {as_str(cd.get('sales_channels'))}",
        f"Méthodes de paiement: {as_str(cd.get('payment_methods'))}",
        f"Renouvellement abonnement: {as_str(cd.get('subscription_renewal'))}",
        "",
        "Caractéristiques matérielles:",
        f"Hardware: {as_str(hw)}",
        "",
        "Contraintes / cible:",
        f"Plateformes supportées: {as_str(cons.get('supported_platforms'))}",
        f"Public cible: {as_str(cons.get('target_audience'))}",
        f"Niveaux scolaires: {as_str(cons.get('target_school_levels'))}",
        "",
        "Logistique:",
        f"Mode de livraison: {as_str(logi.get('delivery_mode'))}",
        f"Activation: {as_str(logi.get('activation_flow'))}",
        "",
        "Droits / contenu:",
        f"Scope contenu: {as_str(ent.get('content_scope'))}",
        f"Langues contenu: {as_str(ent.get('content_language'))}",
        f"Langues interface: {as_str(ent.get('app_interface_languages'))}",
        "",
        "Spécifications / fonctionnalités:",
        f"Features: {as_str(spec.get('service_features'))}",
        f"Modèles: {as_str(spec.get('models'))}",
        f"Plans / offres: {as_str(spec.get('service_plans'))}",
        "",
        "SAV / garantie:",
        f"SAV: {as_str(af)}",
        "",
        "Infos diverses:",
        as_str(d.get("_unmapped_or_extra_info")),
    ]

    return "\n".join(p for p in parts if p)


def load_docs(data: List[Dict[str, Any]]) -> List[ProductDoc]:
    docs: List[ProductDoc] = []
    for i, d in enumerate(data):
        pi = d.get("product_info", {}) or {}
        product_name = (pi.get("name") or "").lower()
        category = (pi.get("category") or "").lower()
        provider = (pi.get("provider") or "").lower()

        doc = ProductDoc(
            id=i,
            raw=d,
            text=build_text(d),
            keywords=[k.lower() for k in d.get("keywords", [])],
            product_name=product_name,
            category=category,
            provider=provider,
        )
        docs.append(doc)
    return docs
