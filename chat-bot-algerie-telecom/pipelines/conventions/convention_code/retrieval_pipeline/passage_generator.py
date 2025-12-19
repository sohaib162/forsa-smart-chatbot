"""
Passage Generator - ÉTAPE 0.1
Génère des passages factuels courts à partir des documents JSON.
Chaque ligne du tableau, note, document requis devient un passage.
Objectif: 1 doc → 20 à 50 passages
"""

import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from .normalizer import parse_price, parse_speed, normalize_beneficiary


@dataclass
class Passage:
    """Représente un passage factuel extrait d'un document."""
    id: str
    doc_id: str  # Reference to parent document
    establishment: str
    entity_code: str  # AD, AC, P, etc.
    passage_type: str  # OFFER, EQUIPMENT, DOCUMENTS, NOTE, BENEFICIARY, TELEPHONY, GENERAL
    text: str  # Le texte du passage
    
    # Champs normalisés (ÉTAPE 0.2)
    price_value: Optional[int] = None
    speed_mbps: Optional[float] = None
    is_free: bool = False
    beneficiary: Optional[str] = None
    offer_type: Optional[str] = None  # ADSL, VDSL, FIBRE
    category_segment: Optional[str] = None
    
    # Métadonnées supplémentaires
    signature_tokens: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PassageGenerator:
    """Génère des passages factuels à partir des documents JSON."""
    
    # Mapping establishment name to code
    ENTITY_CODES = {
        "L'établissement P": "P",
        "L'établissement V": "V",
        "L'Établissement F": "F",
        "L'établissement A": "A",
        "L'établissement N": "N",
        "L'établissement O": "O",
        "L'Établissement I": "I",
        "L'établissement AD": "AD",
        "L'établissement AC": "AC",
        "L'établissement AY": "AY",
        "L'établissement E": "E",
        "L'établissement H": "H",
        "L'établissement J": "J",
        "L'établissement K": "K",
        "L'établissement L": "L",
        "L'établissement M": "M",
        "L'établissement Q": "Q",
        "L'établissement R": "R",
        "L'établissement S": "S",
        "L'établissement T": "T",
        "L'établissement U": "U",
        "L'établissement W": "W",
        "L'établissement X": "X",
    }
    
    # Signature tokens par catégorie de bénéficiaires
    SIGNATURE_TOKENS = {
        "cadres": ["cadres supérieurs", "cadres", "dirigeants"],
        "retraites": ["retraités", "en retraite", "retraite"],
        "actifs": ["personnel actif", "en activité", "employés actifs"],
        "ayants_droit": ["ayants droit", "ayant droit", "action sociale"],
        "tous": ["tous bénéficiaires", "tous les employés", "tout le personnel"]
    }
    
    def __init__(self):
        self.passages: List[Passage] = []
        
    def _generate_id(self, doc_id: str, passage_type: str, index: int) -> str:
        """Génère un ID unique pour un passage."""
        content = f"{doc_id}_{passage_type}_{index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_entity_code(self, establishment: str) -> str:
        """Extrait le code de l'établissement."""
        # Cherche dans le mapping
        for name, code in self.ENTITY_CODES.items():
            if name.lower() in establishment.lower():
                return code
        
        # Sinon, extrait la dernière lettre/mot
        parts = establishment.split()
        if parts:
            return parts[-1].upper()
        return "UNK"
    
    def _extract_signature_tokens(self, text: str, beneficiary: str = None) -> List[str]:
        """Extrait les tokens de signature à partir du texte."""
        tokens = []
        text_lower = text.lower()
        
        for category, keywords in self.SIGNATURE_TOKENS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    tokens.append(keyword)
                    
        if beneficiary:
            tokens.append(beneficiary.lower())
            
        return list(set(tokens))
    
    def generate_from_document(self, doc: Dict[str, Any]) -> List[Passage]:
        """Génère tous les passages à partir d'un document."""
        passages = []
        doc_id = doc.get("filename", "unknown")
        establishment = doc.get("establishment", "")
        entity_code = self._get_entity_code(establishment)
        keywords = doc.get("keywords", [])
        
        # 1. Passage général du document (purpose + beneficiaries)
        passages.extend(self._generate_general_passages(doc, doc_id, establishment, entity_code, keywords))
        
        # 2. Passages des offres Internet
        passages.extend(self._generate_internet_offer_passages(doc, doc_id, establishment, entity_code, keywords))
        
        # 3. Passages des offres téléphonie
        passages.extend(self._generate_telephony_passages(doc, doc_id, establishment, entity_code, keywords))
        
        # 4. Passages des équipements (ONT, etc.)
        passages.extend(self._generate_equipment_passages(doc, doc_id, establishment, entity_code, keywords))
        
        # 5. Passages des documents requis
        passages.extend(self._generate_document_passages(doc, doc_id, establishment, entity_code, keywords))
        
        # 6. Passages des notes
        passages.extend(self._generate_note_passages(doc, doc_id, establishment, entity_code, keywords))
        
        return passages
    
    def _generate_general_passages(self, doc: Dict, doc_id: str, establishment: str, 
                                   entity_code: str, keywords: List[str]) -> List[Passage]:
        """Génère les passages généraux (purpose, beneficiaries)."""
        passages = []
        
        # Passage sur l'objet de la convention
        if doc.get("purpose"):
            text = f"[Etab={entity_code}][Type=General] {doc['purpose']}"
            passages.append(Passage(
                id=self._generate_id(doc_id, "general", 0),
                doc_id=doc_id,
                establishment=establishment,
                entity_code=entity_code,
                passage_type="GENERAL",
                text=text,
                keywords=keywords,
                signature_tokens=self._extract_signature_tokens(doc.get("purpose", ""))
            ))
        
        # Passage sur les bénéficiaires
        if doc.get("beneficiaries"):
            text = f"[Etab={entity_code}][Type=Beneficiary] Bénéficiaires: {doc['beneficiaries']}"
            beneficiary = normalize_beneficiary(doc['beneficiaries'])
            passages.append(Passage(
                id=self._generate_id(doc_id, "beneficiary", 0),
                doc_id=doc_id,
                establishment=establishment,
                entity_code=entity_code,
                passage_type="BENEFICIARY",
                text=text,
                beneficiary=beneficiary,
                keywords=keywords,
                signature_tokens=self._extract_signature_tokens(doc['beneficiaries'], beneficiary)
            ))
            
        return passages
    
    def _generate_internet_offer_passages(self, doc: Dict, doc_id: str, establishment: str,
                                          entity_code: str, keywords: List[str]) -> List[Passage]:
        """Génère un passage pour chaque offre Internet."""
        passages = []
        internet_offers = doc.get("internet_offers_table", [])
        
        for idx, offer in enumerate(internet_offers):
            category = offer.get("category_segment", "Tous")
            offer_type = offer.get("offer_type", "").upper()
            speed = offer.get("speed", "")
            price = offer.get("price", "")
            benefits = offer.get("benefits", "")
            
            # Parse les valeurs numériques
            price_value = parse_price(price)
            speed_mbps = parse_speed(speed)
            is_free = "gratuit" in price.lower() or price_value == 0
            
            # Normalise le type d'offre
            if "FIBRE" in offer_type or "FTTH" in offer_type:
                normalized_offer_type = "FIBRE"
            elif "VDSL" in offer_type:
                normalized_offer_type = "VDSL"
            elif "ADSL" in offer_type:
                normalized_offer_type = "ADSL"
            else:
                normalized_offer_type = offer_type
            
            # Normalise le bénéficiaire
            beneficiary = normalize_beneficiary(category)
            
            # Construit le texte du passage
            text = f"[Etab={entity_code}][Type=Offer][Benef={beneficiary}] "
            text += f"Idoom {normalized_offer_type} {speed} à {price}"
            if benefits:
                text += f" ({benefits})"
                
            passages.append(Passage(
                id=self._generate_id(doc_id, "internet_offer", idx),
                doc_id=doc_id,
                establishment=establishment,
                entity_code=entity_code,
                passage_type="OFFER",
                text=text,
                price_value=price_value,
                speed_mbps=speed_mbps,
                is_free=is_free,
                beneficiary=beneficiary,
                offer_type=normalized_offer_type,
                category_segment=category,
                keywords=keywords,
                signature_tokens=self._extract_signature_tokens(category, beneficiary)
            ))
            
        return passages
    
    def _generate_telephony_passages(self, doc: Dict, doc_id: str, establishment: str,
                                     entity_code: str, keywords: List[str]) -> List[Passage]:
        """Génère un passage pour chaque offre téléphonie."""
        passages = []
        telephony_offers = doc.get("telephony_offers_table", [])
        
        for idx, offer in enumerate(telephony_offers):
            offer_name = offer.get("offer", "")
            price = offer.get("price", "")
            benefits = offer.get("benefits", "")
            
            price_value = parse_price(price)
            
            text = f"[Etab={entity_code}][Type=Telephony] {offer_name}: {price}"
            if benefits:
                text += f" - {benefits}"
                
            passages.append(Passage(
                id=self._generate_id(doc_id, "telephony", idx),
                doc_id=doc_id,
                establishment=establishment,
                entity_code=entity_code,
                passage_type="TELEPHONY",
                text=text,
                price_value=price_value,
                offer_type="FIXE",
                keywords=keywords
            ))
            
        return passages
    
    def _generate_equipment_passages(self, doc: Dict, doc_id: str, establishment: str,
                                     entity_code: str, keywords: List[str]) -> List[Passage]:
        """Génère des passages pour les équipements (ONT, etc.)."""
        passages = []
        other_tables = doc.get("other_tables", [])
        
        for table_idx, table in enumerate(other_tables):
            table_name = table.get("table_name", "")
            data = table.get("data", [])
            
            # Vérifie si c'est un tableau d'équipement
            if "ONT" in table_name.upper() or "ÉQUIPEMENT" in table_name.upper() or "MODEM" in table_name.upper():
                for idx, row in enumerate(data):
                    category = row.get("Catégorie", row.get("category", ""))
                    tarif = row.get("Tarif", row.get("tarif", row.get("price", "")))
                    
                    price_value = parse_price(str(tarif))
                    is_free = "gratuit" in str(tarif).lower() or price_value == 0
                    beneficiary = normalize_beneficiary(category)
                    
                    text = f"[Etab={entity_code}][Type=Equipment] {table_name}: {category} - {tarif}"
                    
                    passages.append(Passage(
                        id=self._generate_id(doc_id, f"equipment_{table_idx}", idx),
                        doc_id=doc_id,
                        establishment=establishment,
                        entity_code=entity_code,
                        passage_type="EQUIPMENT",
                        text=text,
                        price_value=price_value,
                        is_free=is_free,
                        beneficiary=beneficiary,
                        keywords=keywords,
                        signature_tokens=self._extract_signature_tokens(category, beneficiary)
                    ))
            else:
                # Autres tableaux (réductions internationales, etc.)
                for idx, row in enumerate(data):
                    row_text = ", ".join(f"{k}: {v}" for k, v in row.items() if v)
                    text = f"[Etab={entity_code}][Type=Other] {table_name}: {row_text}"
                    
                    passages.append(Passage(
                        id=self._generate_id(doc_id, f"other_{table_idx}", idx),
                        doc_id=doc_id,
                        establishment=establishment,
                        entity_code=entity_code,
                        passage_type="OTHER",
                        text=text,
                        keywords=keywords
                    ))
                    
        return passages
    
    def _generate_document_passages(self, doc: Dict, doc_id: str, establishment: str,
                                    entity_code: str, keywords: List[str]) -> List[Passage]:
        """Génère des passages pour les documents requis."""
        passages = []
        
        # Documents pour nouvelle souscription
        docs_new = doc.get("required_documents_new", [])
        if docs_new:
            text = f"[Etab={entity_code}][Type=Documents][Action=New] Documents requis nouvelle souscription: {'; '.join(docs_new)}"
            passages.append(Passage(
                id=self._generate_id(doc_id, "docs_new", 0),
                doc_id=doc_id,
                establishment=establishment,
                entity_code=entity_code,
                passage_type="DOCUMENTS",
                text=text,
                keywords=keywords
            ))
            
            # Aussi générer un passage par document individuel
            for idx, doc_req in enumerate(docs_new):
                text = f"[Etab={entity_code}][Type=Documents] Document requis: {doc_req}"
                passages.append(Passage(
                    id=self._generate_id(doc_id, "doc_item_new", idx),
                    doc_id=doc_id,
                    establishment=establishment,
                    entity_code=entity_code,
                    passage_type="DOCUMENTS",
                    text=text,
                    keywords=keywords
                ))
        
        # Documents pour basculement/switch
        docs_switch = doc.get("required_documents_switch", [])
        if docs_switch:
            text = f"[Etab={entity_code}][Type=Documents][Action=Switch] Documents requis basculement: {'; '.join(docs_switch)}"
            passages.append(Passage(
                id=self._generate_id(doc_id, "docs_switch", 0),
                doc_id=doc_id,
                establishment=establishment,
                entity_code=entity_code,
                passage_type="DOCUMENTS",
                text=text,
                keywords=keywords
            ))
            
        return passages
    
    def _generate_note_passages(self, doc: Dict, doc_id: str, establishment: str,
                                entity_code: str, keywords: List[str]) -> List[Passage]:
        """Génère des passages pour les notes."""
        passages = []
        notes = doc.get("notes", [])
        
        for idx, note in enumerate(notes):
            text = f"[Etab={entity_code}][Type=Note] {note}"
            passages.append(Passage(
                id=self._generate_id(doc_id, "note", idx),
                doc_id=doc_id,
                establishment=establishment,
                entity_code=entity_code,
                passage_type="NOTE",
                text=text,
                keywords=keywords,
                signature_tokens=self._extract_signature_tokens(note)
            ))
            
        return passages
    
    def generate_all_passages(self, documents: List[Dict[str, Any]]) -> List[Passage]:
        """Génère tous les passages à partir d'une liste de documents."""
        all_passages = []
        
        for doc in documents:
            passages = self.generate_from_document(doc)
            all_passages.extend(passages)
            
        self.passages = all_passages
        return all_passages
    
    def save_passages(self, filepath: str):
        """Sauvegarde les passages dans un fichier JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([p.to_dict() for p in self.passages], f, ensure_ascii=False, indent=2)
            
    def load_passages(self, filepath: str) -> List[Passage]:
        """Charge les passages depuis un fichier JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.passages = []
        for item in data:
            # Convertir les champs par défaut
            if 'signature_tokens' not in item:
                item['signature_tokens'] = []
            if 'keywords' not in item:
                item['keywords'] = []
            self.passages.append(Passage(**item))
            
        return self.passages


# Script de génération autonome
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python passage_generator.py <input_json> [output_json]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "passages.json"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    generator = PassageGenerator()
    passages = generator.generate_all_passages(documents)
    generator.save_passages(output_file)
    
    print(f"✅ Generated {len(passages)} passages from {len(documents)} documents")
    print(f"📄 Saved to {output_file}")
