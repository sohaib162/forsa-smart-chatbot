"""
advanced_pipeline.py
====================
Graph-Augmented RAG + Argument Mining + Citation-Faithful Generation
for Algérie Télécom internal documents.

Architecture Overview
---------------------

  Raw Documents (PDF / JSON)
         │
         ▼
  ┌─────────────────────────┐
  │  DocumentStructureParser │  ← strips pages, articles, clauses with coordinates
  └───────────┬─────────────┘
              │
     ┌─────────┴──────────┐
     ▼                    ▼
  ┌──────────────┐  ┌─────────────────────┐
  │ ArgumentMiner│  │ EntityRelationExtrac │
  │  (FR/AR NLP) │  │       -tor           │
  └──────┬───────┘  └──────────┬──────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
         ┌──────────────────────┐
         │  KnowledgeGraph       │  ← NetworkX DiGraph persisted as JSON
         │  (nodes + edges)      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ GraphAugmentedRetriever│  ← BM25/dense first, then KG hop-expansion
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │CitationFaithfulGen    │  ← structured prompt → LLM → regex validator
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ HallucinationValidator│  ← cross-checks every claim vs graph nodes
         └──────────────────────┘

Wire-in
-------
Each existing pipeline (conventions, guide, offers, depot) can call
``AdvancedPipeline.run(query, retrieved_docs)`` as a drop-in replacement for
the bare ``call_local_llm(SYSTEM_PROMPT, context_str)`` call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RAG ENHANCEMENTS — SemanticCache + ContextCompressor
# ---------------------------------------------------------------------------
try:
    import sys as _sys
    _rag_root = str(Path(__file__).resolve().parents[1])
    if _rag_root not in _sys.path:
        _sys.path.insert(0, _rag_root)
    from rag_enhancements import SemanticCache as _SemanticCache
    from rag_enhancements import ContextCompressor as _ContextCompressor
    _ENHANCEMENTS_AVAILABLE = True
except ImportError:
    _ENHANCEMENTS_AVAILABLE = False
    _SemanticCache = None
    _ContextCompressor = None

# ---------------------------------------------------------------------------
# 1. ENUMERATIONS & CONSTANTS
# ---------------------------------------------------------------------------

class ArgumentType(str, Enum):
    CONDITION   = "CONDITION"    # si / lorsque / en cas de
    OBLIGATION  = "OBLIGATION"   # doit / s'engage à / il est obligatoire
    PROHIBITION = "PROHIBITION"  # il est interdit / ne peut pas
    PENALTY     = "PENALTY"      # pénalité / amende / sanction / résiliation
    EXCEPTION   = "EXCEPTION"    # sauf / à l'exception de / hormis
    DEFINITION  = "DEFINITION"   # s'entend comme / désigne / on entend par
    DEADLINE    = "DEADLINE"     # délai de / au plus tard / jours ouvrables
    RIGHT       = "RIGHT"        # peut / est autorisé à / a le droit de
    AMOUNT      = "AMOUNT"       # montant / prix / tarif + chiffre
    UNKNOWN     = "UNKNOWN"


class NodeType(str, Enum):
    DOCUMENT  = "DOCUMENT"
    SECTION   = "SECTION"
    ARTICLE   = "ARTICLE"
    CLAUSE    = "CLAUSE"
    ENTITY    = "ENTITY"
    ARGUMENT  = "ARGUMENT"
    PASSAGE   = "PASSAGE"


class RelationType(str, Enum):
    CONTAINS    = "CONTAINS"     # structural parent → child
    REFERENCES  = "REFERENCES"   # cross-document pointer (Art. X → Art. Y)
    GOVERNS     = "GOVERNS"      # rule → entity it governs
    DEFINES     = "DEFINES"      # definition clause → concept
    APPLIES_TO  = "APPLIES_TO"   # condition / obligation → subject
    SUPERSEDES  = "SUPERSEDES"   # newer doc replaces older
    RELATED_TO  = "RELATED_TO"   # semantic proximity


# AT-domain entity patterns (compiled once at import)
_AT_ORG_PATTERNS = re.compile(
    r"\b(Algérie\s+Télécom|AT\b|SATICOM|Direction\s+\w+|ministère\s+\w+|"
    r"établissement\s+[A-Z]{1,4}|Partenaire|Sous-traitant|Co-contractant)\b",
    re.IGNORECASE,
)
_AT_PRODUCT_PATTERNS = re.compile(
    r"\b(Idoom\s*(ADSL|VDSL|Fibre|FTTH|4G)?|MSAN|ONT|FMT|Passeport\s+Biométrique|"
    r"Box\s*AT|décodeur)\b",
    re.IGNORECASE,
)
_LEGAL_PATTERNS = re.compile(
    r"\b(article|clause|alinéa|paragraphe|chapitre|titre|section|annexe|avenant|"
    r"convention|contrat|accord|décision|circulaire)\s+(\d+[\w.-]*)",
    re.IGNORECASE,
)
_AMOUNT_PATTERNS = re.compile(
    r"(\d[\d\s]*(?:[,\.]\d+)?)\s*(DA|DZD|dinars?|%|jours?|mois)",
    re.IGNORECASE,
)

# Argument mining trigger phrases (French)
_ARG_TRIGGERS: Dict[ArgumentType, List[str]] = {
    ArgumentType.CONDITION: [
        r"\bsi\b", r"\blorsque\b", r"\bdans le cas où\b", r"\ben cas de\b",
        r"\bsous réserve que\b", r"\bà condition que\b", r"\bdès lors que\b",
    ],
    ArgumentType.OBLIGATION: [
        r"\bdoit\b", r"\bdoivent\b", r"\best tenu\b", r"\bsont tenus\b",
        r"\bs'engage\b", r"\bobligation\b", r"\bil est obligatoire\b",
        r"\bil incombe\b", r"\bdevra\b", r"\bdevront\b",
    ],
    ArgumentType.PROHIBITION: [
        r"\bil est interdit\b", r"\bne peut pas\b", r"\bne peuvent pas\b",
        r"\best exclu\b", r"\bn'est pas autorisé\b", r"\baucune\b.*\bne\b",
        r"\binterdit\b", r"\bproscrit\b",
    ],
    ArgumentType.PENALTY: [
        r"\bpénalité\b", r"\bpénalités\b", r"\bamende\b", r"\bsanction\b",
        r"\brésiliation\b", r"\bretenue\b", r"\bindemnité\b", r"\bdommages\b",
        r"\bintérêts de retard\b", r"\bpénalisation\b",
    ],
    ArgumentType.EXCEPTION: [
        r"\bsauf\b", r"\bexcepté\b", r"\bà moins que\b", r"\bà l'exception\b",
        r"\bhormis\b", r"\bsauf disposition contraire\b", r"\bnonobstant\b",
    ],
    ArgumentType.DEFINITION: [
        r"\bs'entend comme\b", r"\bdésigne\b", r"\best défini\b",
        r"\bsignifie\b", r"\bon entend par\b", r"\bau sens du présent\b",
        r"\bcorrespond à\b",
    ],
    ArgumentType.DEADLINE: [
        r"\bdélai de\b", r"\bdans un délai\b", r"\bau plus tard\b",
        r"\bjours ouvrables\b", r"\bjours calendaires\b", r"\bécheance\b",
        r"\bà compter de\b",
    ],
    ArgumentType.RIGHT: [
        r"\bpeut\b", r"\bpeuven\b", r"\best autorisé\b", r"\ba le droit\b",
        r"\bfacultatif\b", r"\boptionnel\b",
    ],
}

# Pre-compile all patterns
_COMPILED_ARG_TRIGGERS: Dict[ArgumentType, List[re.Pattern]] = {
    atype: [re.compile(p, re.IGNORECASE) for p in patterns]
    for atype, patterns in _ARG_TRIGGERS.items()
}


# ---------------------------------------------------------------------------
# 2. DATA MODELS
# ---------------------------------------------------------------------------

@dataclass
class ArgumentFragment:
    """A sentence or clause with an identified logical role."""
    fragment_id: str
    text: str
    arg_type: ArgumentType
    confidence: float          # 0.0 – 1.0 (count of triggered patterns / max)
    source_doc: str
    source_page: int
    article_ref: Optional[str] = None   # e.g. "Article 12"
    clause_ref:  Optional[str] = None   # e.g. "Clause 3.2"
    amounts:     List[str] = field(default_factory=list)
    entities:    List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["arg_type"] = self.arg_type.value
        return d


@dataclass
class GraphNode:
    node_id:   str
    node_type: NodeType
    label:     str
    content:   str
    metadata:  Dict[str, Any] = field(default_factory=dict)
    # metadata keys expected:
    #   doc_name, page, article_num, clause_num, arg_type,
    #   entity_code, s3_key, score (for retrieval ranking)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        return d


@dataclass
class GraphEdge:
    src:      str
    dst:      str
    relation: RelationType
    weight:   float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["relation"] = self.relation.value
        return d


@dataclass
class KGRetrievalContext:
    """Enriched retrieval context returned by GraphAugmentedRetriever."""
    query:               str
    direct_nodes:        List[GraphNode]   # top-k from vector/BM25
    expanded_nodes:      List[GraphNode]   # 1-2 hop neighbours from KG
    argument_fragments:  List[ArgumentFragment]
    citation_map:        Dict[str, str]    # citation_key → node_id
    provenance_chain:    List[str]         # ordered list of evidence node_ids


@dataclass
class ValidationReport:
    """Cross-check of LLM output against the knowledge graph."""
    response_text:        str
    total_citations:      int
    valid_citations:      int
    invalid_citations:    List[str]   # cited but not in graph
    unsupported_amounts:  List[str]   # amounts in response not found in graph
    faithfulness_score:   float       # valid_citations / total_citations (1.0 = perfect)
    is_faithful:          bool        # faithfulness_score >= threshold
    flagged_sentences:    List[str]   # sentences with unverifiable claims


@dataclass
class AdvancedPipelineResult:
    query:             str
    answer:            str
    sources:           List[Dict[str, Any]]
    validation:        ValidationReport
    kg_nodes_used:     int
    argument_types:    List[str]
    faithfulness_score: float
    latency_ms:        float


# ---------------------------------------------------------------------------
# 3. ARGUMENT MINER
# ---------------------------------------------------------------------------

class ArgumentMiner:
    """
    Pattern-based argument mining for French administrative/legal text.

    Input:  list of text chunks, each with (text, doc_name, page_num)
    Output: list of ArgumentFragment objects tagged with logical role.

    Uses a trigger-pattern approach: each sentence is tested against compiled
    regex banks. Confidence is the ratio of unique argument types triggered
    in the sentence to the maximum possible.  Ties go to the type with the
    most pattern hits.
    """

    def mine(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[ArgumentFragment]:
        """
        Parameters
        ----------
        chunks : list of dicts with keys:
            text      – raw text of the chunk
            doc_name  – source document filename
            page      – source page number (int)
            article   – article reference string (optional)
            clause    – clause reference string (optional)

        Returns
        -------
        List of ArgumentFragment, one per sentence that has ≥ 1 trigger hit.
        """
        fragments: List[ArgumentFragment] = []

        for chunk in chunks:
            text     = chunk.get("text", "")
            doc_name = chunk.get("doc_name", "unknown")
            page     = chunk.get("page", 0)
            article  = chunk.get("article")
            clause   = chunk.get("clause")

            sentences = self._split_sentences(text)
            for sent in sentences:
                frag = self._classify_sentence(
                    sent, doc_name, page, article, clause
                )
                if frag and frag.arg_type != ArgumentType.UNKNOWN:
                    fragments.append(frag)

        return fragments

    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        """Split on '.', ';', and numbered lists (1. …)."""
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Split on sentence-ending punctuation, keeping the delimiter
        parts = re.split(r"(?<=[.;])\s+(?=[A-ZÀ-Ö0-9«\-])", text)
        # Also split on "Article X –" patterns that start a new clause
        expanded: List[str] = []
        for part in parts:
            sub = re.split(
                r"(?i)(?=(?:article|clause|chapitre|section|alinéa)\s+\d)",
                part,
            )
            expanded.extend([s.strip() for s in sub if s.strip()])
        return expanded

    def _classify_sentence(
        self,
        sentence: str,
        doc_name: str,
        page: int,
        article: Optional[str],
        clause: Optional[str],
    ) -> Optional[ArgumentFragment]:
        if len(sentence) < 15:
            return None

        # Count pattern hits per argument type
        hits: Dict[ArgumentType, int] = {}
        for atype, patterns in _COMPILED_ARG_TRIGGERS.items():
            count = sum(1 for p in patterns if p.search(sentence))
            if count:
                hits[atype] = count

        if not hits:
            arg_type   = ArgumentType.UNKNOWN
            confidence = 0.0
        else:
            # Priority order for tie-breaking: high-specificity legal types win
            # over broad contextual markers (CONDITION appears in almost every clause).
            _PRIORITY = {
                ArgumentType.PENALTY:     8,
                ArgumentType.PROHIBITION: 7,
                ArgumentType.OBLIGATION:  6,
                ArgumentType.DEFINITION:  5,
                ArgumentType.EXCEPTION:   4,
                ArgumentType.DEADLINE:    3,
                ArgumentType.RIGHT:       2,
                ArgumentType.CONDITION:   1,
                ArgumentType.AMOUNT:      0,
            }
            max_hits = max(hits.values())
            top_types = [t for t, c in hits.items() if c == max_hits]
            arg_type  = max(top_types, key=lambda t: _PRIORITY.get(t, 0))
            n_patterns = len(_COMPILED_ARG_TRIGGERS[arg_type])
            confidence = min(1.0, hits[arg_type] / n_patterns)

        # Extract amounts and entities for the fragment
        amounts  = [m.group(0) for m in _AMOUNT_PATTERNS.finditer(sentence)]
        orgs     = [m.group(0) for m in _AT_ORG_PATTERNS.finditer(sentence)]
        products = [m.group(0) for m in _AT_PRODUCT_PATTERNS.finditer(sentence)]
        entities = list(set(orgs + products))

        frag_id = hashlib.md5(
            f"{doc_name}{page}{sentence[:40]}".encode()
        ).hexdigest()[:12]

        return ArgumentFragment(
            fragment_id=frag_id,
            text=sentence,
            arg_type=arg_type,
            confidence=confidence,
            source_doc=doc_name,
            source_page=page,
            article_ref=article,
            clause_ref=clause,
            amounts=amounts,
            entities=entities,
        )


# ---------------------------------------------------------------------------
# 4. ENTITY / RELATION EXTRACTOR
# ---------------------------------------------------------------------------

class EntityRelationExtractor:
    """
    Extracts (entity, relation, entity) triples from text for KG construction.

    Lightweight, rule-based – no fine-tuned NER model required.
    Designed for the specific vocabulary of AT conventions / guides.
    """

    def extract(
        self,
        text: str,
        doc_name: str,
        page: int = 0,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Returns
        -------
        entities : list of {id, label, type, doc_name, page}
        relations : list of {src_id, dst_id, relation}
        """
        entities: List[Dict] = []
        relations: List[Dict] = []
        seen_labels: Set[str] = set()

        def _add_entity(label: str, etype: str) -> str:
            eid = hashlib.md5(f"{label}{etype}".encode()).hexdigest()[:10]
            if label not in seen_labels:
                seen_labels.add(label)
                entities.append(
                    {"id": eid, "label": label, "type": etype,
                     "doc_name": doc_name, "page": page}
                )
            return eid

        # --- Organisations ---
        for m in _AT_ORG_PATTERNS.finditer(text):
            _add_entity(m.group(0).strip(), "ORGANIZATION")

        # --- Products ---
        for m in _AT_PRODUCT_PATTERNS.finditer(text):
            _add_entity(m.group(0).strip(), "PRODUCT")

        # --- Legal references  (e.g. "Article 12", "Clause 3.2") ---
        for m in _LEGAL_PATTERNS.finditer(text):
            ref_label = m.group(0).strip()
            eid = _add_entity(ref_label, "LEGAL_REF")

            # Build REFERENCES relation between the current doc and the ref
            doc_eid = hashlib.md5(f"{doc_name}DOCUMENT".encode()).hexdigest()[:10]
            relations.append(
                {"src_id": doc_eid, "dst_id": eid,
                 "relation": RelationType.REFERENCES.value}
            )

        # --- GOVERNS relations: penalty clause governs its subject ---
        # Heuristic: if a PENALTY sentence mentions an org or product, it governs it.
        penalty_entities = [
            m.group(0) for m in _AT_ORG_PATTERNS.finditer(text)
        ] + [m.group(0) for m in _AT_PRODUCT_PATTERNS.finditer(text)]

        if re.search(r"\bpénalité|sanction|amende\b", text, re.I):
            for pe in penalty_entities:
                pe_eid = hashlib.md5(f"{pe}ORGANIZATION".encode()).hexdigest()[:10]
                penalty_node_eid = hashlib.md5(
                    f"{doc_name}{page}PENALTY".encode()
                ).hexdigest()[:10]
                relations.append(
                    {"src_id": penalty_node_eid, "dst_id": pe_eid,
                     "relation": RelationType.GOVERNS.value}
                )

        return entities, relations


# ---------------------------------------------------------------------------
# 5. KNOWLEDGE GRAPH BUILDER
# ---------------------------------------------------------------------------

class KnowledgeGraphBuilder:
    """
    Builds a NetworkX DiGraph from structured document data.

    Node schema
    -----------
    Each node carries a 'data' attribute that is a GraphNode (serialised as dict).

    Persistence
    -----------
    The graph is saved as two JSON files (nodes.json + edges.json) so it can
    be reloaded without re-processing.  A pickle cache is also supported for
    speed.
    """

    def __init__(self, graph_dir: str):
        self.graph_dir  = Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.G: nx.DiGraph = nx.DiGraph()
        self._arg_miner  = ArgumentMiner()
        self._er_extract = EntityRelationExtractor()

    # ------------------------------------------------------------------
    # Build from document list (JSON schema used throughout the project)
    # ------------------------------------------------------------------

    def build_from_documents(
        self,
        documents: List[Dict[str, Any]],
        domain: str = "general",
    ) -> nx.DiGraph:
        """
        Ingests a list of document dicts (as used by conventions, guides, etc.)
        and constructs the knowledge graph.

        Parameters
        ----------
        documents : list of document dicts. Each dict must have at least:
            - 'filename' or 'id'
            - some text-bearing fields (see _extract_text_chunks)
        domain : tag for all nodes (e.g. 'conventions', 'guides', 'offers')
        """
        for doc in documents:
            self._ingest_document(doc, domain)

        logger.info(
            "KG built: %d nodes, %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )
        return self.G

    def _ingest_document(self, doc: Dict, domain: str):
        doc_name = doc.get("filename") or doc.get("id") or "unknown"
        doc_nid  = self._make_nid("doc", doc_name)

        # Document-level node
        doc_node = GraphNode(
            node_id=doc_nid,
            node_type=NodeType.DOCUMENT,
            label=doc_name,
            content=doc.get("purpose") or doc.get("summary") or "",
            metadata={
                "doc_name": doc_name,
                "domain": domain,
                "s3_key": doc.get("s3_key", ""),
                "establishment": doc.get("establishment", ""),
                "entity_code": doc.get("entity_code", ""),
            },
        )
        self._add_node(doc_node)

        # Extract all text chunks from this document
        chunks = self._extract_text_chunks(doc, doc_name)

        # Argument mining over the chunks
        fragments = self._arg_miner.mine(chunks)
        for frag in fragments:
            frag_nid = self._make_nid("arg", frag.fragment_id)
            frag_node = GraphNode(
                node_id=frag_nid,
                node_type=NodeType.ARGUMENT,
                label=f"{frag.arg_type.value} @ {doc_name}:p{frag.source_page}",
                content=frag.text,
                metadata={
                    "doc_name": frag.source_doc,
                    "page": frag.source_page,
                    "arg_type": frag.arg_type.value,
                    "article_ref": frag.article_ref,
                    "clause_ref": frag.clause_ref,
                    "amounts": frag.amounts,
                    "entities": frag.entities,
                    "confidence": frag.confidence,
                },
            )
            self._add_node(frag_node)
            self._add_edge(doc_nid, frag_nid, RelationType.CONTAINS)

        # Entity / relation extraction over the full doc text
        full_text = " ".join(c["text"] for c in chunks)
        entities, relations = self._er_extract.extract(full_text, doc_name)
        for ent in entities:
            ent_nid  = self._make_nid("ent", ent["id"])
            ent_node = GraphNode(
                node_id=ent_nid,
                node_type=NodeType.ENTITY,
                label=ent["label"],
                content=ent["label"],
                metadata={
                    "entity_type": ent["type"],
                    "doc_name": doc_name,
                    "domain": domain,
                },
            )
            self._add_node(ent_node)
            # Document APPLIES_TO this entity
            self._add_edge(doc_nid, ent_nid, RelationType.APPLIES_TO)

        for rel in relations:
            src_nid = self._make_nid("ent", rel["src_id"])
            dst_nid = self._make_nid("ent", rel["dst_id"])
            if self.G.has_node(src_nid) and self.G.has_node(dst_nid):
                self._add_edge(
                    src_nid, dst_nid,
                    RelationType(rel["relation"]),
                )

    def _extract_text_chunks(
        self, doc: Dict, doc_name: str
    ) -> List[Dict[str, Any]]:
        """
        Recursively flatten all text-bearing fields in a document dict into
        a list of {text, doc_name, page, article, clause} dicts.
        """
        chunks: List[Dict[str, Any]] = []
        page = 0

        def _emit(text: str, article: str = None, clause: str = None):
            if text and len(text.strip()) > 10:
                chunks.append(
                    {"text": text, "doc_name": doc_name,
                     "page": page, "article": article, "clause": clause}
                )

        # Top-level text fields
        for key in ("purpose", "summary", "beneficiaries", "notes_raw"):
            if doc.get(key):
                _emit(str(doc[key]))

        # Notes list
        for note in doc.get("notes", []):
            _emit(str(note))

        # Internet / telephony tables
        for offer in doc.get("internet_offers_table", []) + doc.get(
            "telephony_offers_table", []
        ):
            row_text = "; ".join(
                f"{k}: {v}" for k, v in offer.items() if v
            )
            _emit(row_text)

        # Required documents lists
        for key in ("required_documents_new", "required_documents_switch"):
            for item in doc.get(key, []):
                _emit(str(item))

        # Generic sections list (used in guides)
        for sec in doc.get("sections", []):
            art = sec.get("title") or sec.get("section_title")
            for sub in sec.get("steps", []) + sec.get("content_blocks", []):
                _emit(str(sub.get("text", sub) if isinstance(sub, dict) else sub),
                      article=art)

        # Other tables
        for table in doc.get("other_tables", []):
            for row in table.get("data", []):
                row_text = "; ".join(
                    f"{k}: {v}" for k, v in row.items() if v
                )
                _emit(row_text)

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_nid(self, prefix: str, raw: str) -> str:
        return f"{prefix}_{hashlib.md5(raw.encode()).hexdigest()[:12]}"

    def _add_node(self, node: GraphNode):
        if not self.G.has_node(node.node_id):
            self.G.add_node(node.node_id, data=node.to_dict())

    def _add_edge(self, src: str, dst: str, relation: RelationType, weight: float = 1.0):
        if src in self.G and dst in self.G and not self.G.has_edge(src, dst):
            self.G.add_edge(src, dst, relation=relation.value, weight=weight)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Persist graph to graph_dir/nodes.json + edges.json + graph.pkl."""
        nodes_path = self.graph_dir / "nodes.json"
        edges_path = self.graph_dir / "edges.json"
        pkl_path   = self.graph_dir / "graph.pkl"

        nodes_data = {n: d for n, d in self.G.nodes(data=True)}
        edges_data = [
            {"src": u, "dst": v, **d}
            for u, v, d in self.G.edges(data=True)
        ]
        with open(nodes_path, "w", encoding="utf-8") as f:
            json.dump(nodes_data, f, ensure_ascii=False, indent=2)
        with open(edges_path, "w", encoding="utf-8") as f:
            json.dump(edges_data, f, ensure_ascii=False, indent=2)
        with open(pkl_path, "wb") as f:
            pickle.dump(self.G, f)

        logger.info("KG saved to %s", self.graph_dir)

    def load(self) -> bool:
        """Load graph from pickle cache. Returns True if successful."""
        pkl_path = self.graph_dir / "graph.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                self.G = pickle.load(f)
            logger.info(
                "KG loaded: %d nodes, %d edges",
                self.G.number_of_nodes(),
                self.G.number_of_edges(),
            )
            return True
        return False


# ---------------------------------------------------------------------------
# 6. GRAPH-AUGMENTED RETRIEVER
# ---------------------------------------------------------------------------

class GraphAugmentedRetriever:
    """
    Enriches a list of already-retrieved passages with knowledge-graph context.

    Flow
    ----
    1. Map each retrieved passage to the best-matching graph node (by doc_name
       / entity_code / article_ref fields).
    2. Perform a 1-2 hop expansion in the KG:
         - Follow CONTAINS edges upward to get the parent Document node.
         - Follow REFERENCES edges to pull in cross-document clauses.
         - Follow GOVERNS edges to get the entity / subject that a penalty
           or obligation acts upon.
    3. Collect all ARGUMENT-type nodes in the expanded neighbourhood.
    4. Return a KGRetrievalContext with:
         - direct_nodes (mapped from the original retrieval)
         - expanded_nodes (KG neighbours)
         - argument_fragments (argument-typed nodes)
         - citation_map {citation_key: node_id}  for the generator
         - provenance_chain (ordered list of node ids used)
    """

    def __init__(self, kg_builder: KnowledgeGraphBuilder):
        self.kg = kg_builder.G
        self._node_data: Dict[str, Dict] = {
            n: d["data"] for n, d in self.kg.nodes(data=True)
            if "data" in d
        }

    def enrich(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
        hop_depth: int = 2,
    ) -> KGRetrievalContext:
        """
        Parameters
        ----------
        query : original user query
        retrieved_passages : list of passage dicts from existing BM25/dense retrieval
        hop_depth : number of KG hops (1 or 2 recommended)
        """
        direct_nodes: List[GraphNode] = []
        seen_nids: Set[str] = set()
        citation_map: Dict[str, str] = {}

        # --- Step 1: map passages → graph nodes ---
        for passage in retrieved_passages:
            nid = self._find_best_node(passage)
            if nid and nid not in seen_nids:
                seen_nids.add(nid)
                direct_nodes.append(GraphNode(**self._node_data[nid]))
                ckey = self._make_citation_key(self._node_data[nid])
                citation_map[ckey] = nid

        # --- Step 2: KG hop expansion ---
        expanded_nids: Set[str] = set()
        frontier = set(n.node_id for n in direct_nodes)

        for _ in range(hop_depth):
            next_frontier: Set[str] = set()
            for nid in frontier:
                # Follow all outgoing edges (CONTAINS, REFERENCES, GOVERNS, …)
                for _, dst, edata in self.kg.out_edges(nid, data=True):
                    relation = edata.get("relation", "")
                    if relation in (
                        RelationType.CONTAINS.value,
                        RelationType.REFERENCES.value,
                        RelationType.GOVERNS.value,
                        RelationType.APPLIES_TO.value,
                    ):
                        if dst not in seen_nids and dst not in expanded_nids:
                            expanded_nids.add(dst)
                            next_frontier.add(dst)
                # Follow incoming CONTAINS to reach parent document node
                for src, _, edata in self.kg.in_edges(nid, data=True):
                    if edata.get("relation") == RelationType.CONTAINS.value:
                        if src not in seen_nids and src not in expanded_nids:
                            expanded_nids.add(src)
                            next_frontier.add(src)
            frontier = next_frontier

        expanded_nodes: List[GraphNode] = []
        for nid in expanded_nids:
            if nid in self._node_data:
                nd = GraphNode(**self._node_data[nid])
                expanded_nodes.append(nd)
                ckey = self._make_citation_key(self._node_data[nid])
                if ckey not in citation_map:
                    citation_map[ckey] = nid

        # --- Step 3: collect ARGUMENT fragments ---
        all_nids = seen_nids | expanded_nids
        arg_fragments: List[ArgumentFragment] = []
        for nid in all_nids:
            nd = self._node_data.get(nid)
            if nd and nd.get("node_type") == NodeType.ARGUMENT.value:
                meta = nd.get("metadata", {})
                arg_fragments.append(
                    ArgumentFragment(
                        fragment_id=nid,
                        text=nd.get("content", ""),
                        arg_type=ArgumentType(
                            meta.get("arg_type", "UNKNOWN")
                        ),
                        confidence=meta.get("confidence", 0.0),
                        source_doc=meta.get("doc_name", ""),
                        source_page=meta.get("page", 0),
                        article_ref=meta.get("article_ref"),
                        clause_ref=meta.get("clause_ref"),
                        amounts=meta.get("amounts", []),
                        entities=meta.get("entities", []),
                    )
                )

        provenance_chain = (
            [n.node_id for n in direct_nodes]
            + list(expanded_nids)
        )

        return KGRetrievalContext(
            query=query,
            direct_nodes=direct_nodes,
            expanded_nodes=expanded_nodes,
            argument_fragments=arg_fragments,
            citation_map=citation_map,
            provenance_chain=provenance_chain,
        )

    # ------------------------------------------------------------------

    def _find_best_node(self, passage: Dict) -> Optional[str]:
        """
        Attempt to map a passage dict to a graph node using:
        1. Exact doc_id / doc_name match on DOCUMENT nodes
        2. entity_code match
        3. Fallback: None (passage has no KG match)
        """
        # Try doc_name fields
        for key in ("doc_id", "doc_name", "filename", "id"):
            val = passage.get(key)
            if not val:
                continue
            nid_candidate = f"doc_{hashlib.md5(str(val).encode()).hexdigest()[:12]}"
            if self.kg.has_node(nid_candidate):
                return nid_candidate

        # Try entity_code among ARGUMENT nodes
        entity_code = passage.get("entity_code")
        if entity_code:
            for nid, nd in self._node_data.items():
                if nd.get("metadata", {}).get("entity_code") == entity_code:
                    return nid

        return None

    def _make_citation_key(self, nd: Dict) -> str:
        """Generate a citation key like 'conv_P.docx:p3:Article_12'."""
        meta     = nd.get("metadata", {})
        doc_name = meta.get("doc_name", nd.get("label", "unknown"))
        page     = meta.get("page", 0)
        article  = meta.get("article_ref") or meta.get("article_num", "")
        parts    = [doc_name, f"p{page}"]
        if article:
            parts.append(article.replace(" ", "_"))
        return ":".join(parts)


# ---------------------------------------------------------------------------
# 7. CITATION-FAITHFUL GENERATOR
# ---------------------------------------------------------------------------

_CITATION_REGEX = re.compile(
    r"\[Source:\s*([^\],]+?)(?:,\s*Page\s*(\d+))?(?:,\s*(Article|Clause|Art\.?|Cl\.?)\s*(\S+))?\]",
    re.IGNORECASE,
)
_AMOUNT_IN_RESPONSE = re.compile(
    r"(\d[\d\s]*(?:[,\.]\d+)?)\s*(DA|DZD|dinars?|%)",
    re.IGNORECASE,
)


def _build_citation_faithful_system_prompt(domain: str) -> str:
    """
    Returns a system prompt that forces the model to cite exactly.

    The format required is:
      [Source: <document_name>, Page <N>, Article <X>]

    This is designed for Qwen 2.5 3B which, while small, can follow
    structured output instructions reliably when they are concrete and
    repeated.
    """
    return f"""Tu es un assistant juridique et administratif spécialisé dans les documents internes d'Algérie Télécom (domaine: {domain}).

RÈGLE ABSOLUE DE CITATION :
Chaque affirmation factuelle DOIT être suivie immédiatement d'une citation au format exact :
  [Source: <nom_du_document>, Page <N>, Article <X>]

Si tu n'as pas de source précise pour une information, n'inclus PAS cette information.
N'invente AUCUNE citation. Utilise UNIQUEMENT les informations des documents fournis ci-dessous.

FORMAT DE RÉPONSE OBLIGATOIRE :
1. Commence directement par la réponse (sans introduction).
2. Chaque fait est suivi de sa citation entre crochets.
3. Exemple :
   La pénalité de retard est de **0,5% par jour** [Source: convention_P.docx, Page 4, Article 12].
   Les bénéficiaires sont les cadres supérieurs et retraités [Source: convention_P.docx, Page 1, Article 2].

Si aucun document pertinent n'est trouvé :
  "Aucune information correspondante dans les documents fournis."

Ne reformule pas les articles - cite-les précisément."""


def _build_citation_context_block(
    kg_context: KGRetrievalContext,
    retrieved_raw: List[Dict[str, Any]],
) -> str:
    """
    Builds the user-facing context block that is injected into the prompt.

    Includes:
    - A numbered list of source citations the model is ALLOWED to use
    - The content of each node / passage
    - Argument fragments tagged by type (PENALTY, OBLIGATION, etc.)
    - **Raw retrieved passages** (always included as fallback so the LLM
      always has document content to work with, even when KG is empty)
    """
    lines: List[str] = []
    lines.append(f"=== QUESTION ===\n{kg_context.query}\n")

    # --- KG-sourced nodes (may be empty for pipelines without KG data) ---
    all_nodes = kg_context.direct_nodes + kg_context.expanded_nodes

    if all_nodes:
        lines.append("=== SOURCES DISPONIBLES (utilise UNIQUEMENT ces sources) ===\n")

        # Limit to top 10 nodes to stay within token budget
        # Direct nodes come first (most relevant), then expanded
        show_nodes = all_nodes[:10]

        # Numbered citation index
        cite_index: Dict[str, int] = {}
        for i, node in enumerate(show_nodes, start=1):
            meta     = node.metadata
            doc_name = meta.get("doc_name", node.label)
            page     = meta.get("page", "?")
            article  = meta.get("article_ref") or meta.get("article_num") or ""
            art_str  = f", Article {article}" if article else ""
            cite_key = f"[Source: {doc_name}, Page {page}{art_str}]"
            cite_index[node.node_id] = i

            lines.append(f"[{i}] {cite_key}")
            lines.append(f"    Type: {node.node_type}")
            if meta.get("arg_type"):
                lines.append(f"    Rôle logique: {meta['arg_type']}")
            lines.append(f"    Contenu: {node.content[:400]}")
            lines.append("")

    # --- Always include raw retrieved passages so LLM has document content ---
    if retrieved_raw:
        lines.append("=== DOCUMENTS RÉCUPÉRÉS ===\n")
        for idx, passage in enumerate(retrieved_raw[:3], start=1):  # max 3 passages
            doc_name = passage.get("doc_name") or passage.get("doc_id") or f"Document {idx}"
            text = passage.get("text", "")
            # Truncate to keep total prompt manageable for 3B model
            if len(text) > 2000:
                text = text[:2000] + "… [tronqué]"
            lines.append(f"--- Document {idx}: {doc_name} ---")
            lines.append(text)
            lines.append("")

    # Argument fragments grouped by type
    if kg_context.argument_fragments:
        lines.append("=== FRAGMENTS ARGUMENTATIFS IDENTIFIÉS ===\n")
        by_type: Dict[str, List[ArgumentFragment]] = {}
        for frag in kg_context.argument_fragments:
            by_type.setdefault(frag.arg_type.value, []).append(frag)
        for atype, frags in sorted(by_type.items()):
            lines.append(f"--- {atype} ---")
            for frag in frags[:5]:  # max 5 per type to stay within token budget
                doc_name = frag.source_doc
                page     = frag.source_page
                art_str  = (
                    f", {frag.article_ref}" if frag.article_ref else ""
                )
                lines.append(
                    f"  [Source: {doc_name}, Page {page}{art_str}]  {frag.text[:300]}"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. HALLUCINATION VALIDATOR
# ---------------------------------------------------------------------------

class HallucinationValidator:
    """
    Cross-checks the LLM's response against the knowledge graph.

    Strategy
    --------
    1. Extract all [Source: …] citation tags from the response.
    2. For each citation, check that the cited document exists in the
       citation_map built by the retriever.
    3. Extract all numeric amounts (DA, %, days) from the response and
       verify each appears in at least one graph node's content.
    4. Flag sentences that make factual claims (contain amounts, entity names,
       or legal references) but have no inline citation.
    5. Return a ValidationReport.
    """

    def __init__(self, faithfulness_threshold: float = 0.60):
        self.threshold = faithfulness_threshold

    def validate(
        self,
        response_text: str,
        kg_context: KGRetrievalContext,
    ) -> ValidationReport:
        # 1. Extract citations from response
        cited_sources: List[str] = []
        for m in _CITATION_REGEX.finditer(response_text):
            doc_part = m.group(1).strip()
            cited_sources.append(doc_part)

        # 2. Validate each citation against citation_map keys
        valid_docs = {
            key.split(":")[0]  # just the doc_name portion
            for key in kg_context.citation_map.keys()
        }
        invalid_citations: List[str] = []
        for doc in cited_sources:
            # Partial match: tolerate variations in casing / spacing
            matched = any(
                doc.lower() in valid.lower() or valid.lower() in doc.lower()
                for valid in valid_docs
            )
            if not matched:
                invalid_citations.append(doc)

        # 3. Validate amounts in response against graph node contents
        all_node_text = " ".join(
            (nd.content or "")
            for nd in kg_context.direct_nodes + kg_context.expanded_nodes
        ) + " ".join(
            frag.text for frag in kg_context.argument_fragments
        )
        unsupported_amounts: List[str] = []
        for m in _AMOUNT_IN_RESPONSE.finditer(response_text):
            amount_str = m.group(0).strip()
            # Numeric match: strip spaces from the amount in node text too
            normalised_amount = re.sub(r"\s+", "", amount_str)
            normalised_node   = re.sub(r"\s+", "", all_node_text)
            if normalised_amount.lower() not in normalised_node.lower():
                unsupported_amounts.append(amount_str)

        # 4. Flag sentences without citations that contain factual markers
        flagged: List[str] = []
        sentences = re.split(r"(?<=[.;])\s+", response_text)
        for sent in sentences:
            has_citation = bool(_CITATION_REGEX.search(sent))
            has_amount   = bool(_AMOUNT_IN_RESPONSE.search(sent))
            has_legal    = bool(_LEGAL_PATTERNS.search(sent))
            has_entity   = bool(
                _AT_ORG_PATTERNS.search(sent) or _AT_PRODUCT_PATTERNS.search(sent)
            )
            if (has_amount or has_legal or has_entity) and not has_citation:
                if len(sent.strip()) > 20:
                    flagged.append(sent.strip())

        total  = len(cited_sources)
        valid  = total - len(invalid_citations)
        score  = (valid / total) if total > 0 else 1.0  # vacuously faithful if no citations

        return ValidationReport(
            response_text=response_text,
            total_citations=total,
            valid_citations=valid,
            invalid_citations=invalid_citations,
            unsupported_amounts=unsupported_amounts,
            faithfulness_score=round(score, 3),
            is_faithful=score >= self.threshold,
            flagged_sentences=flagged,
        )

    def build_faithfulness_annotation(
        self, report: ValidationReport
    ) -> str:
        """
        Returns a short annotation appended to the response so the frontend
        can display a trust indicator.
        """
        if report.is_faithful:
            return (
                f"\n\n---\n*Fiabilité: {report.faithfulness_score:.0%} "
                f"({report.valid_citations}/{report.total_citations} "
                f"citations vérifiées)*"
            )
        else:
            return (
                f"\n\n---\n*⚠️ Attention: {report.faithfulness_score:.0%} "
                f"de citations vérifiées. Certaines affirmations peuvent "
                f"nécessiter une vérification manuelle.*"
            )


# ---------------------------------------------------------------------------
# 9. ADVANCED PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------

class AdvancedPipeline:
    """
    Drop-in replacement for the bare ``call_local_llm(SYSTEM_PROMPT, context)``
    call in each existing pipeline.

    Usage (from any existing pipeline file)
    ----------------------------------------
    >>> from pipelines.advanced_pipeline import AdvancedPipeline
    >>> pipeline = AdvancedPipeline(
    ...     graph_dir="pipelines/conventions/convention_code/kg",
    ...     domain="conventions",
    ...     faithfulness_threshold=0.6,
    ... )
    >>> # At startup: build KG from documents
    >>> pipeline.build_graph(documents)           # once, then cached
    >>>
    >>> # At query time: replace call_local_llm(...)
    >>> result = pipeline.run(
    ...     query=user_query,
    ...     retrieved_passages=retrieved_passages,   # from existing retriever
    ...     llm_client=get_llm_client(),
    ...     max_new_tokens=512,
    ... )
    >>> return {"answer": result.answer, "sources": result.sources}
    """

    def __init__(
        self,
        graph_dir: str,
        domain: str = "general",
        faithfulness_threshold: float = 0.60,
        annotate_response: bool = True,
    ):
        self.domain       = domain
        self.annotate     = annotate_response
        self.kg_builder   = KnowledgeGraphBuilder(graph_dir)
        self.retriever    = None   # set after graph is built / loaded
        self.validator    = HallucinationValidator(faithfulness_threshold)
        self._system_prompt = _build_citation_faithful_system_prompt(domain)
        self._is_ready    = False

        # ── RAG enhancements ────────────────────────────────────────────────
        if _ENHANCEMENTS_AVAILABLE:
            self._cache      = _SemanticCache(max_size=256, ttl_seconds=3600.0)
            self._compressor = _ContextCompressor(max_sentences=8)
        else:
            self._cache      = None
            self._compressor = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def build_graph(
        self,
        documents: List[Dict[str, Any]],
        force_rebuild: bool = False,
    ) -> None:
        """Build (or reload from cache) the knowledge graph."""
        if not force_rebuild and self.kg_builder.load():
            logger.info("KG loaded from cache for domain '%s'", self.domain)
        else:
            logger.info("Building KG for domain '%s'…", self.domain)
            self.kg_builder.build_from_documents(documents, self.domain)
            self.kg_builder.save()

        self.retriever = GraphAugmentedRetriever(self.kg_builder)
        self._is_ready = True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
        llm_client,              # LocalLLMClient instance
        sources: List[Dict[str, Any]] = None,
        max_new_tokens: int = 512,
    ) -> AdvancedPipelineResult:
        """
        Full pipeline: KG enrichment → citation prompt → LLM → validation.

        Parameters
        ----------
        query             : user's original question
        retrieved_passages: list of passage dicts from the existing retriever
        llm_client        : instance of LocalLLMClient (from local_llm_client.py)
        sources           : list of source metadata dicts (passed through to output)
        max_new_tokens    : max tokens for the LLM response

        Returns
        -------
        AdvancedPipelineResult
        """
        if not self._is_ready:
            raise RuntimeError(
                "AdvancedPipeline not ready. Call build_graph() first."
            )

        # ── Semantic cache check (sub-ms for repeated / paraphrase queries) ──
        if self._cache is not None:
            cache_key    = f"{self.domain}::{query}"
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                logger.info("AdvancedPipeline [%s] cache HIT for query", self.domain)
                return cached_result

        t0 = time.time()

        # ── Context compression — fewer tokens, same information ─────────────
        if self._compressor is not None and retrieved_passages:
            retrieved_passages = self._compressor.compress(retrieved_passages, query)

        # --- Speed optimisation: skip KG enrichment if graph is trivially small ---
        kg_has_data = len(self.kg_builder.G.nodes) > 5

        if kg_has_data:
            # Step 1: KG-augmented retrieval (2-hop for rich graphs, 1-hop for speed)
            hop_depth = 2 if len(self.kg_builder.G.nodes) < 500 else 1
            kg_context = self.retriever.enrich(
                query=query,
                retrieved_passages=retrieved_passages,
                hop_depth=hop_depth,
            )
            t_kg = time.time()
            logger.debug("KG enrichment: %.0fms", (t_kg - t0) * 1000)
        else:
            # Fast path: no KG data — create an empty KGRetrievalContext
            kg_context = KGRetrievalContext(
                query=query,
                direct_nodes=[],
                expanded_nodes=[],
                argument_fragments=[],
                citation_map={},
                provenance_chain=[],
            )
            t_kg = time.time()
            logger.debug("KG skipped (empty graph): %.0fms", (t_kg - t0) * 1000)

        # Step 2: Build the context block injected into the prompt
        context_block = _build_citation_context_block(kg_context, retrieved_passages)

        # Step 3: LLM generation (citation-faithful prompt)
        # Use generate_with_citations for lower temperature, retry logic,
        # and stricter decoding — critical for legal/admin accuracy.
        response_text = llm_client.generate_with_citations(
            system_prompt=self._system_prompt,
            user_content=context_block,
            max_new_tokens=max_new_tokens,
            require_citations=True,
        )
        t_llm = time.time()
        logger.debug("LLM generation: %.0fms", (t_llm - t_kg) * 1000)

        # Step 4: Validate citations & amounts
        report = self.validator.validate(response_text, kg_context)

        # Step 5: Optionally append faithfulness annotation
        final_answer = response_text
        if self.annotate:
            final_answer += self.validator.build_faithfulness_annotation(report)

        # Collect unique argument types used
        arg_types = list({
            frag.arg_type.value for frag in kg_context.argument_fragments
        })

        latency_ms = (time.time() - t0) * 1000
        logger.info(
            "AdvancedPipeline [%s] | total=%.0fms | kg=%.0fms | llm=%.0fms | faith=%.2f",
            self.domain, latency_ms, (t_kg - t0) * 1000, (t_llm - t_kg) * 1000,
            report.faithfulness_score,
        )

        result = AdvancedPipelineResult(
            query=query,
            answer=final_answer,
            sources=sources or [],
            validation=report,
            kg_nodes_used=len(kg_context.direct_nodes) + len(kg_context.expanded_nodes),
            argument_types=arg_types,
            faithfulness_score=report.faithfulness_score,
            latency_ms=round(latency_ms, 1),
        )

        # ── Store in semantic cache for future identical/similar queries ──────
        if self._cache is not None:
            self._cache.put(f"{self.domain}::{query}", result)

        return result

    # ------------------------------------------------------------------
    # Convenience: run without a pre-built graph (builds on first call)
    # ------------------------------------------------------------------

    def run_with_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        retrieved_passages: List[Dict[str, Any]],
        llm_client,
        sources: List[Dict[str, Any]] = None,
        max_new_tokens: int = 512,
    ) -> AdvancedPipelineResult:
        """
        Builds the graph from documents if not already ready, then runs.
        Useful for pipelines that load documents dynamically at query time.
        """
        if not self._is_ready:
            self.build_graph(documents)
        return self.run(
            query=query,
            retrieved_passages=retrieved_passages,
            llm_client=llm_client,
            sources=sources,
            max_new_tokens=max_new_tokens,
        )


# ---------------------------------------------------------------------------
# 10. FACTORY HELPERS — one per existing pipeline domain
# ---------------------------------------------------------------------------

_pipeline_cache: Dict[str, "AdvancedPipeline"] = {}


def get_advanced_pipeline(
    domain: str,
    graph_dir: str,
    faithfulness_threshold: float = 0.60,
    annotate_response: bool = True,
) -> AdvancedPipeline:
    """
    Singleton factory.  Returns a cached AdvancedPipeline per domain.
    The graph is loaded from disk on first call; subsequent calls reuse
    the in-memory graph.
    """
    if domain not in _pipeline_cache:
        pipeline = AdvancedPipeline(
            graph_dir=graph_dir,
            domain=domain,
            faithfulness_threshold=faithfulness_threshold,
            annotate_response=annotate_response,
        )
        _pipeline_cache[domain] = pipeline
    return _pipeline_cache[domain]


def build_conventions_pipeline(
    conventions_json_path: str,
    graph_dir: str,
    force_rebuild: bool = False,
) -> AdvancedPipeline:
    """
    Convenience builder for the conventions domain.

    Call once at application startup:
    >>> pipeline = build_conventions_pipeline(
    ...     conventions_json_path="pipelines/conventions/convention_code/data/conventions.json",
    ...     graph_dir="pipelines/conventions/convention_code/kg",
    ... )
    """
    with open(conventions_json_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    pipeline = AdvancedPipeline(
        graph_dir=graph_dir,
        domain="conventions",
        faithfulness_threshold=0.60,
    )
    pipeline.build_graph(documents, force_rebuild=force_rebuild)
    return pipeline


def build_guides_pipeline(
    guides_json_path: str,
    graph_dir: str,
    force_rebuild: bool = False,
) -> AdvancedPipeline:
    """Convenience builder for the guides domain."""
    with open(guides_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Guide_NGBSS.json wraps data in {"guides": [...]}
    documents = raw.get("guides", raw) if isinstance(raw, dict) else raw

    pipeline = AdvancedPipeline(
        graph_dir=graph_dir,
        domain="guides",
        faithfulness_threshold=0.65,
    )
    pipeline.build_graph(documents, force_rebuild=force_rebuild)
    return pipeline
