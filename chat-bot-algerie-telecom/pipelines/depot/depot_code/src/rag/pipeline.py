# src/rag/pipeline.py
from typing import List
from ..models.product_doc import ProductDoc
from ..retrievers.three_layer import ThreeLayerRetriever

class RAGPipeline:
    """
    RAG en mode *retrieval-only*:
    - utilise ThreeLayerRetriever (règles + BM25 + dense Gemini)
    - ne fait PAS appel à un LLM
    - renvoie juste un résumé des meilleurs documents
    """

    def __init__(self, docs: List[ProductDoc], top_k: int = 4):
        self.retriever = ThreeLayerRetriever(docs)
        self.top_k = top_k

    @staticmethod
    def _format_doc(doc: ProductDoc, score: float, layer: str) -> str:
        raw = doc.raw
        pi = raw.get("product_info", {}) or {}
        cd = raw.get("commercial_details", {}) or {}

        name = pi.get("name", "Produit")
        category = pi.get("category", "")
        provider = pi.get("provider", "")
        pricing = cd.get("pricing") or cd.get("pricing_options")

        lines: List[str] = []
        lines.append(f"[score={score:.3f}] {name}")
        lines.append(f"  Layer: {layer}")  # Add layer info
        if category:
            lines.append(f"  Catégorie: {category}")
        if provider:
            lines.append(f"  Fournisseur: {provider}")
        if pricing:
            lines.append(f"  Prix / options: {pricing}")

        # Petit extrait du texte brut
        snippet = doc.text[:400].replace("\n", " ")
        lines.append(f"  Extrait: {snippet}...")
        return "\n".join(lines)

    def answer(self, query: str) -> str:
        """
        Retourne une chaîne avec les top-k documents (pour debug / RAG-only).
        """
        results = self.retriever.retrieve(query, k=self.top_k)

        if not results:
            return (
                f"Question: {query}\n\n"
                "Aucun document pertinent n'a été trouvé pour cette question."
            )

        lines: List[str] = []
        lines.append(f"Question: {query}")
        lines.append("")
        lines.append(f"Top {self.top_k} documents récupérés (règles + BM25 + dense):")
        lines.append("")

        # Track the layer for each document
        for doc, score in results:
            layer = self._get_layer_for_doc(doc)
            lines.append(self._format_doc(doc, score, layer))
            lines.append("")  # Ligne vide entre docs

        return "\n".join(lines)

    def _get_layer_for_doc(self, doc: ProductDoc) -> str:
        if doc.id in self.retriever.last_rules_ids:
            return "rules"
        if doc.id in self.retriever.last_sparse_ids:
            return "sparse"
        return "dense"
