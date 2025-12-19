# src/rag/pipeline.py
from src.retrievers.three_layer import ThreeLayerRetriever

class RAGPipeline:
    def __init__(self, docs):
        self.retriever = ThreeLayerRetriever(docs)

    def run(self, query):
        best_doc, best_score, best_layer = self.retriever.retrieve(query, k=1)

        if best_doc is None:
            return (
                f"Question: {query}\n\n"
                "Aucun document pertinent n'a été trouvé pour cette question."
            )

        # Format the output
        lines = []
        lines.append(f"Question: {query}")
        lines.append("")
        lines.append(f"Top document récupéré (règles + BM25 + dense):")
        lines.append("")
        lines.append(self._format_doc(best_doc, best_score, best_layer))
        return "\n".join(lines)

    def _format_doc(self, doc, score, layer):
        # Format document output (you can customize this method)
        return f"[score={score:.3f}] {doc.name}\n  Layer: {layer}\n  Excerpt: {doc.extract}"
