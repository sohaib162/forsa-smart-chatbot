from typing import List, Tuple, Optional
import re

from rank_bm25 import BM25Okapi

from ..models.product_doc import ProductDoc


class SparseRetriever:
    """
    Sparse retriever basé sur BM25 (rank_bm25).
    Pas de scikit-learn, pas de SciPy, 100% CPU-friendly.
    """

    def __init__(self, docs: List[ProductDoc]):
        self.docs = docs
        # Tokenize each document
        self.corpus_tokens = [self._tokenize(d.text) for d in docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        # Keep letters/numbers with basic accents
        text = re.sub(r"[^a-z0-9éèàçùôîïüâ']+", " ", text)
        return text.split()

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: Optional[List[ProductDoc]] = None,
    ) -> List[Tuple[ProductDoc, float]]:
        """
        Returns a sorted list of (doc, score) by descending score.
        If `candidates` is provided, restrict the search to this subset.
        """
        q_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(q_tokens)  # Scores for all documents

        results: List[Tuple[ProductDoc, float]] = []

        if candidates is not None:
            # Restrict to the candidates provided
            candidate_ids = [self.docs.index(d) for d in candidates]  # Get the indices of candidates in the full docs list
            cand_scores = [scores[i] for i in candidate_ids]

            # Sort the candidates by score
            ordered = sorted(
                range(len(candidates)),
                key=lambda idx: cand_scores[idx],
                reverse=True,
            )[:k]

            # Collect the results using the sorted candidate indices
            for idx in ordered:
                score = float(cand_scores[idx])
                if score > 0:
                    results.append((candidates[idx], score))
        else:
            # Simple case: all documents
            ordered = sorted(
                range(len(self.docs)),
                key=lambda i: scores[i],
                reverse=True,
            )[:k]

            # Collect the results using the sorted indices
            for i in ordered:
                score = float(scores[i])
                if score > 0:
                    results.append((self.docs[i], score))

        return results
