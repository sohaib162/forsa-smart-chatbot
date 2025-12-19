from typing import List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
from src.models.product_doc import ProductDoc
from src.retrievers.three_layer import ThreeLayerRetriever

def evaluate_retrieval(retriever: ThreeLayerRetriever, query: str, relevant_docs: List[ProductDoc], k: int = 1) -> dict:
    """
    Evaluate the retrieval system by comparing the top-k retrieved documents to the relevant documents.

    :param retriever: The retrieval model (e.g., ThreeLayerRetriever).
    :param query: The query string to search for.
    :param relevant_docs: List of relevant ProductDocs.
    :param k: The number of top documents to retrieve.

    :return: A dictionary containing evaluation metrics.
    """
    # Step 1: Retrieve the top-k documents for the given query
    retrieved_docs = retriever.retrieve(query, k=k)

    # Step 2: Extract the IDs of the relevant documents and retrieved documents
    relevant_doc_ids = {doc.id for doc in relevant_docs}
    retrieved_doc_ids = {doc.id for doc, _ in retrieved_docs}

    # Step 3: Calculate precision, recall, and F1-score
    true_positives = len(relevant_doc_ids.intersection(retrieved_doc_ids))
    false_positives = len(retrieved_doc_ids) - true_positives
    false_negatives = len(relevant_doc_ids) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Step 4: Calculate Mean Reciprocal Rank (MRR)
    mrr = 0
    for i, (doc, _) in enumerate(retrieved_docs):
        if doc.id in relevant_doc_ids:
            mrr = 1 / (i + 1)  # MRR is the reciprocal rank of the first relevant document
            break

    # Return a dictionary of evaluation metrics
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mrr": mrr
    }

# Example usage for the evaluation function
def test_evaluation():
    # Example: Sample query and relevant documents (replace with real documents)
    query = "gaming multimedia box"

    # Sample relevant documents (in a real test, this list comes from your dataset)
    relevant_docs = [
        ProductDoc(id=1, raw={}, text="A powerful multimedia box for TV.", keywords=["multimedia", "box", "tv"], product_name="TWIN BOX", category="Box TV", provider="Company A"),
        ProductDoc(id=3, raw={}, text="Gaming console with 4K support.", keywords=["gaming", "console", "4K"], product_name="Gaming Console", category="Electronics", provider="Company C"),
    ]

    # Initialize the retriever (your retriever instance)
    # Ensure you load your docs correctly and create an instance of ThreeLayerRetriever
    docs = []  # Replace this with actual documents
    retriever = ThreeLayerRetriever(docs)

    # Evaluate the retrieval
    evaluation_metrics = evaluate_retrieval(retriever, query, relevant_docs, k=2)

    # Output the evaluation metrics
    print("Evaluation Metrics:")
    print(f"Precision: {evaluation_metrics['precision']:.4f}")
    print(f"Recall: {evaluation_metrics['recall']:.4f}")
    print(f"F1-Score: {evaluation_metrics['f1_score']:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {evaluation_metrics['mrr']:.4f}")

# Run the test
if __name__ == "__main__":
    test_evaluation()
