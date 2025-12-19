# src/main.py
from pathlib import Path
from .utils.io_utils import load_products, load_config
from .models.product_doc import load_docs
from .rag.pipeline import RAGPipeline
from .retrievers.three_layer import ThreeLayerRetriever  # Import the updated ThreeLayerRetriever


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "products.json"
    config_path = root / "configs" / "config.yaml"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {data_path}")

    config = load_config(config_path)
    print("Config chargée:", config)

    raw_products = load_products(data_path)
    docs = load_docs(raw_products)
    print(f"Chargé {len(docs)} documents produits.")

    # on récupère top_k depuis le config (par défaut: 4)
    top_k = int(config.get("top_k", 4))

    # Read flags to block layers from the config file
    block_rule_layer = config.get("block_rule_layer", False)
    block_bm25_layer = config.get("block_bm25_layer", False)
    block_dense_layer = config.get("block_dense_layer", False)

    # Initialize the retriever with blocking flags
    retriever = ThreeLayerRetriever(docs, block_rule_layer, block_bm25_layer, block_dense_layer)

    print("\n=== MODE RETRIEVAL-ONLY (règles + BM25 + dense) ===")
    print("Tape une question (ou 'quit' pour sortir)\n")

    while True:
        try:
            q = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir 👋")
            break

        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            print("Au revoir 👋")
            break

        try:
            # Use the retriever to get the answer
            answer = retriever.retrieve(q)
            print("\n📚 Résultats de recherche:\n")
            print(answer)
            print("\n" + "=" * 60 + "\n")
        except Exception as e:
            print(f"Erreur pendant la récupération: {e}")


if __name__ == "__main__":
    main()
