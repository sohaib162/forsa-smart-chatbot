# RAG Products (3-layer retriever)

Pipeline:
1. Rule-based router (if/else sur produits / catégories)
2. Sparse retriever (TF-IDF / BM25)
3. Dense retriever (embeddings)

Lancer:
- Mettre le dataset dans data/products.json
- Installer les dépendances: pip install -r requirements.txt
- Exécuter: python -m src.main
- bel w9t : python3 scripts/process_qa.py     tests/sample_questions.json     results/qa_results.json
- evaluation : python3 -m pytest tests/test_evaluation.py -v -s
- one single question : python3 -m src.muuh