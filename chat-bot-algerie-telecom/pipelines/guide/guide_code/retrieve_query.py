from .scripts.retrieval_function import retrieve
result = retrieve("Quels sont les frais d'accès pour un nouveau Pack IDOOM Fibre ?", top_k=1)
print(result)

# from scripts.retrieval_function import retrieve_to_file
# retrieve_to_file("Your question here", "guide_retrieval_output.json")