# -*- coding: utf-8 -*-
from retrieval_pipeline.hybrid_ranker import BM25Index
import json

with open('data/passages.json', 'r', encoding='utf-8') as f:
    passages = json.load(f)

bm25 = BM25Index()
bm25.build_index(passages, text_field='text')
results = bm25.search('etablissement P', top_k=5)
print('BM25 results:', results)

# Check sample passage text
for idx, score in results[:3]:
    print(f"\n--- Passage {idx} (score={score:.4f}) ---")
    print(passages[idx].get('text', '')[:200])
