# Optimal Retrieval Pipeline
# Target: Recall@1 ≈ 85%
# 
# Pipeline Architecture:
# 1. Passage Generation (factual passages from documents)
# 2. Normalization (structured fields for prices, speeds, etc.)
# 3. Intent Classification (PRICE, SPEED, DOCUMENTS, BENEFICIARY, GENERAL)
# 4. Entity Detection with Hard Filter
# 5. Dual Retrieval (BM25 + Dense)
# 6. Hybrid Scoring (intent-based weights)
# 7. Numeric Hard Boost
# 8. Signature Boosting
# 9. Cross-Encoder Rerank
# 10. Document Aggregation

from .passage_generator import PassageGenerator
from .normalizer import Normalizer, parse_price, parse_speed
from .intent_classifier import IntentClassifier, Intent
from .entity_detector import EntityDetector
from .hybrid_ranker import HybridRanker
from .signature_booster import SignatureBooster
from .cross_encoder_reranker import CrossEncoderReranker
from .pipeline import RetrievalPipeline, PipelineConfig

__all__ = [
    'PassageGenerator',
    'Normalizer',
    'parse_price',
    'parse_speed',
    'IntentClassifier',
    'Intent',
    'EntityDetector',
    'HybridRanker',
    'SignatureBooster',
    'CrossEncoderReranker',
    'RetrievalPipeline',
    'PipelineConfig'
]
