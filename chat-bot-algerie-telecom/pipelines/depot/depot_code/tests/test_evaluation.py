# ============================================================================
# tests/test_evaluation.py
# FINAL – CONFIGURATION-AWARE, SINGLE-RESULT RETRIEVAL
# ============================================================================

import pytest
from typing import List, Dict
from src.models.product_doc import load_docs
from src.retrievers.three_layer import ThreeLayerRetriever


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def normalize_title(title: str) -> str:
    return title.lower().strip() if title else ""


# ---------------------------------------------------------------------------
# Metrics (SINGLE RESULT RETRIEVAL)
# ---------------------------------------------------------------------------

class RetrievalMetrics:
    """
    Metrics adapted to a retriever that returns ONLY ONE document per query.
    """

    @staticmethod
    def accuracy(results: List[Dict]) -> float:
        return sum(1 for r in results if r["is_correct"]) / len(results)

    @staticmethod
    def coverage(results: List[Dict]) -> float:
        """% of queries that returned ANY document"""
        return sum(1 for r in results if r["predicted_title"] is not None) / len(results)

    @staticmethod
    def average_score(results: List[Dict]) -> float:
        scores = [r["score"] for r in results if r["score"] is not None]
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def layer_accuracy(results: List[Dict]) -> Dict[str, float]:
        layers = {}
        for r in results:
            layer = r["predicted_layer"]
            layers.setdefault(layer, {"ok": 0, "tot": 0})
            layers[layer]["tot"] += 1
            if r["is_correct"]:
                layers[layer]["ok"] += 1

        return {
            layer: data["ok"] / data["tot"]
            for layer, data in layers.items()
            if data["tot"] > 0
        }

    @staticmethod
    def layer_confusion(results: List[Dict]) -> Dict[str, int]:
        return {
            "correct_layer": sum(1 for r in results if r["is_layer_correct"]),
            "wrong_layer": sum(1 for r in results if not r["is_layer_correct"]),
        }


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestRetrievalEvaluation:

    # -----------------------------------------------------------------------
    # Test data
    # -----------------------------------------------------------------------

    @pytest.fixture
    def test_queries(self):
      return [
              {
                  "id": 1,
                  "query": "buzz 6 pro prix",
                  "expected_title": "Buzz 6 Pro",
                  "expected_layer": "Layer 1 (Rule-based)",
                  "category": "exact",
                  "difficulty": "easy",
              },
              {
                  "id": 2,
                  "query": "zte blade a55 caractéristiques",
                  "expected_title": "ZTE Blade A55",
                  "expected_layer": "Layer 1 (Rule-based)",
                  "category": "exact",
                  "difficulty": "easy",
              },
              {
                  "id": 3,
                  "query": "ibox stockage cloud",
                  "expected_title": "iBox Cloud Storage",
                  "expected_layer": "Layer 1 (Rule-based)",
                  "category": "exact",
                  "difficulty": "easy",
              },
              {
                  "id": 4,
                  "query": "smartphone pas cher",
                  "expected_title": "ZTE Blade A35",
                  "expected_layer": "Layer 2 (BM25)",
                  "category": "generic",
                  "difficulty": "medium",
              },
              {
                  "id": 5,
                  "query": "solution de sauvegarde cloud",
                  "expected_title": "iBox Cloud Storage",
                  "expected_layer": "Layer 3 (Dense)",
                  "category": "semantic",
                  "difficulty": "hard",
              },
              {
                  "id": 6,
                  "query": "cache modem bois",
                  "expected_title": "Caches modems",
                  "expected_layer": "Layer 2 (BM25)",
                  "category": "feature_specific",
                  "difficulty": "medium",
              },
              {
                  "id": 7,
                  "query": "comment cacher mon routeur wifi salon",
                  "expected_title": "Caches modems",
                  "expected_layer": "Layer 3 (Dense)",
                  "category": "problem_solving",
                  "difficulty": "hard",
              },
              {
                  "id": 8,
                  "query": "préparation bac 2025 en ligne",
                  "expected_title": "MOALIM",
                  "expected_layer": "Layer 3 (Dense)",
                  "category": "semantic",
                  "difficulty": "hard",
              },
              {
                  "id": 9,
                  "query": "twin box 4k",
                  "expected_title": "TWIN BOX",
                  "expected_layer": "Layer 1 (Rule-based)",
                  "category": "exact",
                  "difficulty": "easy",
              },
              {
                  "id": 10,
                  "query": "téléphone pliable 24 go ram",
                  "expected_title": "Buzz 6 Flip",
                  "expected_layer": "Layer 2 (BM25)",
                  "category": "spec_specific",
                  "difficulty": "medium",
              },
              {
                  "id": 11,
                  "query": "plateforme gestion école privée",
                  "expected_title": "Plateforme ClassaTeck",
                  "expected_layer": "Layer 3 (Dense)",
                  "category": "b2b_intent",
                  "difficulty": "hard",
              },
              {
                  "id": 12,
                  "query": "application lecture livres algérie",
                  "expected_title": "Bibliothèque numérique EKOTEB",
                  "expected_layer": "Layer 3 (Dense)",
                  "category": "semantic",
                  "difficulty": "medium",
              },
              {
                  "id": 13,
                  "query": "dorouscom code 16 chiffres",
                  "expected_title": "Cartes de recharge Dorouscom",
                  "expected_layer": "Layer 2 (BM25)",
                  "category": "technical_detail",
                  "difficulty": "medium",
              },
              {
                  "id": 14,
                  "query": "suivi colis idoom market",
                  "expected_title": "Boutique en ligne Idoom Market",
                  "expected_layer": "Layer 1 (Rule-based)",
                  "category": "service_lookup",
                  "difficulty": "easy",
              },
              {
                  "id": 15,
                  "query": "box tv android avec puce",
                  "expected_title": "TWIN BOX",
                  "expected_layer": "Layer 3 (Dense)",
                  "category": "semantic_description",
                  "difficulty": "hard",
              }
          ]


    @pytest.fixture
    def sample_docs(self):
        data = [
            {
                "keywords": ["buzz", "smartphone"],
                "metadata": {"document_title": "Buzz 6 Pro"},
                "product_info": {"name": "Buzz 6 Pro"},
            },
            {
                "keywords": ["zte", "android"],
                "metadata": {"document_title": "ZTE Blade A55"},
                "product_info": {"name": "ZTE Blade A55"},
            },
            {
                "keywords": ["ibox", "cloud", "stockage"],
                "metadata": {"document_title": "iBox Cloud Storage"},
                "product_info": {"name": "iBox"},
            },
        ]
        return load_docs(data)

    # -----------------------------------------------------------------------
    # MAIN EVALUATION TEST
    # -----------------------------------------------------------------------

    def test_full_evaluation(self, test_queries, sample_docs):

        # IMPORTANT:
        # We keep default configuration to respect your retriever behavior
        retriever = ThreeLayerRetriever(
            sample_docs,

            verbose=False,
        )

        results = []

        print("\n" + "=" * 80)
        print("🧪 RETRIEVAL SYSTEM EVALUATION")
        print("=" * 80)

        for tc in test_queries:
            query = tc["query"]
            expected_title = tc["expected_title"]
            expected_layer = tc["expected_layer"]

            result = retriever.retrieve(query)

            if result:
                predicted_title, predicted_layer, score, *_ = result
            else:
                predicted_title = None
                predicted_layer = None
                score = 0.0

            is_correct = (
                normalize_title(predicted_title)
                == normalize_title(expected_title)
            )

            is_layer_correct = predicted_layer == expected_layer

            results.append({
                "query": query,
                "expected_title": expected_title,
                "predicted_title": predicted_title,
                "expected_layer": expected_layer,
                "predicted_layer": predicted_layer,
                "score": score,
                "is_correct": is_correct,
                "is_layer_correct": is_layer_correct,
                "category": tc["category"],
                "difficulty": tc["difficulty"],
            })

            status = "✅" if is_correct else "❌"
            print(f"\n{status} {query}")
            print(f"   Expected: {expected_title} @ {expected_layer}")
            print(f"   Got:      {predicted_title} @ {predicted_layer} (score={score:.3f})")

        # -------------------------------------------------------------------
        # Metrics & Reporting
        # -------------------------------------------------------------------

        metrics = RetrievalMetrics()

        accuracy = metrics.accuracy(results)
        coverage = metrics.coverage(results)
        avg_score = metrics.average_score(results)
        layer_conf = metrics.layer_confusion(results)

        print("\n" + "=" * 80)
        print("📊 METRICS")
        print("=" * 80)

        print(f"Accuracy@1 : {accuracy:.2%}")
        print(f"Coverage   : {coverage:.2%}")
        print(f"Avg score  : {avg_score:.3f}")

        print("\n🧠 Layer Accuracy:")
        for layer, acc in metrics.layer_accuracy(results).items():
            print(f"   {layer}: {acc:.2%}")

        print("\n⚠️ Layer Confusion:")
        print(layer_conf)

        # -------------------------------------------------------------------
        # ASSERTIONS (CONFIGURATION-AWARE)
        # -------------------------------------------------------------------

        assert accuracy >= 0.6, "Accuracy below 60%"
        assert coverage == 1.0, "Some queries returned no document"

        # Layer correctness is ONLY meaningful if layers are enabled
        if not retriever.block_rule_layer or not retriever.block_bm25_layer:
            assert layer_conf["wrong_layer"] <= len(results) * 0.4
        else:
            print("\nℹ️ Layer accuracy check skipped (layers blocked by configuration)")
