# ============================================================================
# tests/test_retrievers/test_three_layer.py
# ============================================================================

import pytest
from src.models.product_doc import ProductDoc, load_docs
from src.retrievers.three_layer import ThreeLayerRetriever


@pytest.fixture
def sample_docs():
    """Fixture: Sample product documents for testing"""
    data = [
        {
            "keywords": ["buzz", "smartphone", "5g"],
            "metadata": {"document_title": "Buzz 6 Pro"},
            "product_info": {
                "name": "Buzz 6 Pro",
                "category": "smartphone",
                "provider": "Algérie Télécom"
            },
            "commercial_details": {"pricing": "45000 DZD"},
        },
        {
            "keywords": ["zte", "blade", "android"],
            "metadata": {"document_title": "ZTE Blade A55"},
            "product_info": {
                "name": "ZTE Blade A55",
                "category": "smartphone",
                "provider": "ZTE"
            },
            "commercial_details": {"pricing": "25000 DZD"},
        },
        {
            "keywords": ["ibox", "cloud", "stockage"],
            "metadata": {"document_title": "iBox Cloud Storage"},
            "product_info": {
                "name": "iBox",
                "category": "cloud",
                "provider": "Algérie Télécom"
            },
            "commercial_details": {"pricing": "500 DZD/mois"},
        },
    ]
    return load_docs(data)


class TestThreeLayerRetriever:
    """Test suite for ThreeLayerRetriever"""
    
    def test_layer1_exact_match(self, sample_docs):
        """Test Layer 1 returns exact product match"""
        retriever = ThreeLayerRetriever(sample_docs)
        result = retriever.retrieve("buzz 6 pro")
        
        assert result is not None
        title, layer, score = result
        assert "Buzz 6" in title
        assert layer == "Layer 1 (Rule-based)"
        assert score == 1.0
    
    def test_layer1_blocked(self, sample_docs):
        """Test that blocking Layer 1 skips to Layer 2"""
        retriever = ThreeLayerRetriever(
            sample_docs,
            block_rule_layer=True
        )
        result = retriever.retrieve("buzz 6")
        
        assert result is not None
        title, layer, score = result
        assert layer != "Layer 1 (Rule-based)"
    
    def test_layer2_bm25(self, sample_docs):
        """Test Layer 2 (BM25) finds keyword matches"""
        retriever = ThreeLayerRetriever(
            sample_docs,
            block_rule_layer=True,  # Skip Layer 1
            block_dense_layer=True,  # Skip Layer 3
        )
        result = retriever.retrieve("smartphone android")
        
        assert result is not None
        title, layer, score = result
        assert layer == "Layer 2 (BM25)"
        assert score > 0
    
    def test_bm25_threshold(self, sample_docs):
        """Test BM25 score threshold blocks low scores"""
        retriever = ThreeLayerRetriever(
            sample_docs,
            block_rule_layer=True,
            bm25_score_threshold=100.0,  # Impossibly high
        )
        result = retriever.retrieve("smartphone")
        
        # Should move to Layer 3 or return None
        if result:
            title, layer, score = result
            assert layer == "Layer 3 (Dense)"
    
    def test_dense_layer_only(self, sample_docs):
        """Test using only Layer 3 (Dense)"""
        retriever = ThreeLayerRetriever(
            sample_docs,
            block_rule_layer=True,
            block_bm25_layer=True,
        )
        result = retriever.retrieve("stockage cloud")
        
        assert result is not None
        title, layer, score = result
        assert layer == "Layer 3 (Dense)"
    
    def test_no_match(self, sample_docs):
        """Test query with no matches returns None"""
        retriever = ThreeLayerRetriever(
            sample_docs,
            bm25_score_threshold=100.0,
            dense_score_threshold=0.99,
        )
        result = retriever.retrieve("xyz unknown product")
        
        # May return None or low confidence result
        assert result is None or result[2] < 0.5
    
    def test_cascade_flow(self, sample_docs):
        """Test that cascade works: Layer 1 → Layer 2 → Layer 3"""
        retriever = ThreeLayerRetriever(
            sample_docs,
            bm25_score_threshold=0.1,
            dense_score_threshold=0.1,
        )
        
        # This should find something in Layer 1
        result1 = retriever.retrieve("buzz 6")
        assert result1[1] == "Layer 1 (Rule-based)"
        
        # This should skip Layer 1, use Layer 2
        result2 = retriever.retrieve("smartphone pas cher")
        assert result2 is not None
        # Could be Layer 2 or Layer 3


# ============================================================================
# tests/test_retrievers/test_rules.py
# ============================================================================

import pytest
from src.models.product_doc import load_docs
from src.retrievers.rules import rule_based_filter, normalize


def test_normalize():
    """Test text normalization"""
    assert normalize("Buzz 6 Pro") == "buzz 6 pro"
    assert normalize("ZTE BLADE") == "zte blade"


def test_product_pattern_match():
    """Test matching specific product patterns"""
    data = [
        {
            "keywords": ["buzz", "smartphone"],
            "metadata": {"document_title": "Buzz 6"},
            "product_info": {"name": "Buzz 6", "category": "smartphone"},
        }
    ]
    docs = load_docs(data)
    
    results = rule_based_filter("buzz 6 pro", docs)
    assert len(results) > 0
    assert "buzz" in results[0].product_name.lower()


def test_category_match():
    """Test matching by category"""
    data = [
        {
            "keywords": ["mobile"],
            "metadata": {"document_title": "Generic Smartphone"},
            "product_info": {"name": "Phone", "category": "smartphone"},
        }
    ]
    docs = load_docs(data)
    
    results = rule_based_filter("smartphone android", docs)
    assert len(results) > 0


def test_no_match():
    """Test query with no matches returns empty list"""
    data = [
        {
            "keywords": ["cloud"],
            "metadata": {"document_title": "Cloud Service"},
            "product_info": {"name": "Cloud", "category": "cloud"},
        }
    ]
    docs = load_docs(data)
    
    results = rule_based_filter("xyz unknown", docs)
    assert len(results) == 0


# ============================================================================
# tests/conftest.py - Shared fixtures
# ============================================================================

import pytest
import os
from src.models.product_doc import load_docs


@pytest.fixture(scope="session")
def test_data_path():
    """Path to test data directory"""
    return os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def sample_products():
    """Full set of sample products for testing"""
    data = [
        {
            "keywords": ["buzz", "smartphone", "5g"],
            "metadata": {"document_title": "Buzz 6 Pro"},
            "product_info": {
                "name": "Buzz 6 Pro",
                "category": "smartphone",
                "provider": "Algérie Télécom"
            },
            "commercial_details": {"pricing": "45000 DZD"},
        },
        {
            "keywords": ["zte", "blade", "android"],
            "metadata": {"document_title": "ZTE Blade A55"},
            "product_info": {
                "name": "ZTE Blade A55",
                "category": "smartphone",
                "provider": "ZTE"
            },
            "commercial_details": {"pricing": "25000 DZD"},
        },
        {
            "keywords": ["ibox", "cloud", "stockage"],
            "metadata": {"document_title": "iBox Cloud Storage"},
            "product_info": {
                "name": "iBox",
                "category": "cloud",
                "provider": "Algérie Télécom"
            },
            "commercial_details": {"pricing": "500 DZD/mois"},
        },
        {
            "keywords": ["ekoteb", "éducation", "e-learning"],
            "metadata": {"document_title": "Ekoteb Platform"},
            "product_info": {
                "name": "Ekoteb",
                "category": "éducation",
                "provider": "Algérie Télécom"
            },
            "commercial_details": {"pricing": "1500 DZD/an"},
        },
    ]
    return load_docs(data)


# ============================================================================
# pytest.ini - Pytest configuration file (put in project root)
# ============================================================================

"""
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
"""


# ============================================================================
# How to run tests
# ============================================================================

"""
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest

# Run specific test file
pytest tests/test_retrievers/test_three_layer.py

# Run specific test
pytest tests/test_retrievers/test_three_layer.py::TestThreeLayerRetriever::test_layer1_exact_match

# Run with coverage
pytest --cov=src --cov-report=html

# Run only fast tests (skip slow ones)
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run and stop at first failure
pytest -x
"""
