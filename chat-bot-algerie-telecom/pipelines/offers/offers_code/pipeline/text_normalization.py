"""
Text normalization utilities for multilingual (French + Arabic) queries and documents.

This module provides language detection and normalization functions that work
consistently across the pipeline (rule router, sparse BM25, dense embeddings).

Key features:
- Language detection (French, Arabic, or mixed)
- Arabic text normalization (diacritics, letter variants, tatweel)
- French text normalization (lowercase, accents, punctuation)
- Preserves Latin acronyms like "4G LTE", "ONT", "ADSL" in Arabic text
"""

import re
import unicodedata
from typing import Tuple


def detect_language(text: str) -> str:
    """
    Detect language of text based on character composition.

    Args:
        text: Input text string

    Returns:
        Language code: 'ar' (Arabic), 'fr' (French), or 'mixed'
    """
    if not text:
        return 'fr'  # default

    # Count Arabic characters (Unicode range: 0600-06FF, 0750-077F, 08A0-08FF, FB50-FDFF, FE70-FEFF)
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))

    # Count Latin characters
    latin_chars = len(re.findall(r'[a-zA-ZÀ-ÿ]', text))

    total = arabic_chars + latin_chars
    if total == 0:
        return 'fr'

    arabic_ratio = arabic_chars / total

    if arabic_ratio > 0.7:
        return 'ar'
    elif arabic_ratio < 0.3:
        return 'fr'
    else:
        return 'mixed'


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for consistent matching and retrieval.

    Normalization steps:
    1. Remove diacritics (tashkeel): ً ٌ ٍ َ ُ ِ ّ ْ ـ
    2. Normalize alef variants: أ إ آ → ا
    3. Normalize yaa variants: ى → ي
    4. Normalize taa marbouta: ة → ه
    5. Remove tatweel (ـ) and extra punctuation
    6. Preserve Latin acronyms and numbers
    7. Normalize whitespace

    Args:
        text: Input Arabic text

    Returns:
        Normalized Arabic text
    """
    if not text:
        return ""

    # Remove Arabic diacritics (tashkeel)
    diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    text = diacritics.sub('', text)

    # Normalize alef variants
    text = re.sub(r'[أإآٱ]', 'ا', text)

    # Normalize yaa variants
    text = re.sub(r'ى', 'ي', text)

    # Normalize taa marbouta (optional, can be disabled if it causes issues)
    text = re.sub(r'ة', 'ه', text)

    # Remove tatweel (character stretching)
    text = re.sub(r'ـ', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def normalize_french(text: str) -> str:
    """
    Normalize French/Latin text for consistent matching and retrieval.

    Normalization steps:
    1. Convert to lowercase
    2. Normalize accents (optional - can preserve for better matching)
    3. Remove excessive punctuation
    4. Normalize whitespace

    Args:
        text: Input French text

    Returns:
        Normalized French text
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Optional: normalize accents (we'll preserve them for better matching)
    # Uncomment below if you want to remove accents:
    # text = unicodedata.normalize('NFKD', text)
    # text = ''.join([c for c in text if not unicodedata.combining(c)])

    # Remove excessive punctuation but keep basic ones
    # Preserve hyphens and apostrophes which are meaningful in French
    text = re.sub(r'[^\w\s\-\'àâäéèêëïîôùûüÿæœç]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def normalize_text_multilingual(text: str, language: str = None) -> str:
    """
    Normalize text based on detected or specified language.

    This is the main normalization function that should be used throughout
    the pipeline for consistent text processing.

    Args:
        text: Input text
        language: Optional language hint ('ar', 'fr', 'mixed'). If None, auto-detect.

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Detect language if not specified
    if language is None:
        language = detect_language(text)

    if language == 'ar':
        # Pure Arabic text
        return normalize_arabic(text)
    elif language == 'fr':
        # Pure French text
        return normalize_french(text)
    else:
        # Mixed text - apply both normalizations
        # First normalize Arabic parts, then French parts
        # This preserves Latin acronyms in Arabic text
        text_ar = normalize_arabic(text)
        text_fr = normalize_french(text_ar)
        return text_fr


def tokenize_multilingual(text: str) -> list:
    """
    Tokenize multilingual text for BM25 and matching.

    Works with French, Arabic, and mixed text.
    Preserves meaningful multi-character tokens like "4g", "lte", "ont", etc.

    Args:
        text: Input text (already normalized)

    Returns:
        List of tokens
    """
    if not text:
        return []

    # Split on whitespace and punctuation, but preserve alphanumeric sequences
    tokens = re.findall(r'\w+', text, re.UNICODE)

    # Filter out single-character tokens that are not meaningful
    # (but keep numbers)
    tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]

    return tokens


def extract_query_keywords(query: str) -> Tuple[str, list, str]:
    """
    Extract normalized query text, tokens, and detected language.

    This is a convenience function for the pipeline to get all query
    information at once.

    Args:
        query: Raw user query

    Returns:
        Tuple of (normalized_text, tokens, language)
    """
    language = detect_language(query)
    normalized = normalize_text_multilingual(query, language)
    tokens = tokenize_multilingual(normalized)

    return normalized, tokens, language


# ============================================================================
# Utility functions for matching
# ============================================================================

def simple_normalize(text: str) -> str:
    """
    Simple normalization for title matching (used in evaluation).

    This is a lightweight version used for matching document titles
    to labeled query titles in evaluation and query augmentation.

    Args:
        text: Input text

    Returns:
        Simple normalized text (lowercase + whitespace cleanup)
    """
    if not text:
        return ""

    text = text.lower()

    # Remove Arabic diacritics
    diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    text = diacritics.sub('', text)

    # Normalize common Arabic variants
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
