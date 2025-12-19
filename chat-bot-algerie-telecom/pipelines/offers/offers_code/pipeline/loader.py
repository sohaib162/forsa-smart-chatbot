"""
Document loader module for Algérie Télécom JSON documents.

This module handles loading and normalizing document data from JSON files.
"""

import json
import os
import re
import string
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent matching and retrieval.

    DEPRECATED: Use text_normalization.normalize_text_multilingual() instead.
    This function is kept for backward compatibility but delegates to the
    new multilingual normalization module.

    Performs:
    - Lowercasing
    - Whitespace normalization
    - Arabic diacritics removal
    - Letter variant normalization

    Args:
        text: Input text string

    Returns:
        Normalized text string
    """
    if not text:
        return ""

    # Import here to avoid circular dependency
    try:
        from .text_normalization import normalize_text_multilingual
        return normalize_text_multilingual(text)
    except ImportError:
        # Fallback to simple normalization if text_normalization not available
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text


def load_documents(data_dir: str) -> List[Dict]:
    """
    Load all JSON documents from a directory.

    This function reads all *.json files from the specified directory
    and returns them as a list of dictionaries. Each document is expected
    to follow the Algérie Télécom document schema.

    Args:
        data_dir: Path to directory containing JSON documents

    Returns:
        List of document dictionaries

    Raises:
        FileNotFoundError: If data_dir does not exist
        ValueError: If no JSON files found in directory
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    if not data_path.is_dir():
        raise ValueError(f"Path is not a directory: {data_dir}")

    # Find all JSON files
    json_files = list(data_path.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in directory: {data_dir}")

    logger.info(f"Found {len(json_files)} JSON files in {data_dir}")

    documents = []
    failed_files = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                doc = json.load(f)

                # Add file path for reference
                doc['_file_path'] = str(json_file)

                # Validate that it has expected structure
                if 'document_id' not in doc:
                    logger.warning(f"Document missing 'document_id': {json_file.name}")

                documents.append(doc)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {json_file.name}: {e}")
            failed_files.append(json_file.name)

        except Exception as e:
            logger.error(f"Error loading file {json_file.name}: {e}")
            failed_files.append(json_file.name)

    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")

    logger.info(f"Successfully loaded {len(documents)} documents")

    return documents


def safe_get(doc: Dict, *keys: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safely retrieve a nested value from a document dictionary.

    Handles None values and missing keys gracefully.

    Args:
        doc: Document dictionary
        *keys: Sequence of keys to traverse (e.g., 'metadata', 'title_fr')
        default: Default value if key not found or value is None

    Returns:
        Retrieved value or default

    Example:
        >>> safe_get(doc, 'metadata', 'title_fr', default='Unknown')
    """
    value = doc
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default

        if value is None:
            return default

    return value if value else default


def safe_get_list(doc: Dict, *keys: str, default: Optional[List] = None) -> List:
    """
    Safely retrieve a list value from a document dictionary.

    Args:
        doc: Document dictionary
        *keys: Sequence of keys to traverse
        default: Default value if key not found or value is None/empty

    Returns:
        Retrieved list or default (empty list if default is None)
    """
    if default is None:
        default = []

    value = safe_get(doc, *keys, default=None)

    if value is None:
        return default

    if isinstance(value, list):
        return value

    # If it's a single value, wrap it in a list
    return [value]
