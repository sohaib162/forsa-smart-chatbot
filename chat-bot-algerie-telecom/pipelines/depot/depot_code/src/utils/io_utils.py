import json
from pathlib import Path
from typing import Any, List, Dict

import yaml


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_products(path: str | Path) -> List[Dict[str, Any]]:
    """
    Charge le dataset JSON des produits.
    """
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Le fichier products.json doit contenir une liste JSON.")
    return data


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
