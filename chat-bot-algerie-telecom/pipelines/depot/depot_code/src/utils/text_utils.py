import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalisation basique: lowercase, strip, remplacement espaces multiples.
    """
    if text is None:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
