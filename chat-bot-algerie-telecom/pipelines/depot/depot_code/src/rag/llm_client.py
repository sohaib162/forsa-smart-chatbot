# src/rag/llm_client.py
import os
from google import genai

GEMINI_CHAT_MODEL = "gemini-2.5-flash"  # fast + cheap, from official quickstart:contentReference[oaicite:3]{index=3}

_client = None


def _get_client() -> genai.Client:
    """Returns a singleton Gemini client configured with GEMINI_API_KEY."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Create a Gemini API key in Google AI Studio and export it "
                "as an environment variable."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def generate_answer(question: str, context: str) -> str:
    """
    Appelle Gemini pour générer une réponse en français
    en se basant sur le contexte RAG.
    """
    client = _get_client()

    prompt = f"""
Tu es un conseiller d'Algérie Télécom.
Utilise UNIQUEMENT le contexte ci-dessous pour répondre à la question du client.
Si une information n'est pas présente dans le contexte, dis-le clairement.

CONTEXT:
{context}

QUESTION:
{question}

Réponse détaillée en français:
""".strip()

    response = client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
    )
    # SDK returns `response.text` for the concatenated text output:contentReference[oaicite:4]{index=4}
    return (response.text or "").strip()
