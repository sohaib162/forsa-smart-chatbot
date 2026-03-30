"""
DeepSeek LLM Client
====================
External LLM fallback via DeepSeek API.

IMPORTANT: Set DEEPSEEK_API_KEY environment variable before use.
Never hardcode API keys in source code.
"""
import os
import logging
import requests

logger = logging.getLogger(__name__)

API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
API_URL = os.getenv(
    "DEEPSEEK_API_URL",
    "https://api.modelarts-maas.com/v2/chat/completions",
)
_REQUEST_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "30"))


def call_deepseek(system_prompt: str, user_content: str) -> str:
    """Call the DeepSeek API with the given prompts.

    Returns the model response text, or an error string prefixed with 'ERROR:'.
    """
    if not API_KEY:
        logger.error("DEEPSEEK_API_KEY not set — cannot call DeepSeek API.")
        return "ERROR: Missing DEEPSEEK_API_KEY environment variable."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": "deepseek-v3.1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }

    try:
        r = requests.post(
            API_URL, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        logger.error("DeepSeek API request timed out after %ds", _REQUEST_TIMEOUT)
        return "ERROR: DeepSeek API request timed out."
    except requests.exceptions.HTTPError as e:
        logger.error("DeepSeek API HTTP error: %s — %s", e, r.text)
        return f"ERROR: DeepSeek API HTTP error: {e}"
    except Exception as e:
        logger.exception("DeepSeek API unexpected error")
        return f"ERROR: DeepSeek API error: {e}"
