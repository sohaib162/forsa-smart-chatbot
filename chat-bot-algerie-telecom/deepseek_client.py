import os
import requests

API_KEY = os.getenv("DEEPSEEK_API_KEY", "hl19FXcq7Aw0POD9SzqNFvTmh7U4WPSq_YV1qx_ADGi4Z82UYU-MMGzUCGQx1jX3S7z82grSpFHzVbWyAppr3w").strip().strip('"').strip("'")
API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.modelarts-maas.com/v2/chat/completions")

def call_deepseek(system_prompt: str, user_content: str):
    if not API_KEY:
        return "ERROR: Missing DEEPSEEK_API_KEY environment variable."

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "deepseek-v3.1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if not r.ok:
            print("API Error Response:", r.text)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"DEBUG ERROR: {e}"
