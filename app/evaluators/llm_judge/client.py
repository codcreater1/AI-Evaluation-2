import os
from typing import Protocol

import httpx


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, prompt: str) -> str: ...


class OpenAICompatClient:
    """Minimal client for any OpenAI-compatible /chat/completions API (OpenAI, Gemini compat, vLLM)."""

    def __init__(self, api_key=None, base_url=None, model=None, timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY", "")
        default_url = "https://api.openai.com/v1"
        self.base_url = (base_url or os.environ.get("JUDGE_BASE_URL", default_url)).rstrip("/")
        self.model = model or os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    def complete(self, system: str, prompt: str) -> str:
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


_default_client: LLMClient | None = None


def set_default_client(client: LLMClient | None) -> None:
    global _default_client
    _default_client = client


def get_default_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = OpenAICompatClient()
    return _default_client
