import os
import re
from dataclasses import dataclass

import requests


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
PERSIAN_RE = re.compile(r"[\u0600-\u06FF].+", re.DOTALL)


def clean_llm_answer(text: str | None) -> str:
    content = (text or "").strip()
    if not content:
        return ""
    content = THINK_BLOCK_RE.sub("", content).strip()
    lower = content.lower()
    if "thinking process" in lower or "analyze user input" in lower:
        persian_blocks = PERSIAN_RE.findall(content)
        if persian_blocks:
            return max(persian_blocks, key=len).strip()
    for marker in ("Final answer:", "Final Answer:", "پاسخ نهایی:", "پاسخ:"):
        index = content.lower().rfind(marker.lower())
        if index != -1:
            return content[index + len(marker) :].strip()
    return content


@dataclass
class OpenRouterClient:
    api_key: str
    model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "OpenRouterClient":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        return cls(api_key=api_key, model=os.getenv("OPENROUTER_MODEL", cls.model))

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
            "reasoning": {"exclude": True},
        }
        try:
            data = self._post(payload)
        except RuntimeError as error:
            if "reasoning" in payload and "400" in str(error):
                payload.pop("reasoning", None)
                data = self._post(payload)
            else:
                raise
        message = data["choices"][0]["message"]
        return clean_llm_answer(message.get("content") or "")

    def _post(self, payload: dict) -> dict:
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(
                "OpenRouter API request failed "
                f"with status {response.status_code}: {response.text}"
            )
        return response.json()
