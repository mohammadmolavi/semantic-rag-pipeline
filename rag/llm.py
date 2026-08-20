import os
from dataclasses import dataclass

import requests


@dataclass
class OpenRouterClient:
    api_key: str
    model: str = "poolside/laguna-s-2.1:free"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "OpenRouterClient":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        return cls(api_key=api_key, model=os.getenv("OPENROUTER_MODEL", cls.model))

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 700,
            },
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(
                "OpenRouter API request failed "
                f"with status {response.status_code}: {response.text}"
            )
        data = response.json()
        return data["choices"][0]["message"]["content"]
