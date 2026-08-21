import os
import re
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI



THINK_BLOCK_RE = re.compile(
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)

PERSIAN_RE = re.compile(
    r"[\u0600-\u06FF].+",
    re.DOTALL,
)


def clean_llm_answer(text: str | None) -> str:
    content = (text or "").strip()

    if not content:
        return ""

    content = THINK_BLOCK_RE.sub("", content).strip()

    lower = content.lower()

    if "thinking process" in lower or "analyze user input" in lower:
        persian_blocks = PERSIAN_RE.findall(content)

        if persian_blocks:
            return max(
                persian_blocks,
                key=len,
            ).strip()

    markers = (
        "Final answer:",
        "Final Answer:",
        "پاسخ نهایی:",
        "پاسخ:",
    )

    for marker in markers:
        index = content.lower().rfind(
            marker.lower()
        )

        if index != -1:
            return content[
                index + len(marker):
            ].strip()

    return content


@dataclass
class OpenRouterClient:
    api_key: str = field(
        repr=False
    )

    model: str = (
        "nvidia/nemotron-3-ultra-550b-a55b:free"
    )

    base_url: str = (
        "https://openrouter.ai/api/v1"
    )

    timeout: int = 60

    max_tokens: int = 400

    _chat_model: ChatOpenAI = field(
        init=False,
        repr=False,
    )

    _chain: Any = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._chat_model = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url.rstrip("/"),
            temperature=0.2,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            max_retries=2,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{system_prompt}",
                ),
                (
                    "human",
                    "{user_prompt}",
                ),
            ]
        )

        self._chain = (
            prompt
            | self._chat_model
            | StrOutputParser()
        )

    @classmethod
    def from_env(cls) -> "OpenRouterClient":
        env_path = (
                Path(__file__)
                .resolve()
                .parent
                .parent
                / ".env"
        )

        load_dotenv(
            dotenv_path=env_path,
            override=False,
        )

        api_key = os.getenv(
            "OPENROUTER_API_KEY",
            "",
        ).strip()

        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                f"Checked environment variables and: {env_path}"
            )

        return cls(
            api_key=api_key,
            model=os.getenv(
                "OPENROUTER_MODEL",
                cls.model,
            ),
            base_url=os.getenv(
                "OPENROUTER_BASE_URL",
                cls.base_url,
            ),
        )

    def answer(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        try:
            answer = self._chain.invoke(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )

        except Exception as error:
            raise RuntimeError(
                "OpenRouter request failed "
                f"for model '{self.model}': {error}"
            ) from error

        return clean_llm_answer(answer)