from dataclasses import dataclass
from typing import Protocol

from .llm import OpenRouterClient
from .retrieval import DocumentChunk


SYSTEM_PROMPT = """You are a Persian RAG assistant.
Write only the final answer in Persian.
Do not write analysis, thinking, English, or source JSON.
Answer only from the provided context.
If the context is not enough, say that the uploaded documents do not contain enough information.
Keep the answer concise and cite chunk numbers, for example (بخش ۰)."""


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[DocumentChunk]


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 4) -> list[DocumentChunk]:
        pass


class RagPipeline:
    def __init__(self, retriever: Retriever, llm: OpenRouterClient) -> None:
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str, *, top_k: int = 4) -> RagAnswer:
        sources = self.retriever.search(question, top_k=top_k)
        prompt = self._build_prompt(question, sources)
        answer = self.llm.answer(SYSTEM_PROMPT, prompt)
        return RagAnswer(answer=answer, sources=sources)

    @staticmethod
    def _build_prompt(question: str, sources: list[DocumentChunk]) -> str:
        context = "\n\n".join(
            f"[chunk {chunk.index} | source: {chunk.source or 'unknown'}]\n{chunk.text}"
            for chunk in sources
        )
        return f"""Context:
{context or "No relevant context was found."}

Question:
{question}

Write the Persian answer only."""
