from dataclasses import dataclass

from .llm import OpenRouterClient
from .retrieval import DocumentChunk, KeywordRetriever


SYSTEM_PROMPT = """You are a Persian RAG assistant.
Answer only from the provided context.
If the context is not enough, say that the uploaded documents do not contain enough information.
Keep the answer concise and cite the used chunk numbers."""


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[DocumentChunk]


class RagPipeline:
    def __init__(self, retriever: KeywordRetriever, llm: OpenRouterClient) -> None:
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

Answer in Persian."""
