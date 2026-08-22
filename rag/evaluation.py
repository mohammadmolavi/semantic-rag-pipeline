"""Deterministic, LLM-free metrics for evaluating retrieval quality."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from langchain_core.documents import Document

from .retrieval import normalize_text


ChunkKey = tuple[str, int]


class Retriever(Protocol):
    def invoke(self, question: str) -> list[Document]: ...


@dataclass(frozen=True)
class RetrievalExample:
    question: str
    relevant_chunks: frozenset[ChunkKey]
    expected_document: str | None = None
    expected_contains: str | None = None
    expects_no_results: bool = False


@dataclass(frozen=True)
class RetrievalCaseResult:
    question: str
    expected_document: str | None
    relevant_chunks: frozenset[ChunkKey]
    retrieved_chunks: tuple[ChunkKey, ...]
    expects_no_results: bool
    first_relevant_rank: int | None
    hit_at_k: float
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float

    @property
    def negative_rejected(self) -> bool:
        return self.expects_no_results and not self.retrieved_chunks

    def as_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "expected_document": self.expected_document,
            "expects_no_results": self.expects_no_results,
            "relevant_chunks": [_format_chunk_key(key) for key in sorted(self.relevant_chunks)],
            "retrieved_chunks": [_format_chunk_key(key) for key in self.retrieved_chunks],
            "first_relevant_rank": self.first_relevant_rank,
            "hit_at_k": self.hit_at_k,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "negative_rejected": self.negative_rejected,
        }


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    top_k: int
    cases: tuple[RetrievalCaseResult, ...]

    @property
    def positive_cases(self) -> tuple[RetrievalCaseResult, ...]:
        return tuple(case for case in self.cases if not case.expects_no_results)

    @property
    def negative_cases(self) -> tuple[RetrievalCaseResult, ...]:
        return tuple(case for case in self.cases if case.expects_no_results)

    @property
    def hit_rate_at_k(self) -> float:
        return _mean(case.hit_at_k for case in self.positive_cases)

    @property
    def mrr_at_k(self) -> float:
        return _mean(case.reciprocal_rank for case in self.positive_cases)

    @property
    def mean_recall_at_k(self) -> float:
        return _mean(case.recall_at_k for case in self.positive_cases)

    @property
    def mean_precision_at_k(self) -> float:
        return _mean(case.precision_at_k for case in self.positive_cases)

    @property
    def negative_rejection_rate(self) -> float:
        return _mean(float(case.negative_rejected) for case in self.negative_cases)

    def as_dict(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "positive_queries": len(self.positive_cases),
            "negative_queries": len(self.negative_cases),
            f"hit_rate_at_{self.top_k}": self.hit_rate_at_k,
            f"mrr_at_{self.top_k}": self.mrr_at_k,
            f"mean_recall_at_{self.top_k}": self.mean_recall_at_k,
            f"mean_precision_at_{self.top_k}": self.mean_precision_at_k,
            "negative_rejection_rate": self.negative_rejection_rate,
            "cases": [case.as_dict() for case in self.cases],
        }


def chunk_key(document: Document) -> ChunkKey | None:
    source = document.metadata.get("source")
    chunk_index = document.metadata.get("chunk_index")

    if source is None or isinstance(chunk_index, bool):
        return None

    try:
        index = int(chunk_index)
    except (TypeError, ValueError):
        return None

    return str(source), index


def build_retrieval_examples(
    dataset: Sequence[Mapping[str, object]],
    corpus: Sequence[Document],
    *,
    source_filenames: Mapping[str, str] | None = None,
) -> tuple[RetrievalExample, ...]:
    filenames = source_filenames or {}
    examples: list[RetrievalExample] = []

    for position, item in enumerate(dataset, start=1):
        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError(f"Dataset item {position} has no question.")

        if item.get("expected_behavior") == "insufficient_context":
            examples.append(
                RetrievalExample(
                    question=question,
                    relevant_chunks=frozenset(),
                    expects_no_results=True,
                )
            )
            continue

        expected_document = str(item.get("document", "")).strip()
        expected_contains = str(item.get("expected_contains", "")).strip()
        if not expected_document or not expected_contains:
            raise ValueError(
                f"Dataset item {position} must define document and expected_contains."
            )

        normalized_expected = normalize_text(expected_contains).casefold()
        relevant_chunks: set[ChunkKey] = set()

        for document in corpus:
            key = chunk_key(document)
            if key is None:
                continue

            actual_filename = filenames.get(
                key[0],
                str(document.metadata.get("document_name", key[0])),
            )
            if not _filename_matches(actual_filename, expected_document):
                continue

            normalized_content = normalize_text(document.page_content).casefold()
            if normalized_expected in normalized_content:
                relevant_chunks.add(key)

        if not relevant_chunks:
            raise ValueError(
                "No relevant chunk was found for dataset item "
                f"{position}: {question!r} in {expected_document!r}."
            )

        examples.append(
            RetrievalExample(
                question=question,
                relevant_chunks=frozenset(relevant_chunks),
                expected_document=expected_document,
                expected_contains=expected_contains,
            )
        )

    return tuple(examples)


def evaluate_retriever(
    retriever: Retriever,
    examples: Sequence[RetrievalExample],
    *,
    top_k: int,
) -> RetrievalEvaluationReport:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    results: list[RetrievalCaseResult] = []

    for example in examples:
        retrieved_keys: list[ChunkKey] = []
        seen: set[ChunkKey] = set()

        for document in retriever.invoke(example.question):
            key = chunk_key(document)
            if key is None or key in seen:
                continue
            seen.add(key)
            retrieved_keys.append(key)
            if len(retrieved_keys) == top_k:
                break

        first_relevant_rank = next(
            (
                rank
                for rank, key in enumerate(retrieved_keys, start=1)
                if key in example.relevant_chunks
            ),
            None,
        )
        relevant_retrieved = len(set(retrieved_keys) & example.relevant_chunks)
        relevant_count = len(example.relevant_chunks)

        results.append(
            RetrievalCaseResult(
                question=example.question,
                expected_document=example.expected_document,
                relevant_chunks=example.relevant_chunks,
                retrieved_chunks=tuple(retrieved_keys),
                expects_no_results=example.expects_no_results,
                first_relevant_rank=first_relevant_rank,
                hit_at_k=float(first_relevant_rank is not None),
                recall_at_k=(
                    relevant_retrieved / relevant_count
                    if relevant_count
                    else 0.0
                ),
                precision_at_k=relevant_retrieved / top_k,
                reciprocal_rank=(
                    1.0 / first_relevant_rank
                    if first_relevant_rank is not None
                    else 0.0
                ),
            )
        )

    return RetrievalEvaluationReport(top_k=top_k, cases=tuple(results))


def _filename_matches(actual: str, expected: str) -> bool:
    actual_path = Path(actual)
    expected_path = Path(expected)

    if actual_path.name == expected_path.name:
        return True

    return (
        actual_path.suffix.casefold() == expected_path.suffix.casefold()
        and actual_path.stem.startswith(f"{expected_path.stem}_")
    )


def _format_chunk_key(key: ChunkKey) -> str:
    return f"{key[0]}#{key[1]}"


def _mean(values) -> float:
    collected = tuple(values)
    return sum(collected) / len(collected) if collected else 0.0
