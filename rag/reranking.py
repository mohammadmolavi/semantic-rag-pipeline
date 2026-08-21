"""Optional cross-encoder reranking for hybrid retrieval candidates."""

from __future__ import annotations

import logging
import os

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

LOGGER = logging.getLogger(
    __name__
)


@lru_cache(
    maxsize=4
)
def _load_cross_encoder(
    model_name: str,
    device: str | None,
) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name,
        device=device,
    )


@dataclass(
    frozen=True
)
class CrossEncoderReranker:
    model_name: str = DEFAULT_RERANKER_MODEL

    device: str | None = None

    batch_size: int = 16

    def __post_init__(
        self,
    ) -> None:
        if not self.model_name.strip():
            raise ValueError(
                "Reranker model name cannot be empty."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "Reranker batch_size must be greater than zero."
            )

    @classmethod
    def from_env(
        cls,
    ) -> "CrossEncoderReranker | None":
        enabled = os.getenv(
            "RERANKER_ENABLED",
            "true",
        ).strip().casefold()

        if enabled not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return None

        model_name = os.getenv(
            "RERANKER_MODEL",
            DEFAULT_RERANKER_MODEL,
        ).strip()

        device = (
            os.getenv(
                "RERANKER_DEVICE",
                "",
            ).strip()
            or None
        )

        raw_batch_size = os.getenv(
            "RERANKER_BATCH_SIZE",
            "16",
        ).strip()

        try:
            batch_size = int(
                raw_batch_size
            )

        except ValueError as error:
            raise ValueError(
                "RERANKER_BATCH_SIZE must be a positive integer."
            ) from error

        return cls(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )

    def rerank(
        self,
        question: str,
        documents: list[Document],
    ) -> list[Document]:
        candidates = list(
            documents
        )

        if len(candidates) <= 1:
            return candidates

        pairs = [
            (
                question,
                document.page_content,
            )
            for document in candidates
        ]

        try:
            model = _load_cross_encoder(
                self.model_name,
                self.device,
            )

            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            scores = [
                float(
                    score
                )
                for score in raw_scores
            ]

            if len(scores) != len(candidates):
                raise ValueError(
                    "Reranker returned a different number "
                    "of scores than candidates."
                )

            ranked_documents = sorted(
                zip(
                    candidates,
                    scores,
                    strict=True,
                ),
                key=lambda item: item[1],
                reverse=True,
            )

            return [
                document
                for document, _ in ranked_documents
            ]

        except Exception as error:
            LOGGER.warning(
                "Cross-encoder reranking failed for model %s; "
                "preserving RRF order: %s",
                self.model_name,
                error,
            )

            return candidates