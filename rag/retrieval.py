import math
import re
import unicodedata

from collections import Counter
from dataclasses import dataclass


TOKEN_RE = re.compile(
    r"[\w\u0600-\u06ff]+",
    re.UNICODE,
)

DIACRITICS_RE = re.compile(
    r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]"
)

WHITESPACE_RE = re.compile(
    r"\s+"
)

CHARACTER_TRANSLATIONS = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ە": "ه",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ـ": None,

        "\u200c": " ",
        "\u200d": " ",
        "\ufeff": " ",

        "\u200e": None,
        "\u200f": None,

        "\u202a": None,
        "\u202b": None,
        "\u202c": None,
        "\u202d": None,
        "\u202e": None,

        "\u2066": None,
        "\u2067": None,
        "\u2068": None,
        "\u2069": None,

        **{
            digit: str(index)
            for index, digit in enumerate(
                "۰۱۲۳۴۵۶۷۸۹"
            )
        },

        **{
            digit: str(index)
            for index, digit in enumerate(
                "٠١٢٣٤٥٦٧٨٩"
            )
        },
    }
)


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source: str = ""
    index: int = 0
    score: float = 0.0


def normalize_text(
    text: str,
) -> str:
    """Normalize Persian and Arabic text for lexical matching only."""

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    normalized = normalized.translate(
        CHARACTER_TRANSLATIONS
    )

    normalized = DIACRITICS_RE.sub(
        "",
        normalized,
    )

    return WHITESPACE_RE.sub(
        " ",
        normalized,
    ).strip()


def tokenize(
    text: str,
) -> list[str]:
    normalized = normalize_text(
        text
    )

    return [
        token.casefold()
        for token in TOKEN_RE.findall(
            normalized
        )
    ]


def split_text(
    text: str,
    *,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[str]:
    cleaned = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not cleaned:
        return []

    if chunk_size <= overlap:
        raise ValueError(
            "chunk_size must be greater than overlap."
        )

    chunks: list[str] = []

    start = 0

    while start < len(cleaned):
        end = min(
            start + chunk_size,
            len(cleaned),
        )

        chunks.append(
            cleaned[start:end].strip()
        )

        if end == len(cleaned):
            break

        start = end - overlap

    return chunks


class KeywordRetriever:
    """Dependency-free keyword retriever based on TF-IDF cosine similarity."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        self.chunks = chunks

        self._term_counts = [
            Counter(
                tokenize(
                    chunk.text
                )
            )
            for chunk in chunks
        ]

        self._idf = self._build_idf(
            self._term_counts
        )

        self._vectors = [
            self._tfidf_vector(
                counts
            )
            for counts in self._term_counts
        ]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        *,
        source: str = "",
    ) -> "KeywordRetriever":
        chunks = [
            DocumentChunk(
                text=text,
                source=source,
                index=index,
            )
            for index, text in enumerate(texts)
            if text.strip()
        ]

        return cls(
            chunks
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
    ) -> list[DocumentChunk]:
        query_counts = Counter(
            tokenize(
                query
            )
        )

        query_vector = self._tfidf_vector(
            query_counts
        )

        scored = []

        for chunk, vector in zip(
            self.chunks,
            self._vectors,
            strict=True,
        ):
            score = self._cosine(
                query_vector,
                vector,
            )

            if score > 0:
                scored.append(
                    DocumentChunk(
                        text=chunk.text,
                        source=chunk.source,
                        index=chunk.index,
                        score=score,
                    )
                )

        return sorted(
            scored,
            key=lambda item: item.score,
            reverse=True,
        )[:top_k]

    @staticmethod
    def _build_idf(
        term_counts: list[Counter[str]],
    ) -> dict[str, float]:
        document_count = len(
            term_counts
        )

        document_frequency: Counter[str] = Counter()

        for counts in term_counts:
            document_frequency.update(
                counts.keys()
            )

        return {
            term: math.log(
                (1 + document_count)
                / (1 + frequency)
            ) + 1
            for term, frequency in document_frequency.items()
        }

    def _tfidf_vector(
        self,
        counts: Counter[str],
    ) -> dict[str, float]:
        total = sum(
            counts.values()
        ) or 1

        return {
            term: (
                count / total
            ) * self._idf.get(
                term,
                1.0,
            )
            for term, count in counts.items()
        }

    @staticmethod
    def _cosine(
        left: dict[str, float],
        right: dict[str, float],
    ) -> float:
        common_terms = set(
            left
        ) & set(
            right
        )

        numerator = sum(
            left[term] * right[term]
            for term in common_terms
        )

        left_norm = math.sqrt(
            sum(
                value * value
                for value in left.values()
            )
        )

        right_norm = math.sqrt(
            sum(
                value * value
                for value in right.values()
            )
        )

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return numerator / (
            left_norm * right_norm
        )