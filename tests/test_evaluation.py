import unittest

from langchain_core.documents import Document

from rag.evaluation import (
    RetrievalExample,
    build_retrieval_examples,
    evaluate_retriever,
)


def make_document(source: str, index: int, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "chunk_index": index},
    )


class MappingRetriever:
    def __init__(self, results: dict[str, list[Document]]) -> None:
        self.results = results

    def invoke(self, question: str) -> list[Document]:
        return list(self.results.get(question, []))


class RetrievalEvaluationTests(unittest.TestCase):
    def test_metrics_are_computed_only_from_retrieved_chunk_identity(self) -> None:
        relevant_a = make_document("document:1", 0, "answer a")
        distractor = make_document("document:2", 0, "distractor")
        retriever = MappingRetriever(
            {
                "first": [distractor, relevant_a],
                "second": [distractor],
                "negative": [],
            }
        )
        examples = (
            RetrievalExample(
                question="first",
                relevant_chunks=frozenset(
                    {("document:1", 0), ("document:1", 1)}
                ),
            ),
            RetrievalExample(
                question="second",
                relevant_chunks=frozenset({("document:1", 1)}),
            ),
            RetrievalExample(
                question="negative",
                relevant_chunks=frozenset(),
                expects_no_results=True,
            ),
        )

        report = evaluate_retriever(retriever, examples, top_k=2)

        self.assertEqual(report.hit_rate_at_k, 0.5)
        self.assertEqual(report.mrr_at_k, 0.25)
        self.assertEqual(report.mean_recall_at_k, 0.25)
        self.assertEqual(report.mean_precision_at_k, 0.25)
        self.assertEqual(report.negative_rejection_rate, 1.0)

    def test_duplicate_retrieved_chunks_do_not_inflate_metrics(self) -> None:
        relevant = make_document("document:1", 0, "answer")
        retriever = MappingRetriever({"question": [relevant, relevant]})
        example = RetrievalExample(
            question="question",
            relevant_chunks=frozenset({("document:1", 0)}),
        )

        report = evaluate_retriever(retriever, (example,), top_k=2)

        self.assertEqual(report.cases[0].retrieved_chunks, (("document:1", 0),))
        self.assertEqual(report.mean_recall_at_k, 1.0)
        self.assertEqual(report.mean_precision_at_k, 0.5)

    def test_dataset_labels_match_normalized_persian_and_storage_suffixes(self) -> None:
        corpus = [
            make_document(
                "document:7",
                3,
                "مبلغ ماهانه پلن حرفه ای ٤٩٠٬٠٠٠ تومان است.",
            )
        ]
        dataset = [
            {
                "question": "قیمت چقدر است؟",
                "document": "pricing.docx",
                "expected_contains": "۴۹۰٬۰۰۰ تومان",
            },
            {
                "question": "پایتخت چین چیست؟",
                "document": None,
                "expected_behavior": "insufficient_context",
            },
        ]

        examples = build_retrieval_examples(
            dataset,
            corpus,
            source_filenames={"document:7": "pricing_ab12.docx"},
        )

        self.assertEqual(examples[0].relevant_chunks, frozenset({("document:7", 3)}))
        self.assertTrue(examples[1].expects_no_results)

    def test_dataset_with_missing_relevant_chunk_is_rejected(self) -> None:
        dataset = [
            {
                "question": "missing",
                "document": "sample.docx",
                "expected_contains": "not present",
            }
        ]

        with self.assertRaisesRegex(ValueError, "No relevant chunk"):
            build_retrieval_examples(dataset, [], source_filenames={})

    def test_invalid_top_k_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k"):
            evaluate_retriever(MappingRetriever({}), (), top_k=0)


if __name__ == "__main__":
    unittest.main()
