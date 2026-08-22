"""Evaluate BM25 or the live hybrid retriever without calling the LLM."""

from __future__ import annotations

import json

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from documents.models import Document as StoredDocument
from rag.evaluation import build_retrieval_examples, evaluate_retriever
from rag.hybrid import HybridRetriever
from rag.langchain_rag import build_retriever, split_document_text
from rag.loaders import load_document
from rag.services import get_lexical_documents, get_reranker, get_vector_store


class Command(BaseCommand):
    help = "Measure Hit Rate, MRR, Recall, and Precision for document retrieval."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--dataset",
            type=Path,
            default=None,
            help="JSON dataset path; defaults to sample_data/sample_questions.json.",
        )
        parser.add_argument("--top-k", type=int, default=4)
        parser.add_argument(
            "--mode",
            choices=("hybrid", "bm25"),
            default="hybrid",
            help="Use the live PostgreSQL index or an offline BM25 baseline.",
        )
        parser.add_argument(
            "--no-reranker",
            action="store_true",
            help="Disable Cross-Encoder reranking in hybrid mode.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            dest="as_json",
            help="Print a machine-readable JSON report.",
        )
        parser.add_argument(
            "--min-hit-rate",
            type=float,
            default=None,
            help="Exit with an error when Hit Rate@K is below this 0..1 threshold.",
        )

    def handle(self, *args, **options):
        top_k = options["top_k"]
        if top_k <= 0:
            raise CommandError("--top-k must be greater than zero.")

        minimum_hit_rate = options["min_hit_rate"]
        if minimum_hit_rate is not None and not 0 <= minimum_hit_rate <= 1:
            raise CommandError("--min-hit-rate must be between 0 and 1.")

        try:
            dataset_path = options["dataset"] or (
                settings.SAMPLE_DATA_DIR / "sample_questions.json"
            )
            dataset = self._load_dataset(dataset_path)

            if options["mode"] == "bm25":
                retriever, corpus, source_filenames = self._build_bm25(
                    dataset,
                    dataset_path,
                    top_k,
                )
            else:
                retriever, corpus, source_filenames = self._build_hybrid(
                    top_k,
                    no_reranker=options["no_reranker"],
                )

            examples = build_retrieval_examples(
                dataset,
                corpus,
                source_filenames=source_filenames,
            )
            report = evaluate_retriever(retriever, examples, top_k=top_k)
        except CommandError:
            raise
        except Exception as error:
            raise CommandError(f"Retrieval evaluation failed: {error}") from error

        if options["as_json"]:
            self.stdout.write(
                json.dumps(report.as_dict(), ensure_ascii=False, indent=2)
            )
        else:
            self._write_human_report(report, mode=options["mode"])

        if (
            minimum_hit_rate is not None
            and report.hit_rate_at_k < minimum_hit_rate
        ):
            raise CommandError(
                f"Hit Rate@{top_k}={report.hit_rate_at_k:.3f} is below "
                f"the required {minimum_hit_rate:.3f}."
            )

    def _load_dataset(self, path: Path) -> list[dict[str, object]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise CommandError(f"Evaluation dataset not found: {path}") from error
        except json.JSONDecodeError as error:
            raise CommandError(f"Invalid evaluation JSON in {path}: {error}") from error

        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise CommandError("Evaluation dataset must be a JSON array of objects.")

        if not data:
            raise CommandError("Evaluation dataset must contain at least one item.")

        return data

    def _build_bm25(self, dataset, dataset_path: Path, top_k: int):
        corpus = []
        source_filenames: dict[str, str] = {}
        filenames = {
            str(item["document"])
            for item in dataset
            if item.get("document")
        }

        for filename in sorted(filenames):
            if Path(filename).name != filename:
                raise CommandError(f"Dataset document must be a filename: {filename}")

            document_path = dataset_path.parent / filename
            if not document_path.is_file():
                raise CommandError(f"Evaluation document not found: {document_path}")

            corpus.extend(
                split_document_text(
                    load_document(document_path),
                    source=filename,
                )
            )
            source_filenames[filename] = filename

        retriever = HybridRetriever(
            None,
            lexical_documents=corpus,
            vector_k=max(top_k * 3, 12),
            lexical_k=max(top_k * 3, 12),
            final_k=top_k,
            vector_weight=0,
            lexical_weight=1,
            reranker=None,
        )
        return retriever, corpus, source_filenames

    def _build_hybrid(self, top_k: int, *, no_reranker: bool):
        stored_documents = list(
            StoredDocument.objects.filter(chunk_count__gt=0).exclude(text="")
        )
        if not stored_documents:
            raise CommandError(
                "No indexed documents were found. Ensure INDEX_DOCUMENTS=true "
                "and upload or replace the files; on a fresh setup, run "
                "`python manage.py load_sample_data`."
            )

        corpus = get_lexical_documents()
        source_filenames = {
            document.vector_source: Path(document.file.name).name
            for document in stored_documents
        }
        retriever = build_retriever(
            get_vector_store(),
            top_k=top_k,
            lexical_documents=corpus,
            reranker=None if no_reranker else get_reranker(),
        )
        return retriever, corpus, source_filenames

    def _write_human_report(self, report, *, mode: str) -> None:
        k = report.top_k
        self.stdout.write(f"Retrieval evaluation mode={mode}, K={k}")
        self.stdout.write(f"Positive queries: {len(report.positive_cases)}")
        self.stdout.write(f"Hit Rate@{k}: {report.hit_rate_at_k:.3f}")
        self.stdout.write(f"MRR@{k}: {report.mrr_at_k:.3f}")
        self.stdout.write(f"Mean Recall@{k}: {report.mean_recall_at_k:.3f}")
        self.stdout.write(f"Mean Precision@{k}: {report.mean_precision_at_k:.3f}")
        self.stdout.write(f"Negative queries: {len(report.negative_cases)}")
        self.stdout.write(
            f"Negative rejection rate: {report.negative_rejection_rate:.3f}"
        )

        for index, case in enumerate(report.cases, start=1):
            if case.expects_no_results:
                detail = f"returned={len(case.retrieved_chunks)}"
            else:
                rank = case.first_relevant_rank or "miss"
                detail = f"rank={rank}, recall={case.recall_at_k:.3f}"
            self.stdout.write(f"[{index}] {detail} | {case.question}")
