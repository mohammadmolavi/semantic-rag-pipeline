import os
import io
import tempfile
import unittest

from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.test_settings")

import django

django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.db import connection
from django.test import override_settings
from rest_framework.test import APIRequestFactory

from documents.models import Document, Question
from documents.serializers import AskSerializer
from documents.services import generate_answer
from documents.views import AskView, DocumentViewSet
from rag.langchain_rag import LangChainRagAnswer
from rag.services import reindex_source


class DocumentLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.media_directory = tempfile.TemporaryDirectory()
        cls.settings_override = override_settings(
            MEDIA_ROOT=cls.media_directory.name,
            INDEX_DOCUMENTS=False,
        )
        cls.settings_override.enable()

        with connection.schema_editor() as schema_editor:
            schema_editor.create_model(Document)
            schema_editor.create_model(Question)

    @classmethod
    def tearDownClass(cls) -> None:
        with connection.schema_editor() as schema_editor:
            schema_editor.delete_model(Question)
            schema_editor.delete_model(Document)

        cls.settings_override.disable()
        cls.media_directory.cleanup()

    def tearDown(self) -> None:
        Question.objects.all().delete()
        Document.objects.all().delete()

    def create_document(self, name: str = "first.txt", content: bytes = b"alpha"):
        document = Document(
            title="Sample",
            file=SimpleUploadedFile(name, content, content_type="text/plain"),
        )
        document.save()
        return document

    def test_create_extracts_text_from_saved_file(self) -> None:
        document = self.create_document(content="متن نمونه".encode())

        self.assertEqual(document.text, "متن نمونه")
        document.refresh_from_db()
        self.assertEqual(document.text, "متن نمونه")

    def test_empty_document_is_rejected_without_saving_file_or_row(self) -> None:
        document = Document(
            title="Empty",
            file=SimpleUploadedFile(
                "empty.txt",
                b" \n\t",
                content_type="text/plain",
            ),
        )

        with self.assertRaisesRegex(ValueError, "no extractable text"):
            document.save()

        self.assertEqual(Document.objects.count(), 0)
        self.assertFalse(document.file.storage.exists("documents/empty.txt"))

    def test_successful_index_updates_instance_and_database_chunk_count(self) -> None:
        with (
            override_settings(INDEX_DOCUMENTS=True),
            patch("rag.services.reindex_source", return_value=3),
        ):
            document = self.create_document()

        self.assertEqual(document.chunk_count, 3)
        self.assertEqual(Document.objects.get(pk=document.pk).chunk_count, 3)

    def test_title_only_update_does_not_reextract_or_reindex_file(self) -> None:
        document = self.create_document()

        with (
            patch("documents.models.load_document") as load_document,
            patch("documents.models._index_document") as index_document,
        ):
            document.title = "Renamed"
            document.save(update_fields=["title"])

        load_document.assert_not_called()
        index_document.assert_not_called()

    def test_replacing_file_removes_previous_upload_after_success(self) -> None:
        document = self.create_document()
        storage = document.file.storage
        old_name = document.file.name

        document.file = SimpleUploadedFile(
            "replacement.txt",
            b"beta",
            content_type="text/plain",
        )
        document.save()

        self.assertFalse(storage.exists(old_name))
        self.assertTrue(storage.exists(document.file.name))
        self.assertEqual(document.text, "beta")

    def test_failed_replacement_restores_database_and_removes_new_upload(self) -> None:
        document = self.create_document()
        storage = document.file.storage
        old_name = document.file.name
        document.file = SimpleUploadedFile(
            "broken.txt",
            b"broken",
            content_type="text/plain",
        )

        with (
            patch("documents.models.load_document", side_effect=ValueError("invalid")),
            self.assertRaisesRegex(ValueError, "invalid"),
        ):
            document.save()

        document.refresh_from_db()
        self.assertEqual(document.file.name, old_name)
        self.assertTrue(storage.exists(old_name))
        self.assertFalse(storage.exists("documents/broken.txt"))

    def test_bulk_delete_removes_uploaded_files(self) -> None:
        document = self.create_document()
        storage = document.file.storage
        file_name = document.file.name

        Document.objects.filter(pk=document.pk).delete()

        self.assertFalse(storage.exists(file_name))


class QuestionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.media_directory = tempfile.TemporaryDirectory()
        cls.settings_override = override_settings(
            MEDIA_ROOT=cls.media_directory.name,
            INDEX_DOCUMENTS=False,
        )
        cls.settings_override.enable()

        with connection.schema_editor() as schema_editor:
            schema_editor.create_model(Document)
            schema_editor.create_model(Question)

    @classmethod
    def tearDownClass(cls) -> None:
        with connection.schema_editor() as schema_editor:
            schema_editor.delete_model(Question)
            schema_editor.delete_model(Document)

        cls.settings_override.disable()
        cls.media_directory.cleanup()

    def tearDown(self) -> None:
        Question.objects.all().delete()
        Document.objects.all().delete()

    def test_document_id_zero_is_rejected_instead_of_becoming_global_search(self) -> None:
        serializer = AskSerializer(
            data={"question": "test", "document_id": 0}
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("document_id", serializer.errors)

    def test_corrupt_docx_returns_400_without_saving_document(self) -> None:
        request = APIRequestFactory().post(
            "/api/documents/",
            {
                "title": "Broken",
                "file": SimpleUploadedFile(
                    "broken.docx",
                    b"not a docx archive",
                    content_type=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                ),
            },
            format="multipart",
        )

        response = DocumentViewSet.as_view({"post": "create"})(request)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(Document.objects.count(), 0)

    def test_generation_failure_returns_503_without_orphan_history(self) -> None:
        request = APIRequestFactory().post(
            "/api/ask/",
            {"question": "test"},
            format="json",
        )

        with patch(
            "documents.views.generate_answer",
            side_effect=RuntimeError("provider unavailable"),
        ):
            response = AskView.as_view()(request)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(Question.objects.count(), 0)

    def test_generate_answer_saves_new_question_only_after_success(self) -> None:
        question = Question(question="test")
        result = LangChainRagAnswer(answer="پاسخ", sources=[])

        with patch("rag.services.ask_question", return_value=result):
            generate_answer(question)

        self.assertIsNotNone(question.pk)
        self.assertEqual(question.answer, "پاسخ")
        self.assertEqual(Question.objects.count(), 1)


class ReindexSafetyTests(unittest.TestCase):
    def test_new_chunks_are_upserted_before_stale_chunks_are_deleted(self) -> None:
        events = []

        class RecordingVectorStore:
            def add_documents(self, documents, ids):
                events.append(("add", tuple(ids)))

            def delete(self, ids):
                events.append(("delete", tuple(ids)))

        with patch("rag.services.get_vector_store", return_value=RecordingVectorStore()):
            chunk_count = reindex_source(
                "document:9",
                "short replacement text",
                previous_chunk_count=3,
            )

        self.assertEqual(chunk_count, 1)
        self.assertEqual(events[0], ("add", ("document:9:0",)))
        self.assertEqual(
            events[1],
            ("delete", ("document:9:1", "document:9:2")),
        )

    def test_failed_upsert_does_not_delete_previous_chunks(self) -> None:
        class FailingVectorStore:
            def __init__(self) -> None:
                self.deleted = []

            def add_documents(self, documents, ids):
                raise RuntimeError("embedding failed")

            def delete(self, ids):
                self.deleted.extend(ids)

        vector_store = FailingVectorStore()

        with (
            patch("rag.services.get_vector_store", return_value=vector_store),
            self.assertRaisesRegex(RuntimeError, "embedding failed"),
        ):
            reindex_source(
                "document:9",
                "replacement",
                previous_chunk_count=3,
            )

        self.assertEqual(vector_store.deleted, [])


class RetrievalEvaluationCommandTests(unittest.TestCase):
    def test_bm25_command_runs_end_to_end_without_database_or_llm(self) -> None:
        output = io.StringIO()

        call_command(
            "evaluate_retrieval",
            mode="bm25",
            top_k=4,
            min_hit_rate=0.9,
            stdout=output,
        )

        report = output.getvalue()
        self.assertIn("Hit Rate@4: 1.000", report)
        self.assertIn("MRR@4: 1.000", report)
        self.assertIn("Negative rejection rate: 1.000", report)


if __name__ == "__main__":
    unittest.main()
