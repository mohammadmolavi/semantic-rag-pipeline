import json
import re
import struct
import unittest

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SCREENSHOTS = (
    "admin-dashboard.png",
    "admin-documents.png",
    "admin-document-detail.png",
    "admin-question-answer.png",
)


class DocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        cls.api_docs = (PROJECT_ROOT / "docs" / "API.md").read_text(encoding="utf-8")
        cls.schema = yaml.safe_load(
            (PROJECT_ROOT / "docs" / "openapi.yaml").read_text(encoding="utf-8")
        )

    def test_readme_includes_setup_architecture_samples_api_and_screenshots(self) -> None:
        sections = (
            "## Retrieval architecture",
            "## Quick start with Docker",
            "## Local development",
            "## Configuration",
            "## Bundled sample documents",
            "## Django Admin workflow",
            "## REST API",
            "## Test suite",
            "## Troubleshooting",
        )

        for section in sections:
            with self.subTest(section=section):
                self.assertIn(section, self.readme)

    def test_openapi_documents_every_implemented_api_operation(self) -> None:
        expected = {
            "/api/documents/": {"get", "post"},
            "/api/documents/{id}/": {"get", "put", "patch", "delete"},
            "/api/ask/": {"post"},
            "/api/history/": {"get"},
            "/api/history/{id}/": {"get"},
        }

        for path, operations in expected.items():
            with self.subTest(path=path):
                self.assertTrue(operations.issubset(self.schema["paths"][path]))

    def test_openapi_request_constraints_match_the_ask_serializer(self) -> None:
        request = self.schema["components"]["schemas"]["AskRequest"]
        top_k = request["properties"]["top_k"]

        self.assertEqual(request["required"], ["question"])
        self.assertEqual(top_k["minimum"], 1)
        self.assertEqual(top_k["maximum"], 10)
        self.assertEqual(top_k["default"], 4)

    def test_all_local_openapi_references_resolve(self) -> None:
        def inspect(value: object) -> None:
            if isinstance(value, list):
                for item in value:
                    inspect(item)

            elif isinstance(value, dict):
                reference = value.get("$ref")

                if isinstance(reference, str):
                    self.assertTrue(reference.startswith("#/"), reference)
                    target = self.schema

                    for part in reference.removeprefix("#/").split("/"):
                        self.assertIn(part, target, reference)
                        target = target[part]

                for item in value.values():
                    inspect(item)

        inspect(self.schema)

    def test_persian_examples_are_saved_as_valid_utf8_without_mojibake(self) -> None:
        self.assertIn("مبلغ ماهانه پلن حرفه‌ای", self.readme)
        self.assertIn("زمان پاسخ اولیه پشتیبانی", self.api_docs)

        for document in (self.readme, self.api_docs):
            self.assertIsNone(re.search(r"(?:Ø.|Ù.|Û.|Ú.)", document))

    def test_every_admin_screenshot_is_linked_and_has_desktop_resolution(self) -> None:
        for name in EXPECTED_SCREENSHOTS:
            with self.subTest(name=name):
                path = PROJECT_ROOT / "docs" / "screenshots" / name
                self.assertIn(f"docs/screenshots/{name}", self.readme)
                self.assertTrue(path.is_file())

                with path.open("rb") as image:
                    header = image.read(24)

                self.assertEqual(header[:8], b"\x89PNG\r\n\x1a\n")
                width, height = struct.unpack(">II", header[16:24])
                self.assertGreaterEqual(width, 1200)
                self.assertGreaterEqual(height, 800)

    def test_documented_sample_files_and_questions_exist(self) -> None:
        samples = (
            "neural_radiance_fields.docx",
            "owncloud_user_guide_fa.docx",
            "support_and_pricing_fa.docx",
        )

        for name in samples:
            with self.subTest(name=name):
                self.assertTrue((PROJECT_ROOT / "sample_data" / name).is_file())
                self.assertIn(name, self.readme)

        questions = json.loads(
            (PROJECT_ROOT / "sample_data" / "sample_questions.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertGreaterEqual(len(questions), 8)

    def test_documentation_discloses_public_demo_api_permissions(self) -> None:
        self.assertIn("unauthenticated", self.readme)
        self.assertIn("AllowAny", self.api_docs)

    def test_example_configuration_covers_documented_runtime_controls(self) -> None:
        environment = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

        for key in (
            "OPENROUTER_API_KEY",
            "DATABASE_URL",
            "DJANGO_SECRET_KEY",
            "DJANGO_DEBUG",
            "INDEX_DOCUMENTS",
            "RERANKER_ENABLED",
            "RERANKER_MODEL",
            "RERANKER_BATCH_SIZE",
        ):
            with self.subTest(key=key):
                self.assertRegex(environment, rf"(?m)^{key}=")
                self.assertIn(f"`{key}`", self.readme)

