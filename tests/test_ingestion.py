import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError

import ingestion


def make_openai_client(embedding=None):
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=embedding or [0.1, 0.2])]
    )
    return client


class ValidateCredentialsTests(unittest.TestCase):
    def test_passes_when_all_credentials_present(self):
        with (
            patch.object(ingestion, "MONGODB_USERNAME", "user"),
            patch.object(ingestion, "MONGODB_PASSWORD", "pass"),
            patch.object(ingestion, "OPENAI_API_KEY", "key"),
        ):
            self.assertIsNone(ingestion.validate_credentials())

    def test_reports_missing_variables(self):
        with (
            patch.object(ingestion, "MONGODB_USERNAME", None),
            patch.object(ingestion, "MONGODB_PASSWORD", "pass"),
            patch.object(ingestion, "OPENAI_API_KEY", None),
            self.assertRaises(ValueError) as ctx,
        ):
            ingestion.validate_credentials()

        message = str(ctx.exception)
        self.assertIn("MONGODB_USERNAME", message)
        self.assertIn("OPENAI_API_KEY", message)
        self.assertNotIn("MONGODB_PASSWORD", message)


class BuildConnectionStringTests(unittest.TestCase):
    def test_url_encodes_credentials(self):
        with (
            patch.object(ingestion, "MONGODB_USERNAME", "user name"),
            patch.object(ingestion, "MONGODB_PASSWORD", "p@ss:word"),
            patch.object(ingestion, "MONGODB_CLUSTER", "cluster.example.net"),
        ):
            uri = ingestion.build_connection_string()

        self.assertTrue(uri.startswith("mongodb+srv://user+name:p%40ss%3Aword@"))
        self.assertIn("@cluster.example.net/", uri)


class GetPdfFilesTests(unittest.TestCase):
    def test_returns_pdfs_sorted_and_case_insensitive(self):
        with TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            (temp_dir / "b.pdf").write_bytes(b"pdf")
            (temp_dir / "a.PDF").write_bytes(b"pdf")
            (temp_dir / "notes.txt").write_text("no", encoding="utf-8")
            (temp_dir / "nested").mkdir()

            names = [path.name for path in ingestion.get_pdf_files(temp_dir)]

        self.assertEqual(names, ["a.PDF", "b.pdf"])

    def test_raises_when_directory_missing(self):
        with (
            TemporaryDirectory() as tmp_dir,
            self.assertRaises(FileNotFoundError),
        ):
            ingestion.get_pdf_files(Path(tmp_dir) / "missing")

    def test_raises_when_path_is_a_file(self):
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "doc.pdf"
            file_path.write_bytes(b"pdf")

            with self.assertRaises(FileNotFoundError):
                ingestion.get_pdf_files(file_path)


class IngestionLogTests(unittest.TestCase):
    def test_load_returns_empty_dict_when_file_missing(self):
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "missing.json"

            self.assertEqual(ingestion.load_ingestion_log(log_path), {})

    def test_load_returns_empty_dict_for_invalid_json(self):
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "log.json"
            log_path.write_text("{not json", encoding="utf-8")

            self.assertEqual(ingestion.load_ingestion_log(log_path), {})

    def test_load_returns_empty_dict_for_non_mapping_payload(self):
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "log.json"
            log_path.write_text("[1, 2, 3]", encoding="utf-8")

            self.assertEqual(ingestion.load_ingestion_log(log_path), {})

    def test_save_appends_entries_and_creates_parent_directory(self):
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "nested" / "log.json"

            ingestion.save_ingestion_log(log_path, "first.pdf", "2026-01-01 00:00:00")
            history = ingestion.save_ingestion_log(
                log_path, "second.pdf", "2026-01-02 00:00:00"
            )

            self.assertEqual(
                history,
                {
                    "first.pdf": "2026-01-01 00:00:00",
                    "second.pdf": "2026-01-02 00:00:00",
                },
            )
            self.assertEqual(
                json.loads(log_path.read_text(encoding="utf-8")),
                history,
            )

    def test_save_defaults_timestamp_to_now(self):
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "log.json"

            history = ingestion.save_ingestion_log(log_path, "first.pdf")

        self.assertRegex(
            history["first.pdf"],
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
        )


class ConfirmIngestionTests(unittest.TestCase):
    def test_prompt_mentions_previous_ingestion(self):
        with patch("builtins.input", return_value="yes") as mock_input:
            self.assertTrue(
                ingestion.confirm_ingestion("doc.pdf", "2026-01-01 00:00:00")
            )

        self.assertIn("2026-01-01 00:00:00", mock_input.call_args.args[0])

    def test_prompt_for_first_time_ingestion(self):
        with patch("builtins.input", return_value="nope") as mock_input:
            self.assertFalse(ingestion.confirm_ingestion("doc.pdf"))

        self.assertIn("doc.pdf", mock_input.call_args.args[0])


class SelectSinglePdfTests(unittest.TestCase):
    def _make_source_dir(self, tmp_dir, names=("a.pdf", "b.pdf")):
        temp_dir = Path(tmp_dir)
        for name in names:
            (temp_dir / name).write_bytes(b"pdf")
        return temp_dir

    def test_returns_selected_file(self):
        with TemporaryDirectory() as tmp_dir:
            source_dir = self._make_source_dir(tmp_dir)

            with patch("builtins.input", return_value="2"):
                selected = ingestion.select_single_pdf(source_dir)

        self.assertEqual(selected.name, "b.pdf")

    def test_reprompts_on_invalid_and_out_of_range_input(self):
        with TemporaryDirectory() as tmp_dir:
            source_dir = self._make_source_dir(tmp_dir)

            with patch("builtins.input", side_effect=["abc", "9", "0", "1"]):
                selected = ingestion.select_single_pdf(source_dir)

        self.assertEqual(selected.name, "a.pdf")

    def test_raises_when_no_pdfs_available(self):
        with TemporaryDirectory() as tmp_dir, self.assertRaises(FileNotFoundError):
            ingestion.select_single_pdf(tmp_dir)


class ExtractTextFromPdfTests(unittest.TestCase):
    def test_joins_page_text(self):
        reader = SimpleNamespace(
            pages=[
                SimpleNamespace(extract_text=lambda: "page one"),
                SimpleNamespace(extract_text=lambda: None),
                SimpleNamespace(extract_text=lambda: "page two"),
            ]
        )

        with patch("pypdf.PdfReader", return_value=reader) as mock_reader:
            text = ingestion.extract_text_from_pdf(Path("sourceFile/doc.pdf"))

        self.assertEqual(text, "page one\n\npage two")
        mock_reader.assert_called_once_with("sourceFile/doc.pdf")

    def test_raises_when_pdf_has_no_text(self):
        reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: "  ")])

        with (
            patch("pypdf.PdfReader", return_value=reader),
            self.assertRaises(ValueError),
        ):
            ingestion.extract_text_from_pdf(Path("sourceFile/doc.pdf"))


class IngestDocumentsTests(unittest.TestCase):
    def _enter_patches(self, stack, mongo_client, openai_client, chunks=None):
        stack.enter_context(patch.object(ingestion, "validate_credentials"))
        stack.enter_context(
            patch.object(
                ingestion, "build_connection_string", return_value="mongodb://uri"
            )
        )
        stack.enter_context(
            patch.object(ingestion, "MongoClient", return_value=mongo_client)
        )
        stack.enter_context(
            patch.object(ingestion, "OpenAI", return_value=openai_client)
        )
        if chunks is not None:
            splitter = MagicMock()
            splitter.split_text.return_value = list(chunks)
            stack.enter_context(
                patch.object(
                    ingestion, "RecursiveCharacterTextSplitter", return_value=splitter
                )
            )

    def test_embeds_and_inserts_every_chunk(self):
        mongo_client = MagicMock()
        collection = mongo_client[ingestion.DATABASE_NAME][ingestion.COLLECTION_NAME]
        openai_client = make_openai_client(embedding=[0.3, 0.4])
        raw_text = "sentence one. " * 40

        with ExitStack() as stack:
            self._enter_patches(stack, mongo_client, openai_client)
            inserted = ingestion.ingest_documents(raw_text)

        self.assertGreater(inserted, 1)
        self.assertEqual(inserted, collection.insert_one.call_count)
        self.assertEqual(inserted, openai_client.embeddings.create.call_count)
        first_document = collection.insert_one.call_args_list[0].args[0]
        self.assertEqual(first_document["chunk_id"], 0)
        self.assertEqual(first_document["text_embedding"], [0.3, 0.4])
        self.assertIn("text", first_document)
        mongo_client.admin.command.assert_called_once_with("ping")
        mongo_client.close.assert_called_once_with()

    def test_skips_duplicate_and_failed_chunks(self):
        mongo_client = MagicMock()
        collection = mongo_client[ingestion.DATABASE_NAME][ingestion.COLLECTION_NAME]
        collection.insert_one.side_effect = [
            MagicMock(),
            DuplicateKeyError("duplicate"),
            RuntimeError("write failed"),
        ]
        openai_client = make_openai_client()

        with ExitStack() as stack:
            self._enter_patches(
                stack, mongo_client, openai_client, chunks=["one", "two", "three"]
            )
            inserted = ingestion.ingest_documents("ignored")

        self.assertEqual(inserted, 1)

    def test_tolerates_index_creation_failure(self):
        mongo_client = MagicMock()
        collection = mongo_client[ingestion.DATABASE_NAME][ingestion.COLLECTION_NAME]
        collection.create_index.side_effect = RuntimeError("no permission")
        openai_client = make_openai_client()

        with ExitStack() as stack:
            self._enter_patches(
                stack, mongo_client, openai_client, chunks=["only chunk"]
            )
            inserted = ingestion.ingest_documents("ignored")

        self.assertEqual(inserted, 1)

    def test_propagates_connection_timeout(self):
        mongo_client = MagicMock()
        mongo_client.admin.command.side_effect = ServerSelectionTimeoutError("timeout")

        with ExitStack() as stack:
            self._enter_patches(stack, mongo_client, make_openai_client())
            with self.assertRaises(ServerSelectionTimeoutError):
                ingestion.ingest_documents("text")

        mongo_client.close.assert_called_once_with()

    def test_propagates_unexpected_errors(self):
        mongo_client = MagicMock()
        mongo_client.admin.command.side_effect = RuntimeError("boom")

        with ExitStack() as stack:
            self._enter_patches(stack, mongo_client, make_openai_client())
            with self.assertRaises(RuntimeError):
                ingestion.ingest_documents("text")

    def test_requires_credentials(self):
        with (
            patch.object(
                ingestion,
                "validate_credentials",
                side_effect=ValueError("missing"),
            ),
            patch.object(ingestion, "MongoClient") as mock_mongo,
            self.assertRaises(ValueError),
        ):
            ingestion.ingest_documents("text")

        mock_mongo.assert_not_called()


class MainTests(unittest.TestCase):
    def _enter_patches(self, stack, confirmed=True, inserted=3):
        stack.enter_context(
            patch.object(
                ingestion, "select_single_pdf", return_value=Path("sourceFile/doc.pdf")
            )
        )
        stack.enter_context(
            patch.object(
                ingestion,
                "load_ingestion_log",
                return_value={"doc.pdf": "2026-01-01 00:00:00"},
            )
        )
        return SimpleNamespace(
            confirm=stack.enter_context(
                patch.object(ingestion, "confirm_ingestion", return_value=confirmed)
            ),
            extract=stack.enter_context(
                patch.object(
                    ingestion, "extract_text_from_pdf", return_value="raw text"
                )
            ),
            ingest=stack.enter_context(
                patch.object(ingestion, "ingest_documents", return_value=inserted)
            ),
            save=stack.enter_context(patch.object(ingestion, "save_ingestion_log")),
        )

    def test_returns_inserted_count_and_logs_ingestion(self):
        with ExitStack() as stack:
            mocks = self._enter_patches(stack)
            result = ingestion.main()

            mocks.confirm.assert_called_once_with("doc.pdf", "2026-01-01 00:00:00")
            mocks.ingest.assert_called_once_with("raw text")
            self.assertEqual(mocks.save.call_args.args[1], "doc.pdf")

        self.assertEqual(result, 3)

    def test_returns_zero_when_user_declines(self):
        with ExitStack() as stack:
            mocks = self._enter_patches(stack, confirmed=False)
            result = ingestion.main()

            mocks.extract.assert_not_called()
            mocks.ingest.assert_not_called()
            mocks.save.assert_not_called()

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
