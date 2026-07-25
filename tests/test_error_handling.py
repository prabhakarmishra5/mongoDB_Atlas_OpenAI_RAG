import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import evaluate_rag
import ingestion as ingest_script
import rag_common
import retrival as retrieval_script


class RagCommonErrorHandlingTests(unittest.TestCase):
    def test_env_int_rejects_non_numeric_values(self):
        with (
            patch.dict("os.environ", {"NUM_CANDIDATES": "many"}),
            self.assertRaises(ValueError) as ctx,
        ):
            rag_common.env_int("NUM_CANDIDATES", 10)

        self.assertIn("must be an integer", str(ctx.exception))

    def test_read_json_dict_warns_on_unexpected_json_shape(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "log.json"
            path.write_text('["not", "a", "dict"]', encoding="utf-8")

            with self.assertLogs(rag_common.logger, level="WARNING") as logs:
                self.assertEqual(rag_common.read_json_dict(path, "bad log"), {})

            self.assertIn("instead of an object", logs.output[0])

    def test_read_json_dict_warns_on_unreadable_file(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "log.json"
            path.write_text("{}", encoding="utf-8")

            with (
                patch.object(Path, "open", side_effect=OSError("denied")),
                self.assertLogs(rag_common.logger, level="WARNING") as logs,
            ):
                self.assertEqual(rag_common.read_json_dict(path, "bad log"), {})

            self.assertIn("denied", logs.output[0])

    def test_write_json_reports_write_failure_with_path(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "log.json"

            with (
                patch.object(Path, "write_text", side_effect=OSError("disk full")),
                self.assertRaises(OSError) as ctx,
            ):
                rag_common.write_json(path, {"a": 1})

        self.assertIn("Failed to write JSON", str(ctx.exception))


@contextmanager
def patched_ingestion_clients(mongo_client=None, collection=None):
    """Stub out credential validation and the Atlas/OpenAI clients."""
    mongo_client = mongo_client or MagicMock()
    collection = collection or MagicMock()
    with (
        patch.object(ingest_script, "validate_credentials"),
        patch.object(
            ingest_script,
            "connect_to_collection",
            return_value=(mongo_client, collection),
        ),
        patch.object(ingest_script, "create_openai_client", return_value=MagicMock()),
    ):
        yield mongo_client, collection


@contextmanager
def patched_retrieval_clients(openai_client=None, collection=None):
    """Stub out the cached retrieval clients and the embedding call."""
    openai_client = openai_client or MagicMock()
    collection = collection or MagicMock()
    with (
        patch.object(retrieval_script, "get_openai_client", return_value=openai_client),
        patch.object(
            retrieval_script, "get_mongo_collection", return_value=collection
        ),
        patch.object(retrieval_script, "embed_text", return_value=[0.1]),
    ):
        yield openai_client, collection


class IngestionErrorHandlingTests(unittest.TestCase):
    def test_ingest_documents_raises_when_every_chunk_fails(self):
        with (
            patched_ingestion_clients() as (mongo_client, _),
            patch.object(
                ingest_script, "embed_text", side_effect=RuntimeError("embed boom")
            ),
            self.assertRaises(RuntimeError) as ctx,
        ):
            ingest_script.ingest_documents("some text " * 100)

        self.assertIn("No chunks were ingested", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        mongo_client.close.assert_called_once()

    def test_ingest_documents_reports_partial_failures(self):
        with (
            patched_ingestion_clients(),
            patch.object(
                ingest_script,
                "embed_text",
                side_effect=[[0.1], RuntimeError("embed boom"), [0.2]],
            ),
            patch.object(ingest_script, "RecursiveCharacterTextSplitter") as splitter,
        ):
            splitter.return_value.split_text.return_value = ["a", "b", "c"]
            with self.assertLogs(ingest_script.logger, level="ERROR") as logs:
                inserted = ingest_script.ingest_documents("text")

        self.assertEqual(inserted, 2)
        self.assertTrue(any("1/3 chunks failed" in line for line in logs.output))

    def test_ingest_documents_warns_when_every_chunk_is_a_duplicate(self):
        collection = MagicMock()
        collection.insert_one.side_effect = ingest_script.DuplicateKeyError("dupe")

        with (
            patched_ingestion_clients(collection=collection),
            patch.object(ingest_script, "embed_text", return_value=[0.1]),
            patch.object(ingest_script, "RecursiveCharacterTextSplitter") as splitter,
        ):
            splitter.return_value.split_text.return_value = ["a", "b"]
            with self.assertLogs(ingest_script.logger, level="WARNING") as logs:
                inserted = ingest_script.ingest_documents("text")

        self.assertEqual(inserted, 0)
        self.assertTrue(any("No new chunks ingested" in line for line in logs.output))

    def test_ingest_documents_rejects_empty_chunk_list(self):
        with (
            patched_ingestion_clients(),
            patch.object(ingest_script, "RecursiveCharacterTextSplitter") as splitter,
        ):
            splitter.return_value.split_text.return_value = []
            with self.assertRaises(ValueError):
                ingest_script.ingest_documents("   ")


class RetrievalErrorHandlingTests(unittest.TestCase):
    def tearDown(self):
        retrieval_script.openai_client = None
        retrieval_script.mongo_client = None
        retrieval_script.collection = None

    def test_malformed_search_setting_reports_variable_name(self):
        with (
            patch.dict("os.environ", {"RESULT_LIMIT": "two"}),
            self.assertRaises(ValueError) as ctx,
        ):
            retrieval_script.get_brain_response("question")

        self.assertIn("RESULT_LIMIT must be an integer", str(ctx.exception))

    def test_operation_failure_is_wrapped_with_cause(self):
        failure = retrieval_script.OperationFailure("index missing")
        collection = MagicMock()
        collection.aggregate.side_effect = failure

        with (
            patched_retrieval_clients(collection=collection),
            self.assertRaises(retrieval_script.RetrievalError) as ctx,
        ):
            retrieval_script.get_brain_response("question")

        self.assertIs(ctx.exception.__cause__, failure)

    def test_server_selection_timeout_becomes_connection_error(self):
        timeout = retrieval_script.ServerSelectionTimeoutError("no servers")

        with (
            patch.object(retrieval_script, "get_openai_client", side_effect=timeout),
            self.assertRaises(ConnectionError) as ctx,
        ):
            retrieval_script.get_brain_response("question")

        self.assertIs(ctx.exception.__cause__, timeout)

    def test_empty_completion_content_raises(self):
        openai_client = MagicMock()
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content=None))]
        openai_client.chat.completions.create.return_value = completion
        collection = MagicMock()
        collection.aggregate.return_value = [{"text": "context"}]

        with (
            patched_retrieval_clients(openai_client, collection),
            self.assertRaises(retrieval_script.RetrievalError),
        ):
            retrieval_script.get_brain_response("question")

    def test_close_clients_logs_but_does_not_raise(self):
        client = MagicMock()
        client.close.side_effect = retrieval_script.PyMongoError("already closed")
        retrieval_script.mongo_client = client

        with self.assertLogs(retrieval_script.logger, level="WARNING"):
            retrieval_script.close_clients()

        self.assertIsNone(retrieval_script.mongo_client)

    def test_main_returns_failure_exit_code_on_retrieval_error(self):
        with patch.object(
            retrieval_script,
            "get_brain_response",
            side_effect=retrieval_script.RetrievalError("boom"),
        ):
            exit_code = retrieval_script.main(["--question", "hi", "--no-confirm"])

        self.assertEqual(exit_code, 1)


class EvaluationErrorHandlingTests(unittest.TestCase):
    def test_failed_question_is_recorded_and_run_continues(self):
        answers = {"good": "answer"}

        def answer_fn(question):
            if question not in answers:
                raise RuntimeError("model unavailable")
            return answers[question]

        results = evaluate_rag.evaluate_questions(
            ["good", "bad"],
            interactive=False,
            answer_fn=answer_fn,
        )

        self.assertIsNone(results[0]["error"])
        self.assertIsNone(results[1]["answer"])
        self.assertIn("model unavailable", results[1]["error"])

    def test_all_questions_failing_raises(self):
        def answer_fn(question):
            raise RuntimeError("model unavailable")

        with self.assertRaises(RuntimeError) as ctx:
            evaluate_rag.evaluate_questions(
                ["a", "b"],
                interactive=False,
                answer_fn=answer_fn,
            )

        self.assertIn("All 2 questions failed", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

    def test_main_returns_non_zero_when_any_question_errors(self):
        with patch.object(
            evaluate_rag,
            "evaluate_questions",
            return_value=[
                {
                    "question": "q",
                    "answer": None,
                    "score": None,
                    "notes": "",
                    "error": "RuntimeError: boom",
                }
            ],
        ):
            self.assertEqual(evaluate_rag.main(), 1)


if __name__ == "__main__":
    unittest.main()
