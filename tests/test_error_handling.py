import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import evaluate_rag
import ingestion as ingest_script
import retrival as retrieval_script


class IngestionErrorHandlingTests(unittest.TestCase):
    def test_load_ingestion_log_warns_on_unexpected_json_shape(self):
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "ingestion_log.json"
            log_path.write_text('["not", "a", "dict"]', encoding="utf-8")

            with self.assertLogs(ingest_script.logger, level="WARNING") as logs:
                history = ingest_script.load_ingestion_log(log_path)

            self.assertEqual(history, {})
            self.assertIn("starting fresh", logs.output[0])

    def test_load_ingestion_log_warns_on_unreadable_file(self):
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "ingestion_log.json"
            log_path.write_text("{}", encoding="utf-8")

            with (
                patch.object(Path, "open", side_effect=OSError("denied")),
                self.assertLogs(ingest_script.logger, level="WARNING") as logs,
            ):
                history = ingest_script.load_ingestion_log(log_path)

            self.assertEqual(history, {})
            self.assertIn("denied", logs.output[0])

    def test_save_ingestion_log_reports_write_failure(self):
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "ingestion_log.json"

            with (
                patch.object(Path, "open", side_effect=OSError("disk full")),
                self.assertRaises(OSError) as ctx,
            ):
                ingest_script.save_ingestion_log(log_path, "sample.pdf", "now")

            self.assertIn("Failed to write ingestion log", str(ctx.exception))

    def test_ingest_documents_raises_when_every_chunk_fails(self):
        openai_client = MagicMock()
        openai_client.embeddings.create.side_effect = RuntimeError("embedding down")

        with (
            patch.object(ingest_script, "validate_credentials"),
            patch.object(ingest_script, "build_connection_string", return_value="uri"),
            patch.object(ingest_script, "MongoClient") as mongo_client_cls,
            patch.object(ingest_script, "OpenAI", return_value=openai_client),
            self.assertRaises(RuntimeError) as ctx,
        ):
            ingest_script.ingest_documents("some text " * 100)

        self.assertIn("failed to ingest", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        mongo_client_cls.return_value.close.assert_called_once()

    def test_ingest_documents_reports_partial_failures(self):
        openai_client = MagicMock()
        embedding_response = MagicMock()
        embedding_response.data = [MagicMock(embedding=[0.1, 0.2])]
        openai_client.embeddings.create.side_effect = [
            embedding_response,
            RuntimeError("embedding down"),
            embedding_response,
        ]

        with (
            patch.object(ingest_script, "validate_credentials"),
            patch.object(ingest_script, "build_connection_string", return_value="uri"),
            patch.object(ingest_script, "MongoClient"),
            patch.object(ingest_script, "OpenAI", return_value=openai_client),
            patch.object(ingest_script, "RecursiveCharacterTextSplitter") as splitter,
        ):
            splitter.return_value.split_text.return_value = ["a", "b", "c"]
            with self.assertLogs(ingest_script.logger, level="ERROR") as logs:
                inserted = ingest_script.ingest_documents("text")

        self.assertEqual(inserted, 2)
        self.assertTrue(any("1/3 chunks failed" in line for line in logs.output))

    def test_ingest_documents_rejects_empty_chunk_list(self):
        with (
            patch.object(ingest_script, "validate_credentials"),
            patch.object(ingest_script, "build_connection_string", return_value="uri"),
            patch.object(ingest_script, "MongoClient"),
            patch.object(ingest_script, "OpenAI"),
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

    def test_env_int_rejects_non_numeric_values(self):
        with (
            patch.dict("os.environ", {"NUM_CANDIDATES": "many"}),
            self.assertRaises(ValueError) as ctx,
        ):
            retrieval_script._env_int("NUM_CANDIDATES", "10")

        self.assertIn("must be an integer", str(ctx.exception))

    def test_operation_failure_is_wrapped_with_cause(self):
        openai_client = MagicMock()
        embedding_response = MagicMock()
        embedding_response.data = [MagicMock(embedding=[0.1])]
        openai_client.embeddings.create.return_value = embedding_response
        failure = retrieval_script.OperationFailure("index missing")
        collection = MagicMock()
        collection.aggregate.side_effect = failure

        with (
            patch.object(
                retrieval_script, "get_openai_client", return_value=openai_client
            ),
            patch.object(
                retrieval_script, "get_mongo_collection", return_value=collection
            ),
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
        embedding_response = MagicMock()
        embedding_response.data = [MagicMock(embedding=[0.1])]
        openai_client.embeddings.create.return_value = embedding_response
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content=None))]
        openai_client.chat.completions.create.return_value = completion
        collection = MagicMock()
        collection.aggregate.return_value = [{"text": "context"}]

        with (
            patch.object(
                retrieval_script, "get_openai_client", return_value=openai_client
            ),
            patch.object(
                retrieval_script, "get_mongo_collection", return_value=collection
            ),
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

        self.assertEqual(results[0]["error"], None)
        self.assertEqual(results[1]["answer"], None)
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

    def test_write_failure_is_reported_with_path(self):
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"

            with (
                patch.object(Path, "write_text", side_effect=OSError("read-only")),
                self.assertRaises(OSError) as ctx,
            ):
                evaluate_rag.evaluate_questions(
                    ["a"],
                    output_path=output_path,
                    interactive=False,
                    answer_fn=lambda question: "answer",
                )

        self.assertIn("Failed to write evaluation results", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
