import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

import retrival


def make_embeddings_response(embedding):
    return SimpleNamespace(data=[SimpleNamespace(embedding=embedding)])


def make_chat_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def make_openai_client(embedding=None, answer="mock answer"):
    client = MagicMock()
    client.embeddings.create.return_value = make_embeddings_response(
        embedding if embedding is not None else [0.1, 0.2, 0.3]
    )
    client.chat.completions.create.return_value = make_chat_response(answer)
    return client


class ResetClientsMixin:
    def setUp(self):
        super().setUp()
        self.addCleanup(self._reset_module_clients)
        self._reset_module_clients()

    @staticmethod
    def _reset_module_clients():
        retrival.openai_client = None
        retrival.mongo_client = None
        retrival.collection = None


class ValidateCredentialsTests(unittest.TestCase):
    def test_passes_when_all_credentials_present(self):
        with (
            patch.object(retrival, "OPENAI_API_KEY", "key"),
            patch.object(retrival, "MONGODB_USERNAME", "user"),
            patch.object(retrival, "MONGODB_PASSWORD", "pass"),
        ):
            self.assertIsNone(retrival.validate_credentials())

    def test_lists_every_missing_variable(self):
        with (
            patch.object(retrival, "OPENAI_API_KEY", None),
            patch.object(retrival, "MONGODB_USERNAME", "user"),
            patch.object(retrival, "MONGODB_PASSWORD", ""),
            self.assertRaises(ValueError) as ctx,
        ):
            retrival.validate_credentials()

        message = str(ctx.exception)
        self.assertIn("OPENAI_API_KEY", message)
        self.assertIn("MONGODB_PASSWORD", message)
        self.assertNotIn("MONGODB_USERNAME", message)


class BuildMongodbUriTests(unittest.TestCase):
    def test_url_encodes_credentials(self):
        with (
            patch.object(retrival, "MONGODB_USERNAME", "user name"),
            patch.object(retrival, "MONGODB_PASSWORD", "p@ss/word"),
            patch.object(retrival, "MONGODB_CLUSTER", "cluster.example.net"),
        ):
            uri = retrival.build_mongodb_uri()

        self.assertTrue(uri.startswith("mongodb+srv://user+name:p%40ss%2Fword@"))
        self.assertIn("@cluster.example.net/", uri)


class ClientFactoryTests(ResetClientsMixin, unittest.TestCase):
    def test_get_openai_client_validates_and_caches(self):
        with (
            patch.object(retrival, "validate_credentials") as mock_validate,
            patch.object(retrival, "OpenAI") as mock_openai,
        ):
            first = retrival.get_openai_client()
            second = retrival.get_openai_client()

        self.assertIs(first, second)
        self.assertIs(first, mock_openai.return_value)
        mock_validate.assert_called_once_with()
        mock_openai.assert_called_once_with(api_key=retrival.OPENAI_API_KEY)

    def test_get_mongo_collection_pings_and_caches(self):
        mock_client = MagicMock()
        with (
            patch.object(retrival, "validate_credentials"),
            patch.object(retrival, "build_mongodb_uri", return_value="mongodb://uri"),
            patch.object(retrival, "MongoClient", return_value=mock_client),
        ):
            first = retrival.get_mongo_collection()
            second = retrival.get_mongo_collection()

        self.assertIs(first, second)
        mock_client.admin.command.assert_called_once_with("ping")
        expected = mock_client[retrival.DATABASE_NAME][retrival.COLLECTION_NAME]
        self.assertIs(first, expected)

    def test_close_clients_resets_state(self):
        mock_client = MagicMock()
        retrival.mongo_client = mock_client
        retrival.collection = MagicMock()

        retrival.close_clients()

        mock_client.close.assert_called_once_with()
        self.assertIsNone(retrival.mongo_client)
        self.assertIsNone(retrival.collection)

    def test_close_clients_is_noop_without_connection(self):
        self.assertIsNone(retrival.close_clients())


class PromptTests(unittest.TestCase):
    def test_prompt_for_question_reprompts_until_non_empty(self):
        with patch("builtins.input", side_effect=["  ", "", " real question "]):
            self.assertEqual(retrival.prompt_for_question(), "real question")

    def test_confirm_proceed_is_case_insensitive(self):
        with patch("builtins.input", return_value=" YES "):
            self.assertTrue(retrival.confirm_proceed())

        with patch("builtins.input", return_value="y"):
            self.assertFalse(retrival.confirm_proceed())


class GetBrainResponseTests(ResetClientsMixin, unittest.TestCase):
    def _patch_clients(self, openai_client, collection):
        return (
            patch.object(retrival, "get_openai_client", return_value=openai_client),
            patch.object(retrival, "get_mongo_collection", return_value=collection),
        )

    def test_rejects_blank_query(self):
        for query in ("", "   ", None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                retrival.get_brain_response(query)

    def test_builds_vector_search_pipeline_and_returns_answer(self):
        openai_client = make_openai_client(embedding=[0.5, 0.6], answer="the answer")
        collection = MagicMock()
        collection.aggregate.return_value = iter(
            [{"text": "chunk one"}, {"text": "chunk two"}]
        )
        openai_patch, collection_patch = self._patch_clients(openai_client, collection)

        with openai_patch, collection_patch:
            answer = retrival.get_brain_response("  What is vector search?  ")

        self.assertEqual(answer, "the answer")
        openai_client.embeddings.create.assert_called_once_with(
            input="What is vector search?",
            model=retrival.EMBEDDING_MODEL,
        )
        pipeline = collection.aggregate.call_args.args[0]
        vector_stage = pipeline[0]["$vectorSearch"]
        self.assertEqual(vector_stage["index"], retrival.VECTOR_INDEX_NAME)
        self.assertEqual(vector_stage["path"], retrival.VECTOR_PATH)
        self.assertEqual(vector_stage["queryVector"], [0.5, 0.6])
        self.assertEqual(vector_stage["numCandidates"], retrival.NUM_CANDIDATES)
        self.assertEqual(vector_stage["limit"], retrival.RESULT_LIMIT)
        self.assertEqual(pipeline[1], {"$project": {"_id": 0, "text": 1}})

        chat_kwargs = openai_client.chat.completions.create.call_args.kwargs
        self.assertEqual(chat_kwargs["model"], retrival.CHAT_MODEL)
        system_message, user_message = chat_kwargs["messages"]
        self.assertIn("chunk one\nchunk two", system_message["content"])
        self.assertEqual(user_message["content"], "What is vector search?")

    def test_returns_message_when_no_documents_match(self):
        openai_client = make_openai_client()
        collection = MagicMock()
        collection.aggregate.return_value = []
        openai_patch, collection_patch = self._patch_clients(openai_client, collection)

        with openai_patch, collection_patch:
            answer = retrival.get_brain_response("question")

        self.assertEqual(answer, "No relevant documents found in the knowledge base.")
        openai_client.chat.completions.create.assert_not_called()

    def test_returns_message_when_documents_have_no_text(self):
        openai_client = make_openai_client()
        collection = MagicMock()
        collection.aggregate.return_value = [{"text": "   "}, {}]
        openai_patch, collection_patch = self._patch_clients(openai_client, collection)

        with openai_patch, collection_patch:
            answer = retrival.get_brain_response("question")

        self.assertEqual(answer, "Retrieved documents contain no text content.")
        openai_client.chat.completions.create.assert_not_called()

    def test_wraps_server_selection_timeout_as_connection_error(self):
        collection = MagicMock()
        collection.aggregate.side_effect = ServerSelectionTimeoutError("timed out")
        openai_patch, collection_patch = self._patch_clients(
            make_openai_client(), collection
        )

        with openai_patch, collection_patch, self.assertRaises(ConnectionError):
            retrival.get_brain_response("question")

    def test_wraps_operation_failure_with_context(self):
        collection = MagicMock()
        collection.aggregate.side_effect = OperationFailure("index missing")
        openai_patch, collection_patch = self._patch_clients(
            make_openai_client(), collection
        )

        with openai_patch, collection_patch, self.assertRaises(Exception) as ctx:
            retrival.get_brain_response("question")

        self.assertIn("MongoDB query failed", str(ctx.exception))

    def test_wraps_unexpected_errors(self):
        openai_client = make_openai_client()
        openai_client.embeddings.create.side_effect = RuntimeError("api down")
        openai_patch, collection_patch = self._patch_clients(openai_client, MagicMock())

        with openai_patch, collection_patch, self.assertRaises(Exception) as ctx:
            retrival.get_brain_response("question")

        self.assertIn("Error in get_brain_response", str(ctx.exception))


class CliTests(unittest.TestCase):
    def test_parser_defaults_and_flags(self):
        parser = retrival.build_parser()

        defaults = parser.parse_args([])
        self.assertIsNone(defaults.question)
        self.assertFalse(defaults.no_confirm)

        parsed = parser.parse_args(["--question", "Why?", "--no-confirm"])
        self.assertEqual(parsed.question, "Why?")
        self.assertTrue(parsed.no_confirm)

    def test_main_uses_question_argument_without_prompting(self):
        with (
            patch.object(
                retrival, "get_brain_response", return_value="answer"
            ) as mock_response,
            patch.object(retrival, "prompt_for_question") as mock_prompt,
            patch.object(retrival, "confirm_proceed") as mock_confirm,
            patch.object(retrival, "close_clients") as mock_close,
        ):
            exit_code = retrival.main(["--question", "Why?", "--no-confirm"])

        self.assertEqual(exit_code, 0)
        mock_response.assert_called_once_with("Why?")
        mock_prompt.assert_not_called()
        mock_confirm.assert_not_called()
        mock_close.assert_called_once_with()

    def test_main_prompts_when_question_missing(self):
        with (
            patch.object(retrival, "prompt_for_question", return_value="Prompted?"),
            patch.object(retrival, "confirm_proceed", return_value=True),
            patch.object(
                retrival, "get_brain_response", return_value="answer"
            ) as mock_response,
            patch.object(retrival, "close_clients"),
        ):
            exit_code = retrival.main([])

        self.assertEqual(exit_code, 0)
        mock_response.assert_called_once_with("Prompted?")

    def test_main_aborts_when_confirmation_declined(self):
        with (
            patch.object(retrival, "confirm_proceed", return_value=False),
            patch.object(retrival, "get_brain_response") as mock_response,
            patch.object(retrival, "close_clients") as mock_close,
        ):
            exit_code = retrival.main(["--question", "Why?"])

        self.assertEqual(exit_code, 0)
        mock_response.assert_not_called()
        mock_close.assert_called_once_with()

    def test_main_returns_error_code_on_failure(self):
        with (
            patch.object(
                retrival, "get_brain_response", side_effect=RuntimeError("boom")
            ),
            patch.object(retrival, "close_clients") as mock_close,
        ):
            exit_code = retrival.main(["--question", "Why?", "--no-confirm"])

        self.assertEqual(exit_code, 1)
        mock_close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
