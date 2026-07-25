import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import rag_common


class RagCommonTests(unittest.TestCase):
    def test_build_mongodb_uri_encodes_credentials(self):
        credentials = {
            "MONGODB_USERNAME": "user@example.com",
            "MONGODB_PASSWORD": "p@ss word",
            "MONGODB_CLUSTER": "cluster0.example.net",
        }
        with patch.multiple(rag_common, **credentials):
            uri = rag_common.build_mongodb_uri()

        self.assertEqual(
            uri,
            "mongodb+srv://user%40example.com:p%40ss+word@cluster0.example.net"
            "/?appName=Cluster0&compressors=zlib",
        )

    def test_validate_credentials_lists_missing_variables(self):
        credentials = {
            "OPENAI_API_KEY": None,
            "MONGODB_USERNAME": "user",
            "MONGODB_PASSWORD": None,
        }
        with (
            patch.multiple(rag_common, **credentials),
            self.assertRaises(ValueError) as ctx,
        ):
            rag_common.validate_credentials()

        self.assertIn("OPENAI_API_KEY", str(ctx.exception))
        self.assertIn("MONGODB_PASSWORD", str(ctx.exception))
        self.assertNotIn("MONGODB_USERNAME", str(ctx.exception))

    def test_embed_text_returns_first_embedding(self):
        openai_client = MagicMock()
        openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2])]
        )

        embedding = rag_common.embed_text(openai_client, "hello", model="test-model")

        self.assertEqual(embedding, [0.1, 0.2])
        openai_client.embeddings.create.assert_called_once_with(
            input="hello",
            model="test-model",
        )

    def test_confirm_yes_is_case_insensitive(self):
        with patch("builtins.input", return_value=" YES "):
            self.assertTrue(rag_common.confirm_yes("prompt: "))

        with patch("builtins.input", return_value="nope"):
            self.assertFalse(rag_common.confirm_yes("prompt: "))

    def test_prompt_non_empty_retries_until_answer(self):
        with patch("builtins.input", side_effect=["", "  ", "answer"]):
            self.assertEqual(rag_common.prompt_non_empty("prompt: "), "answer")

    def test_prompt_choice_retries_until_valid(self):
        with patch("builtins.input", side_effect=["9", "3"]):
            choice = rag_common.prompt_choice("score: ", {"1", "2", "3"}, "invalid")

        self.assertEqual(choice, "3")

    def test_write_json_and_read_json_dict_round_trip(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "data.json"

            rag_common.write_json(path, {"a": 1})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1})
            self.assertEqual(rag_common.read_json_dict(path, "unused"), {"a": 1})

    def test_read_json_dict_handles_missing_and_invalid_files(self):
        with TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.json"
            invalid = Path(tmp_dir) / "invalid.json"
            invalid.write_text("not json", encoding="utf-8")
            not_a_dict = Path(tmp_dir) / "list.json"
            not_a_dict.write_text("[1, 2]", encoding="utf-8")

            self.assertEqual(rag_common.read_json_dict(missing, "warn"), {})
            self.assertEqual(rag_common.read_json_dict(invalid, "warn"), {})
            self.assertEqual(rag_common.read_json_dict(not_a_dict, "warn"), {})


if __name__ == "__main__":
    unittest.main()
