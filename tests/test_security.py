import unittest

from security import MAX_QUERY_LENGTH, redact_credentials, sanitize_query


class RedactCredentialsTests(unittest.TestCase):
    def test_redacts_mongodb_uri_credentials(self):
        message = (
            "connection failed for mongodb+srv://user:s3cret@cluster0.example.net/db"
        )

        self.assertEqual(
            redact_credentials(message),
            "connection failed for mongodb+srv://***:***@cluster0.example.net/db",
        )

    def test_leaves_messages_without_credentials_untouched(self):
        self.assertEqual(redact_credentials("plain failure"), "plain failure")


class SanitizeQueryTests(unittest.TestCase):
    def test_strips_and_returns_query(self):
        self.assertEqual(sanitize_query("  hello  "), "hello")

    def test_rejects_empty_query(self):
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                sanitize_query(value)

    def test_rejects_oversized_query(self):
        with self.assertRaises(ValueError):
            sanitize_query("a" * (MAX_QUERY_LENGTH + 1))

    def test_rejects_null_bytes(self):
        with self.assertRaises(ValueError):
            sanitize_query("hello\x00world")


if __name__ == "__main__":
    unittest.main()
