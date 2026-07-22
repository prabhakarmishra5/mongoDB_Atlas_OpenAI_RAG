import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ingestion as ingest_script
import retrival as retrieval_script


class IngestionFlowTests(unittest.TestCase):
    def test_get_pdf_files_filters_non_pdfs(self):
        with TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            (temp_dir / "doc.pdf").write_bytes(b"pdf")
            (temp_dir / "notes.txt").write_text("not a pdf", encoding="utf-8")

            pdf_files = ingest_script.get_pdf_files(temp_dir)

            self.assertEqual([path.name for path in pdf_files], ["doc.pdf"])

    def test_load_and_save_ingestion_log(self):
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "ingestion_log.json"

            ingest_script.save_ingestion_log(
                log_path,
                "sample.pdf",
                "2026-07-22 12:00:00",
            )
            history = ingest_script.load_ingestion_log(log_path)

            self.assertEqual(history["sample.pdf"], "2026-07-22 12:00:00")

    def test_confirm_ingestion_requires_yes(self):
        with patch("builtins.input", return_value="No"):
            self.assertFalse(ingest_script.confirm_ingestion("sample.pdf", None))

        with patch("builtins.input", return_value="Yes"):
            self.assertTrue(
                ingest_script.confirm_ingestion(
                    "sample.pdf",
                    "2026-07-22 12:00:00",
                )
            )

    def test_prompt_for_question_and_confirmation_requires_yes(self):
        with patch(
            "builtins.input",
            side_effect=[
                "What does Atlas Vector Search eliminate the need for?",
                "Yes",
            ],
        ):
            question = retrieval_script.prompt_for_question()
            confirmed = retrieval_script.confirm_proceed()

        self.assertEqual(
            question,
            "What does Atlas Vector Search eliminate the need for?",
        )
        self.assertTrue(confirmed)


if __name__ == "__main__":
    unittest.main()
