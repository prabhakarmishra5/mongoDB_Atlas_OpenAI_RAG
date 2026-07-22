import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate_rag


class EvaluateRagTests(unittest.TestCase):
    def test_evaluate_questions_returns_results_and_writes_report(self):
        questions = ["What is MongoDB?", "How does vector search work?"]

        with (
            patch(
                "evaluate_rag.run_question",
                return_value="mock answer",
            ) as mock_run,
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            output_path = Path(tmp_dir) / "results.json"
            results = evaluate_rag.evaluate_questions(
                questions,
                output_path=output_path,
                interactive=False,
            )

            self.assertEqual(results[0]["question"], "What is MongoDB?")
            self.assertEqual(results[0]["answer"], "mock answer")
            self.assertEqual(results[0]["score"], None)
            self.assertEqual(results[0]["notes"], "")
            self.assertEqual(mock_run.call_count, 2)
            self.assertTrue(output_path.exists())
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved[0]["question"], "What is MongoDB?")


if __name__ == "__main__":
    unittest.main()
