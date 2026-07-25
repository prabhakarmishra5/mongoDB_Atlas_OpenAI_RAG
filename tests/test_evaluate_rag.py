import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import evaluate_rag


class RunQuestionTests(unittest.TestCase):
    def test_uses_custom_answer_function(self):
        answer_fn = MagicMock(return_value="custom answer")

        self.assertEqual(
            evaluate_rag.run_question("What is RAG?", answer_fn),
            "custom answer",
        )
        answer_fn.assert_called_once_with("What is RAG?")

    def test_defaults_to_retrieval_pipeline(self):
        with patch(
            "retrival.get_brain_response",
            return_value="pipeline answer",
        ) as mock_brain:
            self.assertEqual(
                evaluate_rag.run_question("What is RAG?"),
                "pipeline answer",
            )

        mock_brain.assert_called_once_with("What is RAG?")


class ScoreAnswerTests(unittest.TestCase):
    def test_returns_empty_score_when_not_interactive(self):
        self.assertEqual(
            evaluate_rag.score_answer("answer", interactive=False), (None, "")
        )

    def test_returns_empty_score_without_a_tty(self):
        with patch("sys.stdin.isatty", return_value=False):
            self.assertEqual(evaluate_rag.score_answer("answer"), (None, ""))

    def test_reprompts_until_score_is_valid(self):
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=["9", "abc", " 4 ", "solid"]),
        ):
            score, notes = evaluate_rag.score_answer("answer")

        self.assertEqual(score, 4)
        self.assertEqual(notes, "solid")


class EvaluateQuestionsTests(unittest.TestCase):
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

    def test_uses_custom_scorer_and_skips_report_without_output_path(self):
        score_fn = MagicMock(return_value=(5, "excellent"))

        with patch("evaluate_rag.run_question", return_value="mock answer"):
            results = evaluate_rag.evaluate_questions(["Q1"], score_fn=score_fn)

        score_fn.assert_called_once_with("mock answer")
        self.assertEqual(
            results,
            [
                {
                    "question": "Q1",
                    "answer": "mock answer",
                    "score": 5,
                    "notes": "excellent",
                }
            ],
        )

    def test_creates_missing_parent_directories_for_report(self):
        with (
            patch("evaluate_rag.run_question", return_value="mock answer"),
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            output_path = Path(tmp_dir) / "nested" / "reports" / "results.json"
            evaluate_rag.evaluate_questions(
                ["Q1"],
                output_path=str(output_path),
                interactive=False,
            )

            self.assertTrue(output_path.exists())

    def test_returns_empty_results_for_no_questions(self):
        self.assertEqual(evaluate_rag.evaluate_questions([], interactive=False), [])


if __name__ == "__main__":
    unittest.main()
