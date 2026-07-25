import json
import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional

import retrival

logger = logging.getLogger(__name__)


def run_question(
    question: str,
    answer_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Run a single question through the RAG pipeline and return the answer."""
    if answer_fn is None:
        answer_fn = retrival.get_brain_response
    return answer_fn(question)


def score_answer(answer: str, interactive: bool = True) -> tuple[Optional[int], str]:
    """Apply a simple manual scoring rubric to an answer."""
    if not interactive or not sys.stdin.isatty():
        return None, ""

    print("\nScore the answer on a 1-5 scale:")
    print("1 = poor, 2 = weak, 3 = acceptable, 4 = good, 5 = excellent")
    score = input("Score (1-5): ").strip()
    while score not in {"1", "2", "3", "4", "5"}:
        print("Please enter a number from 1 to 5.")
        score = input("Score (1-5): ").strip()

    notes = input("Optional notes: ").strip()
    return int(score), notes


def evaluate_questions(
    questions: List[str],
    output_path: Optional[Path | str] = None,
    interactive: bool = True,
    score_fn: Optional[Callable[[str], tuple[Optional[int], str]]] = None,
    answer_fn: Optional[Callable[[str], str]] = None,
):
    """
    Evaluate a list of questions and save the results as JSON.

    A failing question is recorded with its error message and evaluation
    continues. If every question fails, the last error is raised so callers
    (and CI) do not treat the run as successful.

    Raises:
        RuntimeError: If no question could be answered.
        OSError: If the results cannot be written to output_path.
    """
    results = []
    failures = []
    scorer = score_fn or (lambda answer: score_answer(answer, interactive=interactive))
    for question in questions:
        try:
            answer = run_question(question, answer_fn=answer_fn)
            score, notes = scorer(answer)
            error = None
        except Exception as exc:
            logger.exception("Evaluation failed for question: %s", question)
            failures.append(exc)
            answer, score, notes = None, None, ""
            error = f"{type(exc).__name__}: {exc}"

        results.append(
            {
                "question": question,
                "answer": answer,
                "score": score,
                "notes": notes,
                "error": error,
            }
        )

    if output_path is not None:
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        except OSError as exc:
            raise OSError(
                f"Failed to write evaluation results to {path}: {exc}"
            ) from exc

    if questions and len(failures) == len(questions):
        raise RuntimeError(
            f"All {len(questions)} questions failed to evaluate; "
            f"last error: {failures[-1]}"
        ) from failures[-1]

    return results


def main() -> int:
    """Run the sample evaluation, reporting failures with a non-zero exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    sample_questions = [
        "What does MongoDB Atlas Vector Search eliminate the need for?",
        "What is the purpose of this system?",
        "Do you know me?",
        "How old I am?",
        "How old are you?",
    ]
    try:
        results = evaluate_questions(
            sample_questions,
            output_path="evaluation_results.json",
        )
    except KeyboardInterrupt:
        print("Evaluation cancelled by user.")
        return 130
    except Exception as exc:
        logger.error("Evaluation run failed: %s", exc)
        return 1
    finally:
        retrival.close_clients()

    for result in results:
        print(
            f"Q: {result['question']}\n"
            f"A: {result['answer']}\n"
            f"Score: {result['score']}\n"
            f"Notes: {result['notes']}\n"
            f"Error: {result['error']}\n",
        )

    return 1 if any(result["error"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
