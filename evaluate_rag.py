import json
import sys
from pathlib import Path
from typing import Callable, List, Optional

import retrival


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
):
    """Evaluate a list of questions and save the results as JSON."""
    results = []
    scorer = score_fn or (lambda answer: score_answer(answer, interactive=interactive))
    for question in questions:
        answer = run_question(question)
        score, notes = scorer(answer)
        results.append(
            {
                "question": question,
                "answer": answer,
                "score": score,
                "notes": notes,
            }
        )

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


if __name__ == "__main__":
    sample_questions = [
        "What does MongoDB Atlas Vector Search eliminate the need for?",
        "What is the purpose of this system?",
        "Do you know me?",
        "How old I am?",
        "How old are you?",
    ]
    results = evaluate_questions(
        sample_questions,
        output_path="evaluation_results.json",
    )
    for result in results:
        print(
            f"Q: {result['question']}\n"
            f"A: {result['answer']}\n"
            f"Score: {result['score']}\n"
            f"Notes: {result['notes']}\n",
        )
