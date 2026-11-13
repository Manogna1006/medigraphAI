"""
Usage:
    python -m scripts.run_benchmark
    python -m scripts.run_benchmark --questions data/eval_questions.json --out reports/latest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.evaluation.benchmark_runner import run_benchmark, save_report
from app.utils.logger import logger

DEFAULT_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"
DEFAULT_OUT_PATH = Path(__file__).resolve().parent.parent / "reports" / "latest_benchmark.json"

FALLBACK_QUESTIONS = [
    "What conditions can cause fever and cough together?",
    "What is the recommended treatment for bacterial pneumonia?",
    "A patient has chest pain — what should be considered?",
    "What treats type 2 diabetes mellitus?",
    "Is ibuprofen safe for a patient with a penicillin allergy?",
]


def load_questions(path: Path) -> list[str]:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(q, str) for q in data):
            return data
        logger.warning("{} did not contain a JSON array of strings; using fallback questions.", path)
    return FALLBACK_QUESTIONS


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MediGraph AI hallucination benchmark")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    logger.info("Running benchmark over {} questions", len(questions))
    result = run_benchmark(questions)

    logger.info(
        "Aggregate hallucination rate: {:.2%} | groundedness: {:.2%} | abstention rate: {:.2%}",
        result.aggregate_hallucination_rate, result.aggregate_groundedness_score, result.abstention_rate,
    )
    save_report(result, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
