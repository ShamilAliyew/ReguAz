#!/usr/bin/env python3
"""
scripts/run_ragas_evaluation.py

Run RAGAS evaluation on the enriched generation answers.

Workflow
--------
1.  Load  results/ragas/answers_with_contexts.csv   (produced by enrich_generation_results.py)
2.  Build RAGAS samples  (question / answer / ground_truth / contexts)
3.  Call  Llama-3.3-70b  (via NVIDIA NIM) as the judge LLM
4.  Save  results/ragas/ragas_scores.csv            (per-question scores)
        results/ragas/ragas_metrics.json            (aggregated means)

Usage
-----
    # Full evaluation (all rows in the enriched CSV)
    poetry run python scripts/run_ragas_evaluation.py

    # Quick smoke test with 5 questions
    poetry run python scripts/run_ragas_evaluation.py --limit 5

    # Custom paths
    poetry run python scripts/run_ragas_evaluation.py \\
        --input  results/ragas/answers_with_contexts.csv \\
        --output results/ragas/
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make project root importable regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.reguaz import config
from backend.reguaz.evaluation.ragas_evaluator import RagasEvaluator
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "ragas_evaluation.log")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS metrics on enriched generation evaluation output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.RESULTS_PATH / "ragas" / "answers_with_contexts.csv"),
        help="Path to the enriched answers CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.RESULTS_PATH / "ragas"),
        help="Directory where RAGAS result files will be saved.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "NVIDIA NIM model to use as judge LLM. "
            "Defaults to JUDGE_MODEL env variable or 'meta/llama-3.3-70b-instruct'."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions (omit to run all).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("=== Starting RAGAS Evaluation ===")
    logger.info("  Input  : %s", args.input)
    logger.info("  Output : %s", args.output)
    logger.info("  Model  : %s", args.judge_model or "from JUDGE_MODEL env / default")
    logger.info("  Limit  : %s", args.limit if args.limit else "All questions")

    try:
        evaluator = RagasEvaluator(
            judge_model=args.judge_model,
            output_dir=args.output,
        )
        scores = evaluator.evaluate(
            input_csv=args.input,
            limit=args.limit,
        )
    except EnvironmentError as exc:
        logger.critical("Configuration error: %s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.critical("Input file not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("RAGAS evaluation failed: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("=== RAGAS Evaluation Completed Successfully ===")
    for metric, score in scores.items():
        logger.info("  %-25s : %.4f", metric, score)


if __name__ == "__main__":
    main()
