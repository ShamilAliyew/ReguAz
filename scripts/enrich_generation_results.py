#!/usr/bin/env python3
"""
Enrich generation evaluation results with full chunk contexts.

Reads a lightweight ``answers.csv`` produced by
:class:`~backend.reguaz.evaluation.generation_evaluator.GenerationEvaluator`
and creates ``answers_with_contexts.csv`` that includes complete chunk text
and metadata — ready for RAGAS evaluation.

Usage
-----
::

    poetry run python scripts/enrich_generation_results.py

    poetry run python scripts/enrich_generation_results.py \\
        --input results/generation/gemma/answers.csv \\
        --output results/ragas/answers_with_contexts.csv
"""

import argparse
import sys
from pathlib import Path

# Add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.reguaz import config
from backend.reguaz.evaluation.context_enricher import ContextEnricher
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "context_enrichment.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich generation answers with full chunk contexts for RAGAS.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(
            config.RESULTS_PATH / "generation" / "gemma" / "answers.csv"
        ),
        help="Path to the generation evaluation CSV (answers.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            config.RESULTS_PATH / "ragas" / "answers_with_contexts.csv"
        ),
        help="Destination path for the enriched CSV.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default=str(config.CHUNKS_DIR),
        help=f"Root directory containing chunk JSONL files. Default: {config.CHUNKS_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=== Starting Context Enrichment ===")
    logger.info("  Input CSV  : %s", args.input)
    logger.info("  Output CSV : %s", args.output)
    logger.info("  Chunks Dir : %s", args.chunks_dir)

    try:
        enricher = ContextEnricher(chunks_dir=args.chunks_dir)
        output_path = enricher.enrich(
            input_csv=args.input,
            output_csv=args.output,
        )
        logger.info("=== Context Enrichment Completed Successfully ===")
        logger.info("Output: %s", output_path)
    except Exception as exc:
        logger.critical("Context enrichment failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
