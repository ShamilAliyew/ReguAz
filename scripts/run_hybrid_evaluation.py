"""
scripts/run_hybrid_evaluation.py

Production-quality Hybrid Retrieval Evaluation pipeline for ReguAZ.

For each embedding model (BGE-M3, E5) this script:
  1. Builds a HybridRetriever (ChromaDB semantic + BM25 + RRF fusion).
  2. Iterates over every question in the gold evaluation dataset.
  3. Computes Recall@K, Precision@K, MRR@K, and nDCG@K.
  4. Writes per-question CSV, retrieval-results CSV, aggregate metrics JSON,
     and a cross-model comparison CSV.

Usage
-----
    # Evaluate both models (default)
    python scripts/run_hybrid_evaluation.py

    # Evaluate a single model
    python scripts/run_hybrid_evaluation.py --model bge_m3

    # Custom settings
    python scripts/run_hybrid_evaluation.py \\
        --model all \\
        --top-k 10 \\
        --top-k-candidates 20 \\
        --rrf-k 60 \\
        --dataset data/evaluation/gold_dataset_for_embedding_excel.xlsx \\
        --chunks-dir data/processed/chunks \\
        --chroma-dir data/chroma \\
        --results-dir results/hybrid_retrieval

Output
------
    results/hybrid_retrieval/
        bge_m3/
            metrics.json
            per_question.csv
            retrieval_results.csv
        e5/
            metrics.json
            per_question.csv
            retrieval_results.csv
        comparison.csv

    logs/
        hybrid_evaluation_bge_m3.log
        hybrid_evaluation_e5.log
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: make the project root importable from the scripts/ directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.reguaz.retrieval.hybrid_retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS: list[str] = ["bge_m3", "e5"]

# Log file name template.
LOG_FILE_TEMPLATE = "hybrid_evaluation_{model_name}.log"

# Column names written to result CSVs — declared once to avoid typos.
_METRIC_COLUMNS: list[str] = [
    "recall@1", "recall@3", "recall@5", "recall@10",
    "precision@1", "precision@3", "precision@5", "precision@10",
    "mrr@10",
    "ndcg@3", "ndcg@5", "ndcg@10",
]

DEFAULT_DATASET_PATH = Path("data/evaluation/gold_dataset_for_embedding_excel.xlsx")
DEFAULT_CHUNKS_DIR = "data/processed/chunks"
DEFAULT_CHROMA_DIR = "data/chroma"
DEFAULT_RESULTS_DIR = str(config.RESULTS_PATH / "hybrid_retrieval")
DEFAULT_TOP_K = 10
DEFAULT_TOP_K_CANDIDATES = 20
DEFAULT_RRF_K = 60


# ===========================================================================
# Logger setup
# ===========================================================================

def setup_hybrid_logger(model_name: str) -> logging.Logger:
    """
    Create a model-specific logger that writes simultaneously to:
      - the console (stdout)
      - ``logs/hybrid_evaluation_{model_name}.log`` (append mode)

    A duplicate-handler guard ensures the function is safe to call multiple
    times within the same process.

    Parameters
    ----------
    model_name : str
        Normalised model name (e.g. ``"bge_m3"`` or ``"e5"``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Reconfigure sys.stdout to support UTF-8 on Windows consoles (e.g. Git Bash)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / LOG_FILE_TEMPLATE.format(model_name=model_name)
    logger_name = f"hybrid_evaluation_{model_name}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Guard against duplicate handlers when the function is called twice.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ===========================================================================
# Dataset helpers
# ===========================================================================

def load_gold_dataset(xlsx_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load and validate the gold evaluation dataset from an Excel file.

    Parameters
    ----------
    xlsx_path : Path
        Path to the ``.xlsx`` gold dataset.
    logger : logging.Logger

    Returns
    -------
    pd.DataFrame
        Validated dataframe with at minimum the required columns.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If any required columns are missing.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"load_gold_dataset: dataset file not found: '{xlsx_path}'"
        )

    logger.info("Loading gold evaluation dataset from: %s", xlsx_path)
    df = pd.read_excel(xlsx_path)

    required_columns = {
        "question",
        "relevant_chunk_ids",
        "expected_document_ids",
        "category",
        "difficulty",
        "notes",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"load_gold_dataset: dataset is missing required columns: {sorted(missing)}"
        )

    logger.info("Loaded %d evaluation question(s).", len(df))
    return df


def parse_chunk_ids(raw_value: Any) -> list[str]:
    """
    Parse a ``relevant_chunk_ids`` cell into a deduplicated list of chunk IDs.

    Handles:
    - Single string IDs
    - Pipe-separated string IDs (``"id1|id2|id3"``)
    - Numeric values (cast to string)
    - NaN / None (returns empty list)

    Parameters
    ----------
    raw_value : Any
        Raw cell value from the dataframe.

    Returns
    -------
    list[str]
        Ordered, deduplicated list of stripped chunk ID strings.
    """
    if pd.isna(raw_value):
        return []

    raw_str = str(raw_value).strip()
    parts = [part.strip() for part in raw_str.split("|")]

    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            unique.append(part)

    return unique


# ===========================================================================
# Metric computation
# ===========================================================================

def _dcg_at_k(hits: list[int], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at *k* for a binary relevance list.

    Parameters
    ----------
    hits : list[int]
        Binary relevance list (1 = relevant, 0 = not relevant), ordered by
        retrieval rank (index 0 = rank 1).
    k : int
        Cut-off position.

    Returns
    -------
    float
        DCG@k value.
    """
    dcg = 0.0
    for i, rel in enumerate(hits[:k], start=1):
        if rel:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def compute_metrics(
    relevant_ids: list[str],
    retrieved_ids: list[str],
    top_k: int,
) -> dict[str, Any]:
    """
    Compute retrieval metrics for a single question.

    Metrics
    -------
    Recall@K    : |retrieved ∩ relevant| / |relevant|
    Precision@K : |retrieved ∩ relevant| / K
    MRR@K       : 1 / rank_of_first_relevant  (0 if none in top-K)
    nDCG@K      : normalised DCG with binary relevance

    Parameters
    ----------
    relevant_ids : list[str]
        Ground-truth relevant chunk IDs (deduplicated).
    retrieved_ids : list[str]
        Ordered list of retrieved IDs (position 0 = rank 1).
    top_k : int
        Maximum number of retrieved results to consider.

    Returns
    -------
    dict[str, Any]
        Dictionary with all computed metric values plus a
        ``rank_of_first_relevant`` field (int, -1 if not found).
    """
    import unicodedata
    relevant_set = {unicodedata.normalize("NFC", rid) for rid in relevant_ids}
    retrieved_truncated = retrieved_ids[:top_k]
    binary_hits = [1 if unicodedata.normalize("NFC", rid) in relevant_set else 0 for rid in retrieved_truncated]

    # Rank of first relevant (1-indexed; -1 if not found in top_k)
    rank_of_first: int = -1
    for rank, rid in enumerate(retrieved_truncated, start=1):
        if unicodedata.normalize("NFC", rid) in relevant_set:
            rank_of_first = rank
            break

    # --- Recall@K ---
    def recall_at(k: int) -> float:
        if not relevant_set:
            return 0.0
        hits = sum(binary_hits[:k])
        return hits / len(relevant_set)

    # --- Precision@K ---
    def precision_at(k: int) -> float:
        if k == 0:
            return 0.0
        hits = sum(binary_hits[:k])
        return hits / k

    # --- MRR@K ---
    mrr = (1.0 / rank_of_first) if rank_of_first != -1 else 0.0

    # --- nDCG@K ---
    def ndcg_at(k: int) -> float:
        if not relevant_set:
            return 0.0
        dcg = _dcg_at_k(binary_hits, k)
        # Ideal DCG: place all relevant docs at the top.
        ideal_hits = [1] * min(len(relevant_set), k)
        idcg = _dcg_at_k(ideal_hits, k)
        return dcg / idcg if idcg > 0 else 0.0

    return {
        "rank_of_first_relevant": rank_of_first,
        "recall@1": recall_at(1),
        "recall@3": recall_at(3),
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "precision@1": precision_at(1),
        "precision@3": precision_at(3),
        "precision@5": precision_at(5),
        "precision@10": precision_at(10),
        "mrr@10": mrr,
        "ndcg@3": ndcg_at(3),
        "ndcg@5": ndcg_at(5),
        "ndcg@10": ndcg_at(10),
    }


# ===========================================================================
# Per-model evaluation loop
# ===========================================================================

def evaluate_model(
    model_name: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Run the hybrid retrieval evaluation pipeline for one embedding model.

    For each question in *df*:
    - Calls ``HybridRetriever.retrieve()``
    - Computes all metrics via ``compute_metrics()``
    - Accumulates per-question rows for two output DataFrames

    Parameters
    ----------
    model_name : str
        Normalised model identifier (``"e5"`` or ``"bge_m3"``).
    df : pd.DataFrame
        Gold evaluation dataset (output of ``load_gold_dataset()``).
    args : argparse.Namespace
        Parsed CLI arguments.
    logger : logging.Logger
        Model-specific logger.

    Returns
    -------
    per_question_df : pd.DataFrame
        One row per evaluated question with all metric columns.
    retrieval_df : pd.DataFrame
        One row per question showing the retrieved IDs and RRF scores.
    aggregate : dict[str, Any]
        Mean of every metric column across all evaluated questions.
    """
    logger.info("=" * 80)
    logger.info("HYBRID EVALUATION  |  model: %s", model_name.upper())
    logger.info("  top_k            : %d", args.top_k)
    logger.info("  top_k_candidates : %d", args.top_k_candidates)
    logger.info("  rrf_k            : %d", args.rrf_k)
    logger.info("  chroma_dir       : %s", args.chroma_dir)
    logger.info("  chunks_dir       : %s", args.chunks_dir)
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Initialise the HybridRetriever
    # ------------------------------------------------------------------
    logger.info("Initialising HybridRetriever for model '%s' …", model_name)
    t_init = time.perf_counter()

    try:
        retriever = HybridRetriever(
            model_name=model_name,
            chroma_dir=args.chroma_dir,
            chunks_dir=args.chunks_dir,
            top_k_semantic=args.top_k_candidates,
            top_k_bm25=args.top_k_candidates,
            rrf_k=args.rrf_k,
        )
    except Exception:
        logger.error(
            "Failed to initialise HybridRetriever for model '%s':\n%s",
            model_name,
            traceback.format_exc(),
        )
        empty_df = pd.DataFrame()
        empty_aggregate: dict[str, Any] = {col: 0.0 for col in _METRIC_COLUMNS}
        empty_aggregate["num_questions_evaluated"] = 0
        return empty_df, empty_df, empty_aggregate

    logger.info(
        "HybridRetriever ready in %.2f s.", time.perf_counter() - t_init
    )

    # ------------------------------------------------------------------
    # Question-level evaluation loop
    # ------------------------------------------------------------------
    per_question_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    num_skipped = 0
    num_errors = 0

    total_questions = len(df)
    logger.info("Starting evaluation over %d question(s) …", total_questions)
    t_loop = time.perf_counter()

    for idx, row in df.iterrows():
        question: str = str(row["question"])
        relevant_ids: list[str] = parse_chunk_ids(row["relevant_chunk_ids"])

        if not relevant_ids:
            logger.warning(
                "[Q%03d] No relevant_chunk_ids found for question '%s…' — skipping.",
                idx,
                question[:60],
            )
            num_skipped += 1
            continue

        logger.info(
            "[Q%03d/%03d] Retrieving for: '%s…'",
            idx,
            total_questions - 1,
            question[:70],
        )

        # Hybrid retrieval
        t_q = time.perf_counter()
        try:
            results = retriever.retrieve(query=question, top_k=args.top_k)
        except Exception:
            logger.error(
                "[Q%03d] HybridRetriever.retrieve failed:\n%s",
                idx,
                traceback.format_exc(),
            )
            num_errors += 1
            continue

        retrieved_ids: list[str] = [r["id"] for r in results]
        elapsed_q = time.perf_counter() - t_q

        # Compute metrics
        metrics = compute_metrics(
            relevant_ids=relevant_ids,
            retrieved_ids=retrieved_ids,
            top_k=args.top_k,
        )

        # Log BM25 and semantic rank details for the first few questions.
        if idx < 5:
            overlap = set(relevant_ids) & set(retrieved_ids)
            logger.info(
                "[Q%03d] Expected: %s | Retrieved top-%d: %s | Overlap: %s",
                idx,
                relevant_ids,
                args.top_k,
                retrieved_ids[:args.top_k],
                list(overlap) if overlap else "NONE",
            )
            for r in results[:5]:
                logger.info(
                    "[Q%03d]   Rank %d | id=%-40s | rrf=%.5f | sem_rank=%s | bm25_rank=%s",
                    idx,
                    r["rank"],
                    r["id"],
                    r["rrf_score"],
                    r.get("semantic_rank", "-"),
                    r.get("bm25_rank", "-"),
                )

        logger.info(
            "[Q%03d] recall@10=%.3f | precision@10=%.3f | mrr@10=%.3f | ndcg@10=%.3f | "
            "time=%.2f s",
            idx,
            metrics["recall@10"],
            metrics["precision@10"],
            metrics["mrr@10"],
            metrics["ndcg@10"],
            elapsed_q,
        )

        # --- Accumulate per-question row ---
        per_question_rows.append(
            {
                "question_idx": idx,
                "question": question,
                "category": str(row.get("category", "")),
                "difficulty": str(row.get("difficulty", "")),
                "expected_chunk_ids": "|".join(relevant_ids),
                "expected_document_ids": str(row.get("expected_document_ids", "")),
                **metrics,
            }
        )

        # --- Accumulate retrieval row ---
        rrf_scores_str = "|".join(
            f"{r['id']}:{r['rrf_score']:.6f}" for r in results
        )
        sem_ranks_str = "|".join(
            str(r.get("semantic_rank") or "") for r in results
        )
        bm25_ranks_str = "|".join(
            str(r.get("bm25_rank") or "") for r in results
        )
        retrieval_rows.append(
            {
                "question_idx": idx,
                "question": question,
                "expected_chunk_ids": "|".join(relevant_ids),
                f"retrieved_top{args.top_k}_ids": "|".join(retrieved_ids),
                "rrf_scores": rrf_scores_str,
                "semantic_ranks": sem_ranks_str,
                "bm25_ranks": bm25_ranks_str,
            }
        )

    loop_elapsed = time.perf_counter() - t_loop
    logger.info(
        "Evaluation loop finished in %.2f s | evaluated=%d | skipped=%d | errors=%d",
        loop_elapsed,
        len(per_question_rows),
        num_skipped,
        num_errors,
    )

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    per_question_df = pd.DataFrame(per_question_rows)
    retrieval_df = pd.DataFrame(retrieval_rows)

    if per_question_df.empty:
        logger.warning(
            "No results produced for model '%s'. Returning zero aggregates.", model_name
        )
        aggregate = {col: 0.0 for col in _METRIC_COLUMNS}
        aggregate["num_questions_evaluated"] = 0
        return per_question_df, retrieval_df, aggregate

    aggregate: dict[str, Any] = {
        col: float(per_question_df[col].mean())
        for col in _METRIC_COLUMNS
        if col in per_question_df.columns
    }
    aggregate["num_questions_evaluated"] = len(per_question_df)

    # Log aggregate summary
    logger.info("-" * 60)
    logger.info("AGGREGATE METRICS  |  model: %s", model_name.upper())
    for metric_name, value in aggregate.items():
        if isinstance(value, float):
            logger.info("  %-30s : %.4f", metric_name, value)
        else:
            logger.info("  %-30s : %s", metric_name, value)
    logger.info("-" * 60)

    return per_question_df, retrieval_df, aggregate


# ===========================================================================
# Output helpers
# ===========================================================================

def save_model_outputs(
    model_name: str,
    per_question_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    aggregate: dict[str, Any],
    results_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Persist all per-model output files.

    Creates the model-specific subdirectory if it does not exist, then writes:
    - ``metrics.json``
    - ``per_question.csv``
    - ``retrieval_results.csv``

    Parameters
    ----------
    model_name : str
        Normalised model name (used as subdirectory name).
    per_question_df : pd.DataFrame
        Per-question metrics DataFrame.
    retrieval_df : pd.DataFrame
        Retrieval results DataFrame.
    aggregate : dict[str, Any]
        Aggregate metrics dictionary.
    results_dir : Path
        Root results directory (e.g. ``results/hybrid_retrieval``).
    logger : logging.Logger
    """
    model_dir = results_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics_path = model_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2, ensure_ascii=False)
    logger.info("Saved metrics         → %s", metrics_path)

    # per_question.csv
    per_question_path = model_dir / "per_question.csv"
    if not per_question_df.empty:
        per_question_df.to_csv(per_question_path, index=False, encoding="utf-8")
    else:
        # Write an empty file to signal the run completed.
        per_question_path.touch()
    logger.info("Saved per_question    → %s", per_question_path)

    # retrieval_results.csv
    retrieval_path = model_dir / "retrieval_results.csv"
    if not retrieval_df.empty:
        retrieval_df.to_csv(retrieval_path, index=False, encoding="utf-8")
    else:
        retrieval_path.touch()
    logger.info("Saved retrieval_results → %s", retrieval_path)


def build_comparison_csv(
    all_aggregates: dict[str, dict[str, Any]],
    results_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Write a side-by-side comparison CSV for all evaluated models.

    Parameters
    ----------
    all_aggregates : dict[str, dict[str, Any]]
        Mapping of ``{model_name: aggregate_metrics_dict}``.
    results_dir : Path
        Root results directory.
    logger : logging.Logger
    """
    rows: list[dict[str, Any]] = []
    for model_name, agg in all_aggregates.items():
        rows.append({"model": model_name, **agg})

    comparison_df = pd.DataFrame(rows)
    comparison_path = results_dir / "comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")
    logger.info("Saved comparison CSV  → %s", comparison_path)

    # Print a readable summary to the console.
    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL — MODEL COMPARISON")
    print("=" * 70)
    for col in comparison_df.columns:
        if col == "model":
            continue
        row_values = "  ".join(
            f"{row['model']:>10}: {row[col]:.4f}" if isinstance(row[col], float)
            else f"{row['model']:>10}: {row[col]}"
            for _, row in comparison_df.iterrows()
        )
        print(f"  {col:<30}  {row_values}")
    print("=" * 70 + "\n")


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ReguAZ Hybrid Retrieval Evaluation Pipeline — "
            "evaluates ChromaDB semantic search + BM25 + RRF fusion."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", *MODELS],
        help="Embedding model to evaluate. 'all' evaluates all supported models.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of final fused results to return per query.",
    )
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=DEFAULT_TOP_K_CANDIDATES,
        help=(
            "Number of candidates to retrieve from each source (semantic & BM25) "
            "before RRF fusion.  Must be >= --top-k."
        ),
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF smoothing constant (canonical value = 60).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the gold evaluation dataset (.xlsx).",
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default=DEFAULT_CHUNKS_DIR,
        help="Root directory containing chunk JSONL files.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=DEFAULT_CHROMA_DIR,
        help="Path to the ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory where all result files will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point.  Orchestrates the full hybrid evaluation pipeline.

    Flow
    ----
    1. Parse CLI arguments.
    2. Set up a minimal root logger for pre-model bootstrap messages.
    3. Load the gold evaluation dataset once (shared across all models).
    4. For each selected model:
       a. Set up a model-specific logger (console + file).
       b. Run ``evaluate_model()``.
       c. Save outputs via ``save_model_outputs()``.
    5. Build and save the cross-model ``comparison.csv``.
    """
    args = _parse_args()

    # Validate --top-k-candidates >= --top-k
    if args.top_k_candidates < args.top_k:
        print(
            f"WARNING: --top-k-candidates ({args.top_k_candidates}) is less than "
            f"--top-k ({args.top_k}).  Overriding --top-k-candidates to {args.top_k}.",
            file=sys.stderr,
        )
        args.top_k_candidates = args.top_k

    # Resolve paths
    dataset_path = Path(args.dataset)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Bootstrap logger (used before model-specific loggers are available)
    bootstrap_logger = setup_hybrid_logger("pipeline")

    bootstrap_logger.info("=" * 80)
    bootstrap_logger.info("ReguAZ  Hybrid Retrieval Evaluation Pipeline")
    bootstrap_logger.info("  model(s)         : %s", args.model)
    bootstrap_logger.info("  top_k            : %d", args.top_k)
    bootstrap_logger.info("  top_k_candidates : %d", args.top_k_candidates)
    bootstrap_logger.info("  rrf_k            : %d", args.rrf_k)
    bootstrap_logger.info("  dataset          : %s", dataset_path)
    bootstrap_logger.info("  chunks_dir       : %s", args.chunks_dir)
    bootstrap_logger.info("  chroma_dir       : %s", args.chroma_dir)
    bootstrap_logger.info("  results_dir      : %s", results_dir)
    bootstrap_logger.info("=" * 80)

    t_pipeline_start = time.perf_counter()

    # Load gold dataset once — shared across all models.
    try:
        df = load_gold_dataset(dataset_path, bootstrap_logger)
    except (FileNotFoundError, ValueError) as exc:
        bootstrap_logger.error("Cannot load evaluation dataset: %s", exc)
        sys.exit(1)

    # Resolve which models to evaluate.
    models_to_evaluate = MODELS if args.model == "all" else [args.model]

    all_aggregates: dict[str, dict[str, Any]] = {}

    for model_name in models_to_evaluate:
        model_logger = setup_hybrid_logger(model_name)

        model_logger.info(
            "Pipeline start for model '%s' at %s",
            model_name,
            time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        t_model_start = time.perf_counter()

        per_question_df, retrieval_df, aggregate = evaluate_model(
            model_name=model_name,
            df=df,
            args=args,
            logger=model_logger,
        )

        save_model_outputs(
            model_name=model_name,
            per_question_df=per_question_df,
            retrieval_df=retrieval_df,
            aggregate=aggregate,
            results_dir=results_dir,
            logger=model_logger,
        )

        all_aggregates[model_name] = aggregate

        model_elapsed = time.perf_counter() - t_model_start
        model_logger.info(
            "Pipeline finished for model '%s' in %.2f s.",
            model_name,
            model_elapsed,
        )

    # Cross-model comparison CSV
    if all_aggregates:
        build_comparison_csv(all_aggregates, results_dir, bootstrap_logger)

    pipeline_elapsed = time.perf_counter() - t_pipeline_start
    bootstrap_logger.info(
        "=" * 80
    )
    bootstrap_logger.info(
        "ALL EVALUATIONS COMPLETE  |  total time: %.2f s", pipeline_elapsed
    )
    bootstrap_logger.info("=" * 80)


if __name__ == "__main__":
    main()
