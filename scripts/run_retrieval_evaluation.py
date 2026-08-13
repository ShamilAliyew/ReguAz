"""
scripts/run_retrieval_evaluation.py

Retrieval evaluation pipeline for ReguAZ.

Loads the gold evaluation dataset, generates query embeddings for each model
(e5, bge_m3), retrieves results from the corresponding ChromaDB collection,
computes ranking metrics, and saves per-question CSV files plus a summary
comparison CSV.

Usage
-----
    python scripts/run_retrieval_evaluation.py [--top-k 10] [--chroma-dir data/chroma]

Allowed to create:
    results/e5_results.csv
    results/bge_m3_results.csv
    results/comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: make the project root importable from the scripts/ directory.
# All existing scripts use this exact pattern (see run_chroma_ingestion.py).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.reguaz.retrieval.retriever import ChromaRetriever
from backend.reguaz.services.embeddings.embedding_factory import EmbeddingFactory
from backend.reguaz.utils.logger import setup_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS: list[str] = ["e5", "bge_m3"]

COLLECTION_MAP: dict[str, str] = {
    "e5": "reguaz_e5",
    "bge_m3": "reguaz_bge_m3",
}

# E5 is an asymmetric model: passages are stored with "passage: " prefix,
# queries must use "query: " prefix at inference time.
QUERY_PREFIX: dict[str, str] = {
    "e5": "query: ",
    "bge_m3": "",
}

DEFAULT_DATASET_PATH = Path("data/evaluation/gold_dataset_for_embedding_excel.xlsx")
DEFAULT_CHROMA_DIR = "data/chroma"
DEFAULT_RESULTS_DIR = str(config.RESULTS_PATH)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(xlsx_path: Path, logger) -> pd.DataFrame:
    """
    Load and normalise the gold evaluation dataset.

    Relevant chunk IDs may be a single value or comma-separated values.
    All IDs are stripped of whitespace.  Duplicate IDs within the same row
    are silently dropped.
    """
    logger.info("Loading evaluation dataset from: %s", xlsx_path)
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
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    logger.info("Loaded %d evaluation questions.", len(df))
    return df


def parse_chunk_ids(raw_value: Any) -> list[str]:
    """
    Parse the relevant_chunk_ids cell into a deduplicated list of stripped strings.

    Handles:
    - Single string IDs
    - Comma-separated string IDs
    - Numeric values (cast to string)
    - NaN / None (returns empty list)
    """
    if pd.isna(raw_value):
        return []

    raw_str = str(raw_value)
    parts = [part.strip() for part in raw_str.split("|")]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            unique.append(part)
    return unique


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    relevant_ids: list[str],
    retrieved_ids: list[str],
    top_k: int,
) -> dict[str, Any]:
    """
    Compute retrieval metrics for a single question.

    Parameters
    ----------
    relevant_ids   : Ground-truth relevant chunk IDs (already deduplicated).
    retrieved_ids  : Ordered list of retrieved IDs (position 0 = rank 1).
    top_k          : Maximum number of retrieved results considered.

    Returns
    -------
    dict containing:
        rank_of_first_relevant, recall@1/3/5/10, mrr, hit@1/3/5/10
    """
    import unicodedata
    relevant_set = {unicodedata.normalize("NFC", rid) for rid in relevant_ids}
    retrieved_ids = retrieved_ids[:top_k]

    # Rank of first relevant result (1-indexed; None if not found in top_k)
    rank_of_first: int | None = None
    for rank, rid in enumerate(retrieved_ids, start=1):
        if unicodedata.normalize("NFC", rid) in relevant_set:
            rank_of_first = rank
            break

    def recall_at(k: int) -> float:
        if not relevant_set:
            return 0.0
        hits = sum(1 for rid in retrieved_ids[:k] if unicodedata.normalize("NFC", rid) in relevant_set)
        return hits / len(relevant_set)

    def hit_at(k: int) -> int:
        return int(any(unicodedata.normalize("NFC", rid) in relevant_set for rid in retrieved_ids[:k]))

    mrr = (1.0 / rank_of_first) if rank_of_first is not None else 0.0

    return {
        "rank_of_first_relevant": rank_of_first if rank_of_first is not None else -1,
        "recall@1": recall_at(1),
        "recall@3": recall_at(3),
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "mrr": mrr,
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        "hit@5": hit_at(5),
        "hit@10": hit_at(10),
    }


# ---------------------------------------------------------------------------
# Per-model evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    df: pd.DataFrame,
    chroma_dir: str,
    top_k: int,
    logger,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Run the full evaluation pipeline for one embedding model.

    Steps
    -----
    1. Load the embedding service via EmbeddingFactory.
    2. Open the ChromaDB collection via ChromaRetriever.
    3. For each question: generate embedding → search → compute metrics.
    4. Return per-question DataFrame and aggregate summary dict.
    """
    collection_name = COLLECTION_MAP[model_name]
    query_prefix = QUERY_PREFIX[model_name]

    logger.info("=" * 60)
    logger.info("Evaluating model: %s  |  collection: %s", model_name, collection_name)
    logger.info("=" * 60)

    # 1. Load embedding service
    logger.info("Loading embedding service for model '%s'…", model_name)
    embedding_service = EmbeddingFactory.get_service(model_name)

    # 2. Open Chroma collection — validates existence, count, and metadata.
    try:
        retriever = ChromaRetriever(
            persist_directory=chroma_dir,
            collection_name=collection_name,
        )
    except Exception as exc:
        logger.error(
            "Cannot open collection '%s' for model '%s': %s — skipping model.",
            collection_name,
            model_name,
            exc,
        )
        empty_df = pd.DataFrame()
        empty_aggregate: dict[str, float] = {
            "Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "Recall@10": 0.0,
            "MRR": 0.0,
            "HitRate@1": 0.0, "HitRate@3": 0.0, "HitRate@5": 0.0, "HitRate@10": 0.0,
            "AverageRank": 0.0,
        }
        return empty_df, empty_aggregate

    rows: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        question: str = str(row["question"])
        relevant_ids: list[str] = parse_chunk_ids(row["relevant_chunk_ids"])

        if not relevant_ids:
            logger.warning(
                "Row %d ('%s'): no relevant_chunk_ids found — skipping.", idx, question[:60]
            )
            continue

        # 3. Generate query embedding
        query_text = query_prefix + question
        try:
            raw_embedding = embedding_service.embed_text(query_text)
            # embed_text() may return numpy.ndarray, torch.Tensor, or list.
            # Coerce to list[float] before passing to ChromaDB.
            if isinstance(raw_embedding, list):
                query_embedding: list[float] = raw_embedding
            else:
                query_embedding = [float(v) for v in raw_embedding]
        except Exception as exc:
            logger.warning(
                "Row %d: embed_text failed for model '%s': %s — skipping.", idx, model_name, exc
            )
            continue

        # 4. Search ChromaDB
        try:
            search_results = retriever.search(query_embedding=query_embedding, top_k=top_k)
        except Exception as exc:
            logger.warning(
                "Row %d: ChromaDB query failed: %s — skipping.", idx, exc
            )
            continue

        retrieved_ids: list[str] = search_results["ids"]

        if idx < 5:
            overlap = set(relevant_ids) & set(retrieved_ids)

            logger.info("=" * 80)
            logger.info("QUESTION %d", idx + 1)
            logger.info("Question: %s", question)
            logger.info("Expected chunk ids: %s", relevant_ids)
            logger.info("Retrieved ids:")

            for rank, rid in enumerate(retrieved_ids, start=1):
                logger.info("  %2d. %s", rank, rid)

            logger.info("Overlap: %s", overlap if overlap else "NONE")
            logger.info("=" * 80)
    

        if idx < 5:
            overlap = set(relevant_ids) & set(retrieved_ids)
            logger.info("Overlap: %s", overlap if overlap else "NONE")
        # 5. Compute metrics
        metrics = compute_metrics(
            relevant_ids=relevant_ids,
            retrieved_ids=retrieved_ids,
            top_k=top_k,
        )

        rows.append(
            {
                "question": question,
                "expected_chunk_ids": ",".join(relevant_ids),
                "expected_document_ids": str(row.get("expected_document_ids", "")),
                "retrieved_top10": ",".join(retrieved_ids[:10]),
                "rank_of_first_relevant": metrics["rank_of_first_relevant"],
                "recall@1": metrics["recall@1"],
                "recall@3": metrics["recall@3"],
                "recall@5": metrics["recall@5"],
                "recall@10": metrics["recall@10"],
                "mrr": metrics["mrr"],
                "hit@1": metrics["hit@1"],
                "hit@3": metrics["hit@3"],
                "hit@5": metrics["hit@5"],
                "hit@10": metrics["hit@10"],
            }
        )

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        logger.warning("No results produced for model '%s'.", model_name)
        aggregate: dict[str, float] = {
            "Recall@1": 0.0,
            "Recall@3": 0.0,
            "Recall@5": 0.0,
            "Recall@10": 0.0,
            "MRR": 0.0,
            "HitRate@1": 0.0,
            "HitRate@3": 0.0,
            "HitRate@5": 0.0,
            "HitRate@10": 0.0,
            "AverageRank": 0.0,
        }
        return results_df, aggregate

    # Compute aggregate metrics (exclude -1 sentinel from average rank)
    valid_ranks = results_df.loc[
        results_df["rank_of_first_relevant"] > 0, "rank_of_first_relevant"
    ]
    average_rank = float(valid_ranks.mean()) if not valid_ranks.empty else 0.0

    aggregate = {
        "Recall@1": float(results_df["recall@1"].mean()),
        "Recall@3": float(results_df["recall@3"].mean()),
        "Recall@5": float(results_df["recall@5"].mean()),
        "Recall@10": float(results_df["recall@10"].mean()),
        "MRR": float(results_df["mrr"].mean()),
        "HitRate@1": float(results_df["hit@1"].mean()),
        "HitRate@3": float(results_df["hit@3"].mean()),
        "HitRate@5": float(results_df["hit@5"].mean()),
        "HitRate@10": float(results_df["hit@10"].mean()),
        "AverageRank": average_rank,
    }

    return results_df, aggregate


# ---------------------------------------------------------------------------
# Console printer
# ---------------------------------------------------------------------------

def print_summary(model_name: str, aggregate: dict[str, float]) -> None:
    """Print a formatted summary block to the console."""
    print("=" * 40)
    print(f"MODEL: {model_name}")
    print(f"  Recall@1     : {aggregate['Recall@1']:.4f}")
    print(f"  Recall@3     : {aggregate['Recall@3']:.4f}")
    print(f"  Recall@5     : {aggregate['Recall@5']:.4f}")
    print(f"  Recall@10    : {aggregate['Recall@10']:.4f}")
    print(f"  MRR          : {aggregate['MRR']:.4f}")
    print(f"  HitRate@1    : {aggregate['HitRate@1']:.4f}")
    print(f"  HitRate@3    : {aggregate['HitRate@3']:.4f}")
    print(f"  HitRate@5    : {aggregate['HitRate@5']:.4f}")
    print(f"  HitRate@10   : {aggregate['HitRate@10']:.4f}")
    print(f"  AverageRank  : {aggregate['AverageRank']:.4f}")
    print("=" * 40)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReguAZ Retrieval Evaluation Pipeline"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to retrieve per query (default: 10).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help=f"Path to the evaluation dataset XLSX file (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=DEFAULT_CHROMA_DIR,
        help=f"Path to the ChromaDB persistence directory (default: {DEFAULT_CHROMA_DIR}).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory where result CSV files will be saved (default: {DEFAULT_RESULTS_DIR}).",
    )
    args = parser.parse_args()

    logger = setup_logger("retrieval_evaluation")

    logger.info("=" * 80)
    logger.info("ReguAZ Retrieval Evaluation Pipeline")
    logger.info("top_k        : %d", args.top_k)
    logger.info("dataset      : %s", args.dataset)
    logger.info("chroma_dir   : %s", args.chroma_dir)
    logger.info("results_dir  : %s", args.results_dir)
    logger.info("=" * 80)

    # Resolve paths with pathlib
    dataset_path = Path(args.dataset)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        logger.error("Dataset file not found: %s", dataset_path)
        sys.exit(1)

    # Load dataset once
    df = load_dataset(dataset_path, logger)

    comparison_rows: list[dict[str, Any]] = []

    for model_name in MODELS:
        results_df, aggregate = evaluate_model(
            model_name=model_name,
            df=df,
            chroma_dir=args.chroma_dir,
            top_k=args.top_k,
            logger=logger,
        )

        # Save per-question CSV
        csv_filename = f"{model_name}_results.csv"
        csv_path = results_dir / csv_filename
        results_df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info("Saved per-question results → %s", csv_path)

        # Print console summary
        print_summary(model_name, aggregate)

        # Accumulate comparison row
        comparison_rows.append({"model": model_name, **aggregate})

    # Save comparison CSV
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = results_dir / "comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")
    logger.info("Saved comparison summary  → %s", comparison_path)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
