"""
backend/reguaz/evaluation/ragas_evaluator.py

RAGAS Evaluation Pipeline for ReguAZ.

Loads the enriched answers CSV (produced by ContextEnricher), builds a
RAGAS Dataset, runs the configured metrics using Llama-3.3-70b as the
Judge LLM (accessed through the NVIDIA NIM API), and saves results.

Metrics computed
----------------
- Faithfulness       : Is the answer grounded in the retrieved context?
- Answer Relevancy   : Does the answer address the question?
- Context Precision  : Are the retrieved chunks ranked well by relevance?
- Context Recall     : Do the retrieved chunks cover the ground truth?

Output
------
results/ragas/
    ragas_scores.csv    – per-question metric scores
    ragas_metrics.json  – aggregated mean scores
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from ragas.run_config import RunConfig

import pandas as pd
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from ragas import evaluate
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from backend.reguaz.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__, "ragas_evaluation.log")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_JUDGE_MODEL = "meta/llama-3.3-70b-instruct"
_NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"


# ---------------------------------------------------------------------------
# Helper: build the Judge LLM
# ---------------------------------------------------------------------------

def _build_judge_llm(
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> LangchainLLMWrapper:
    """Instantiate the Llama Judge LLM via NVIDIA NIM and wrap it for RAGAS.

    Parameters
    ----------
    model : str | None
        Model identifier (defaults to JUDGE_MODEL env var or the built-in
        ``_DEFAULT_JUDGE_MODEL``).
    temperature : float
        Sampling temperature.  Low values (0.0–0.2) give deterministic,
        factual output suitable for judge tasks.
    max_tokens : int
        Maximum tokens to generate per judge call.

    Returns
    -------
    LangchainLLMWrapper
        A RAGAS-compatible LLM wrapper ready to be passed to ``evaluate()``.

    Raises
    ------
    EnvironmentError
        If ``NVIDIA_API_KEY`` is not set in the environment.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NVIDIA_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )

    resolved_model = (
        model
        or os.getenv("JUDGE_MODEL")
        or _DEFAULT_JUDGE_MODEL
    )
    resolved_temperature = float(os.getenv("JUDGE_TEMPERATURE", str(temperature)))
    resolved_max_tokens = int(os.getenv("JUDGE_MAX_TOKENS", str(max_tokens)))

    logger.info(
        "RagasEvaluator: initialising Judge LLM '%s' via NVIDIA NIM …",
        resolved_model,
    )

    chat_llm = ChatNVIDIA(
        model=resolved_model,
        api_key=api_key,
        base_url=_NVIDIA_API_BASE,
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        # extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
    )

    logger.info("RagasEvaluator: Judge LLM ready.")
    return LangchainLLMWrapper(chat_llm)


# ---------------------------------------------------------------------------
# Main Evaluator class
# ---------------------------------------------------------------------------

class RagasEvaluator:
    """Runs RAGAS metrics on the enriched generation evaluation CSV.

    Parameters
    ----------
    judge_model : str | None
        NVIDIA NIM model identifier for the Judge LLM.
        Falls back to the ``JUDGE_MODEL`` env variable or the built-in
        default (``meta/llama-3.3-70b-instruct``).
    output_dir : str | Path
        Directory where RAGAS result files will be saved.
        Default: ``results/ragas/``.
    """

    def __init__(
        self,
        judge_model: str | None = None,
        output_dir: str | Path = "results/ragas",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._judge_llm = _build_judge_llm(model=judge_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        input_csv: str | Path,
        limit: int | None = None,
    ) -> dict[str, float]:
        """Run RAGAS evaluation on the enriched CSV.

        Parameters
        ----------
        input_csv : str | Path
            Path to the enriched CSV produced by
            :class:`~backend.reguaz.evaluation.context_enricher.ContextEnricher`.
            Must contain columns: ``Question``, ``Generated Answer``,
            ``Ground Truth``, ``Retrieved Contexts``.
        limit : int | None
            If set, evaluate only the first *limit* rows (useful for smoke
            tests).

        Returns
        -------
        dict[str, float]
            Aggregated mean scores for each metric.
        """
        input_path = Path(input_csv)
        if not input_path.exists():
            raise FileNotFoundError(
                f"RagasEvaluator: input CSV not found: '{input_path}'"
            )

        logger.info("=== RAGAS Evaluation Pipeline Start ===")
        logger.info("  Input CSV   : %s", input_path)
        logger.info("  Output Dir  : %s", self._output_dir)

        # ── 1. Load enriched CSV ─────────────────────────────────────────
        df = pd.read_csv(input_path)
        if limit:
            df = df.head(limit)
            logger.info("RagasEvaluator: limited to first %d rows.", limit)

        total = len(df)
        logger.info("RagasEvaluator: loaded %d rows for evaluation.", total)

        # ── 2. Build RAGAS dataset ────────────────────────────────────────
        data_dict = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
        skipped = 0

        for idx, row in df.iterrows():
            question   = str(row.get("Question", "") or "")
            answer     = str(row.get("Generated Answer", "") or "")
            ground_truth = str(row.get("Ground Truth", "") or "")
            raw_contexts = row.get("Retrieved Contexts")

            # Parse contexts JSON → list of plain strings
            try:
                ctx_objects: list[dict[str, Any]] = json.loads(raw_contexts)
                contexts: list[str] = [
                    c["text"] for c in ctx_objects if c.get("text")
                ]
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "RagasEvaluator: row %s has invalid Retrieved Contexts — skipping.",
                    row.get("Question ID", idx),
                )
                skipped += 1
                continue

            if not question or not answer or not contexts:
                logger.warning(
                    "RagasEvaluator: row %s missing required fields — skipping.",
                    row.get("Question ID", idx),
                )
                skipped += 1
                continue

            data_dict["question"].append(question)
            data_dict["answer"].append(answer)
            data_dict["contexts"].append(contexts)
            data_dict["ground_truth"].append(ground_truth)

        logger.info(
            "RagasEvaluator: built %d samples (%d skipped).", len(data_dict["question"]), skipped
        )

        if not data_dict["question"]:
            raise ValueError(
                "RagasEvaluator: no valid samples to evaluate. "
                "Check your input CSV for missing fields."
            )

        # ── 3. Define metrics ─────────────────────────────────────────────
        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            # ContextPrecision(),
            ContextRecall(),
        ]

        # ── 4. Run RAGAS evaluate ─────────────────────────────────────────
        logger.info("RagasEvaluator: running evaluation with Judge LLM …")
        t0 = time.perf_counter()

        logger.info("RagasEvaluator: loading local embedding model 'intfloat/multilingual-e5-large' …")
        eval_embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

        dataset = Dataset.from_dict(data_dict)

        run_config = RunConfig(
            max_workers=1,
            timeout=240,
            max_retries=3,
        )

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._judge_llm,
            embeddings=eval_embeddings,
            run_config=run_config,
        )

        elapsed = time.perf_counter() - t0
        logger.info("RagasEvaluator: evaluation completed in %.1f s.", elapsed)

        # ── 5. Persist per-question scores ────────────────────────────────
        scores_df = result.to_pandas()
        scores_path = self._output_dir / "ragas_scores.csv"
        scores_df.to_csv(scores_path, index=False)
        logger.info("RagasEvaluator: per-question scores saved to '%s'.", scores_path)

        # ── 6. Persist aggregated metrics ─────────────────────────────────
        aggregated: dict[str, float] = {
            col: float(scores_df[col].mean())
            for col in scores_df.columns
            if col not in ("question", "answer", "ground_truth", "contexts")
            and pd.api.types.is_numeric_dtype(scores_df[col])
        }

        metrics_path = self._output_dir / "ragas_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_questions": total,
                    "evaluated_questions": len(data_dict["question"]),
                    "skipped_questions": skipped,
                    "evaluation_time_seconds": round(elapsed, 2),
                    "metrics": aggregated,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
        logger.info("RagasEvaluator: aggregated metrics saved to '%s'.", metrics_path)

        logger.info(
            "=== RAGAS Evaluation Complete ===\n%s",
            "\n".join(f"  {k}: {v:.4f}" for k, v in aggregated.items()),
        )

        return aggregated
