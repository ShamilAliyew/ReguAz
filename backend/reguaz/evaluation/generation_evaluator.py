"""
backend/reguaz/evaluation/generation_evaluator.py
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from backend.reguaz.services.generation.generation_pipeline import GenerationPipeline
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "generation_evaluation.log")

class GenerationEvaluator:
    """
    Evaluates the Generation Pipeline against a gold dataset.

    Generates answers for each question and saves them for downstream
    metrics computation or LLM-as-a-judge evaluation.
    """

    def __init__(self, pipeline: GenerationPipeline, output_dir: str | Path) -> None:
        """
        Parameters
        ----------
        pipeline : GenerationPipeline
            The initialized generation pipeline to evaluate.
        output_dir : str | Path
            Directory where results (answers.csv, metrics.json) will be saved.
        """
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self, dataset_path: str | Path, limit: int | None = None) -> None:
        """
        Run the evaluation over the dataset.

        Parameters
        ----------
        dataset_path : str | Path
            Path to the Excel dataset (e.g. gold_dataset_for_llm_generation.xlsx).
        limit : int | None
            Maximum number of questions to evaluate.
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error("Dataset not found at '%s'", dataset_path)
            raise FileNotFoundError(f"Dataset not found at '{dataset_path}'")
            
        logger.info("Loading dataset from '%s'", dataset_path)
        df = pd.read_excel(dataset_path)
        
        if limit is not None and limit > 0:
            df = df.head(limit)
        
        results = []
        successful_gens = 0
        failed_gens = 0
        
        # Timing accumulators
        total_retrieval_time = 0.0
        total_prompt_time = 0.0
        total_generation_time = 0.0
        total_pipeline_time = 0.0
        
        logger.info("Starting evaluation over %d questions...", len(df))
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Generation"):
            question_id = row.get("Question ID")
            question = row.get("Question")
            ground_truth = row.get("Ground Truth Answer")
            category = row.get("Category")
            
            if pd.isna(question) or not str(question).strip():
                logger.warning("Skipping row %d due to empty question.", idx)
                continue
                
            question_str = str(question)
            
            try:
                result = self.pipeline.generate(question_str)
                metrics = result.get("metrics", {})
                
                retrieved_chunks = [s.get("chunk_id") for s in result.get("sources", [])]
                retrieved_docs = [s.get("title") for s in result.get("sources", [])]
                
                results.append({
                    "Question ID": question_id,
                    "Category": category,
                    "Question": question_str,
                    "Ground Truth": ground_truth,
                    "Generated Answer": result.get("answer"),
                    "Retrieved Chunk IDs": json.dumps(retrieved_chunks, ensure_ascii=False),
                    "Retrieved Documents": json.dumps(retrieved_docs, ensure_ascii=False),
                    "Retrieval Time": metrics.get("retrieval_time"),
                    "Prompt Time": metrics.get("prompt_build_time"),
                    "Generation Time": metrics.get("generation_time"),
                    "Total Time": metrics.get("total_time"),
                    "Error": None
                })
                
                successful_gens += 1
                total_retrieval_time += metrics.get("retrieval_time", 0.0)
                total_prompt_time += metrics.get("prompt_build_time", 0.0)
                total_generation_time += metrics.get("generation_time", 0.0)
                total_pipeline_time += metrics.get("total_time", 0.0)
                
            except Exception as exc:
                logger.exception("Failed to generate answer for Question ID %s", question_id)
                results.append({
                    "Question ID": question_id,
                    "Category": category,
                    "Question": question_str,
                    "Ground Truth": ground_truth,
                    "Generated Answer": None,
                    "Retrieved Chunk IDs": None,
                    "Retrieved Documents": None,
                    "Retrieval Time": None,
                    "Prompt Time": None,
                    "Generation Time": None,
                    "Total Time": None,
                    "Error": str(exc)
                })
                failed_gens += 1
                
        self._save_results(results, len(df), successful_gens, failed_gens, 
                           total_retrieval_time, total_prompt_time, total_generation_time, total_pipeline_time)
        
    def _save_results(self, results: list[dict], total_count: int, successful_gens: int, failed_gens: int,
                      total_retrieval_time: float, total_prompt_time: float, total_generation_time: float, total_pipeline_time: float) -> None:
        """Save answers and summary metrics."""
        # Save answers
        out_csv = self.output_dir / "answers.csv"
        results_df = pd.DataFrame(results)
        results_df.to_csv(out_csv, index=False)
        logger.info("Saved generated answers to '%s'", out_csv)
        
        # Compute summary metrics
        success_rate = successful_gens / total_count if total_count > 0 else 0.0
        avg_retrieval_time = total_retrieval_time / successful_gens if successful_gens > 0 else 0.0
        avg_prompt_time = total_prompt_time / successful_gens if successful_gens > 0 else 0.0
        avg_generation_time = total_generation_time / successful_gens if successful_gens > 0 else 0.0
        avg_total_time = total_pipeline_time / successful_gens if successful_gens > 0 else 0.0
        
        metrics = {
            "total_questions": total_count,
            "successful_generations": successful_gens,
            "failed_generations": failed_gens,
            "success_rate": success_rate,
            "average_retrieval_time": avg_retrieval_time,
            "average_prompt_time": avg_prompt_time,
            "average_generation_time": avg_generation_time,
            "average_total_time": avg_total_time
        }
        
        out_json = self.output_dir / "metrics.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
            
        logger.info("Saved basic metrics to '%s'", out_json)
        logger.info("Evaluation completed. Success rate: %.2f%%", success_rate * 100)
