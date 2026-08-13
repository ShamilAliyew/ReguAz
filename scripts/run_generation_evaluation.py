#!/usr/bin/env python3
"""
Run LLM generation evaluation over the ReguAZ pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.reguaz import config
from backend.reguaz.evaluation.generation_evaluator import GenerationEvaluator
from backend.reguaz.retrieval.hybrid_qdrant import HybridQdrantRetriever
from backend.reguaz.services.chunks.chunk_reader import ChunkReader
from backend.reguaz.services.generation.generation_pipeline import GenerationPipeline
from backend.reguaz.services.generation.llm_factory import LLMFactory
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "run_generation_evaluation.log")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM generation on the gold dataset.")
    parser.add_argument(
        "--model-type",
        type=str,
        default=config.DEFAULT_LLM_TYPE,
        help=f"Type of LLM to use. Default: {config.DEFAULT_LLM_TYPE}"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(config.GOLD_DATASET_PATH),
        help="Path to the evaluation dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results. Defaults to results/generation/<model_type>/"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.DEFAULT_EVALUATION_TOP_K,
        help=f"Number of top chunks to retrieve for generation context. Default: {config.DEFAULT_EVALUATION_TOP_K}"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions to evaluate."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    output_dir = args.output_dir or str(config.RESULTS_PATH / "generation" / args.model_type)
    output_path = Path(PROJECT_ROOT) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Starting Generation Evaluation ===")
    logger.info("Configuration:")
    logger.info("  Embedding Model: bge_m3")
    logger.info("  Reranker Model: BAAI/bge-reranker-v2-m3")
    logger.info("  LLM Model Type: %s", args.model_type)
    logger.info("  Dataset Path: %s", args.dataset)
    logger.info("  Output Directory: %s", output_path)
    logger.info("  Context Top K: %d", args.top_k)
    logger.info("  Question Limit: %s", args.limit if args.limit else "All")
    
    # 1. Initialize Retriever
    logger.info("Initializing HybridQdrantRetriever...")
    try:
        retriever = HybridQdrantRetriever(
            model_name="bge_m3",
            qdrant_dir=str(config.QDRANT_PATH),
            chunks_dir=str(config.CHUNKS_DIR),
            top_k_semantic=config.SEMANTIC_TOP_K,
            top_k_bm25=config.BM25_TOP_K,
            rerank_top_k=15,
            final_top_k=args.top_k,
            rrf_k=config.RRF_K,
            reranker_model="BAAI/bge-reranker-v2-m3"
        )
    except Exception as exc:
        logger.critical("Failed to initialize retriever: %s", exc)
        sys.exit(1)
        
    # 2. Build Chunk Lookup for Metadata Hydration
    logger.info("Building Chunk Metadata Lookup...")
    try:
        chunk_reader = ChunkReader()
        chunk_lookup = chunk_reader.build_lookup(config.CHUNKS_DIR)
    except Exception as exc:
        logger.critical("Failed to build chunk lookup: %s", exc)
        sys.exit(1)
        
    # 3. Initialize LLM
    logger.info("Initializing %s LLM...", args.model_type)
    try:
        llm = LLMFactory.create(model_type=args.model_type)
    except Exception as exc:
        logger.critical("Failed to initialize LLM: %s", exc)
        sys.exit(1)
        
    # 4. Initialize Generation Pipeline
    logger.info("Initializing Generation Pipeline...")
    pipeline = GenerationPipeline(
        retriever=retriever,
        chunk_lookup=chunk_lookup,
        llm=llm,
        top_k=args.top_k
    )
    
    # 5. Initialize Evaluator
    evaluator = GenerationEvaluator(
        pipeline=pipeline,
        output_dir=output_path
    )
    
    # 6. Run Evaluation
    logger.info("Starting Evaluation Loop...")
    try:
        evaluator.evaluate(dataset_path=args.dataset, limit=args.limit)
    except Exception as exc:
        logger.critical("Evaluation failed critically: %s", exc)
        sys.exit(1)
        
    logger.info("=== Generation Evaluation Completed Successfully ===")

if __name__ == "__main__":
    main()
