import os
import sys
import argparse
import time
import logging
import itertools
from pathlib import Path
from typing import List, Dict, Any, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.reguaz.database.chroma import ChromaDBManager
from backend.reguaz.services.embeddings.embedding_reader import EmbeddingReader
from backend.reguaz.services.ingestion.chunk_reader import ChunkReader


def setup_ingestion_logger() -> logging.Logger:
    """
    Standalone logger for the ChromaDB ingestion pipeline.
    Writes to console and to logs/chroma_ingestion_logs.log.
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "chroma_ingestion_logs.log"

    logger = logging.getLogger("chroma_ingestion")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if main() is ever called twice
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def ingest_model(
    model_name: str,
    chunks_lookup: Dict[str, Dict[str, Any]],
    embeddings_dir: str,
    chroma_dir: str,
    batch_size: int,
    reset: bool,
    logger
) -> Dict[str, Any]:
    """
    Ingests embeddings for a single model and performs pre-insertion validation.
    Returns statistics for post-ingestion verification.
    """
    logger.info(f"--- Starting ingestion for model: {model_name} ---")
    
    collection_name = f"reguaz_{model_name.replace('-', '_')}"
    chroma_manager = ChromaDBManager(persist_directory=chroma_dir)
    
    record_iterator = EmbeddingReader.read_all(embeddings_dir, model_name)
    try:
        first_record = next(record_iterator)
    except StopIteration:
        logger.warning(f"No embeddings found for model {model_name}.")
        return {
            "model": model_name,
            "collection_name": collection_name,
            "total_attempted": 0,
            "total_inserted_tracked": 0,
            "actual_collection_count": 0,
            "total_orphans": 0,
            "total_duplicates": 0,
            "total_dimension_mismatch": 0,
            "expected_dimension": None,
            "orphan_ids": [],
            "seen_ids": set(),
            "inserted_ids": set()
        }

    first_embedding = first_record.get("embedding")
    expected_dimension = len(first_embedding) if first_embedding else None
    
    collection_metadata = {
        "hnsw:space": "cosine",
        "embedding_model": model_name,
        "distance_metric": "cosine"
    }
    
    if expected_dimension is not None:
        collection_metadata["embedding_dimension"] = expected_dimension
        logger.info(f"Detected embedding dimension: {expected_dimension} for model {model_name}")

    if reset:
        logger.info(f"Reset flag is set. Resetting collection: {collection_name}")
        chroma_manager.reset_collection(collection_name, metadata=collection_metadata)
    else:
        # Pre-create collection to set metadata
        chroma_manager.get_or_create_collection(collection_name, metadata=collection_metadata)
    
    total_records = 0
    total_orphans = 0
    total_duplicates = 0
    total_dimension_mismatch = 0
    
    seen_ids: Set[str] = set()
    inserted_ids: Set[str] = set()
    orphan_ids: List[str] = []
    
    current_batch_chunks = []
    current_batch_embeddings = []
    current_batch_ids = []
    
    def flush_batch():
        nonlocal total_records
        if current_batch_chunks:
            chroma_manager.insert_chunks(
                collection_name=collection_name,
                chunks=current_batch_chunks,
                embeddings=current_batch_embeddings
            )
            inserted_ids.update(current_batch_ids)
            total_records += len(current_batch_ids)
            
            current_batch_chunks.clear()
            current_batch_embeddings.clear()
            current_batch_ids.clear()

    logger.info(f"Reading embeddings for {model_name}...")
    for record in itertools.chain([first_record], record_iterator):
        chunk_id = record.get("chunk_id")
        embedding = record.get("embedding")
        
        # 1. Check for duplicates
        if chunk_id in seen_ids:
            logger.warning(f"Duplicate chunk_id encountered: {chunk_id}. Skipping.")
            total_duplicates += 1
            continue
            
        seen_ids.add(chunk_id)
        
        # 2. Check for orphan embeddings
        chunk_metadata = chunks_lookup.get(chunk_id)
        if not chunk_metadata:
            logger.warning(f"Orphan embedding encountered: {chunk_id} has no matching chunk metadata. Skipping.")
            orphan_ids.append(chunk_id)
            total_orphans += 1
            continue
            
        # 3. Check embedding dimension
        if embedding:
            dim = len(embedding)
            if expected_dimension is None:
                expected_dimension = dim
                logger.info(f"Detected embedding dimension: {expected_dimension} for model {model_name} on later record.")
            elif dim != expected_dimension:
                logger.warning(f"Dimension mismatch for {chunk_id}: expected {expected_dimension}, got {dim}. Skipping.")
                total_dimension_mismatch += 1
                continue
        
        current_batch_chunks.append(chunk_metadata)
        current_batch_embeddings.append(embedding)
        current_batch_ids.append(chunk_id)
        
        if len(current_batch_chunks) >= batch_size:
            flush_batch()
            
    # Flush remaining
    flush_batch()
    
    actual_count = chroma_manager.get_collection_count(collection_name)
    
    logger.info(f"--- Finished ingestion for {model_name} ---")
    
    return {
        "model": model_name,
        "collection_name": collection_name,
        "total_attempted": total_records + total_orphans + total_duplicates + total_dimension_mismatch,
        "total_inserted_tracked": total_records,
        "actual_collection_count": actual_count,
        "total_orphans": total_orphans,
        "total_duplicates": total_duplicates,
        "total_dimension_mismatch": total_dimension_mismatch,
        "expected_dimension": expected_dimension,
        "orphan_ids": orphan_ids,
        "seen_ids": seen_ids,
        "inserted_ids": inserted_ids
    }

def main():
    parser = argparse.ArgumentParser(description="ChromaDB Ingestion Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Embedding model (e5, bge_m3) or 'all'")
    parser.add_argument("--chunks-dir", type=str, default="data/processed/chunks", help="Directory containing chunk JSONL files")
    parser.add_argument("--embeddings-dir", type=str, default="data/processed/embeddings", help="Directory containing embedding JSONL files")
    parser.add_argument("--chroma-dir", type=str, default="data/chroma", help="Directory to store ChromaDB data")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for insertion")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the collection before ingesting")
    
    args = parser.parse_args()
    
    logger = setup_ingestion_logger()
    
    logger.info("=" * 80)
    logger.info("ChromaDB Ingestion Pipeline Started")
    logger.info(f"Target model(s) : {args.model}")
    logger.info(f"Reset collection: {args.reset}")
    logger.info("=" * 80)
    
    pipeline_start = time.perf_counter()
    
    # 1. Build chunk lookup
    logger.info("Building chunk metadata lookup...")
    chunks_lookup = ChunkReader.build_lookup(args.chunks_dir)
    logger.info(f"Loaded {len(chunks_lookup)} chunks into lookup.")
    
    models_to_process = ["e5", "bge_m3"] if args.model.lower() == "all" else [args.model]
    
    stats_list = []
    
    # 2. Process models
    for model_name in models_to_process:
        stats = ingest_model(
            model_name=model_name,
            chunks_lookup=chunks_lookup,
            embeddings_dir=args.embeddings_dir,
            chroma_dir=args.chroma_dir,
            batch_size=args.batch_size,
            reset=args.reset,
            logger=logger
        )
        stats_list.append(stats)
        
    # 3. Post-ingestion Verification
    logger.info("=" * 80)
    logger.info("POST-INGESTION VERIFICATION")
    logger.info("=" * 80)
    
    all_passed = True
    
    for stats in stats_list:
        model = stats["model"]
        logger.info(f"[{model}] Vector Count Check: Tracked {stats['total_inserted_tracked']} vs Actual {stats['actual_collection_count']}")
        if stats["total_inserted_tracked"] == stats["actual_collection_count"]:
            logger.info(f"[{model}] Vector Count: PASS")
        else:
            logger.warning(f"[{model}] Vector Count: FAIL - Mismatch!")
            all_passed = False
            
        if stats["total_duplicates"] == 0:
            logger.info(f"[{model}] Duplicate Check: PASS (0 duplicates)")
        else:
            logger.warning(f"[{model}] Duplicate Check: WARN ({stats['total_duplicates']} duplicates found)")
            
        if stats["total_orphans"] == 0:
            logger.info(f"[{model}] Orphan Check: PASS (0 orphans)")
        else:
            logger.warning(f"[{model}] Orphan Check: WARN ({stats['total_orphans']} orphans skipped)")
            
        if stats["total_dimension_mismatch"] == 0:
            logger.info(f"[{model}] Dimension Consistency: PASS (All {stats['expected_dimension']})")
        else:
            logger.warning(f"[{model}] Dimension Consistency: WARN ({stats['total_dimension_mismatch']} mismatched)")
            
    # Cross-collection parity
    if len(stats_list) > 1:
        logger.info("[Cross-Collection] Checking parity across models...")
        ids_0 = stats_list[0]["inserted_ids"]
        ids_1 = stats_list[1]["inserted_ids"]
        
        if len(ids_0) == len(ids_1) and ids_0 == ids_1:
            logger.info("[Cross-Collection] Parity Check: PASS (Identical chunk_ids across collections)")
        else:
            logger.warning(f"[Cross-Collection] Parity Check: FAIL - Models have different chunk_ids. {stats_list[0]['model']}: {len(ids_0)}, {stats_list[1]['model']}: {len(ids_1)}")
            all_passed = False
            
    pipeline_elapsed = time.perf_counter() - pipeline_start
    
    logger.info("=" * 80)
    if all_passed:
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    else:
        logger.warning("PIPELINE COMPLETED WITH WARNINGS/FAILURES")
    logger.info(f"Total processing time: {pipeline_elapsed:.2f} sec")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()