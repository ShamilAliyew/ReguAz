# import os
# import sys

# # Mühit dəyişənlərinə əsas qovluğu əlavə edirik ki, 'backend' modulu tapılsın
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import glob
# import json
# import argparse
# import logging
# from typing import List, Dict, Any

# from backend.reguaz.services.embeddings.embedding_factory import EmbeddingFactory
# from backend.reguaz.database.chroma import ChromaDBManager

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def load_jsonl_chunks(file_path: str) -> List[Dict[str, Any]]:
#     """Loads chunks from a JSONL file."""
#     chunks = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 chunks.append(json.loads(line))
#     return chunks

# def main():
#     parser = argparse.ArgumentParser(description="Run embedding pipeline and ingest into ChromaDB.")
#     parser.add_argument("--model", type=str, required=True, 
#                         help="Embedding model (bge_m3, e5, jina_v3, qwen)")
#     parser.add_argument("--chunks-dir", type=str, default="data/processed/chunks", 
#                         help="Directory containing JSONL chunk files")
#     parser.add_argument("--chroma-dir", type=str, default="data/chroma", 
#                         help="Directory to store ChromaDB data")
#     parser.add_argument("--batch-size", type=int, default=100, 
#                         help="Batch size for embedding generation")
    
#     args = parser.parse_args()
    
#     logger.info(f"Starting embedding pipeline with model: {args.model}")
    
#     # Initialize embedding service
#     try:
#         embedding_service = EmbeddingFactory.get_service(args.model)
#     except Exception as e:
#         logger.error(f"Failed to initialize embedding service: {e}")
#         return

#     # Initialize ChromaDB manager
#     chroma_manager = ChromaDBManager(persist_directory=args.chroma_dir)
#     collection_name = f"reguaz_{args.model.lower().replace('-', '_')}"
    
#     # Find all JSONL files
#     jsonl_files = glob.glob(os.path.join(args.chunks_dir, "**/*.jsonl"), recursive=True)
#     if not jsonl_files:
#         logger.warning(f"No JSONL files found in {args.chunks_dir}")
#         return
        
#     logger.info(f"Found {len(jsonl_files)} JSONL files to process.")
    
#     total_chunks_processed = 0
    
#     for file_path in jsonl_files:
#         logger.info(f"Processing file: {file_path}")
#         chunks = load_jsonl_chunks(file_path)
        
#         if not chunks:
#             continue
            
#         # Process chunks in batches
#         for i in range(0, len(chunks), args.batch_size):
#             batch_chunks = chunks[i:i + args.batch_size]
            
#             # Extract text for embedding
#             # Note: Depending on the chunk structure, the key might be 'text' or 'content'
#             texts_to_embed = []
#             for chunk in batch_chunks:
#                 text = chunk.get("text") or chunk.get("content") or ""
#                 texts_to_embed.append(text)
            
#             # Generate embeddings
#             try:
#                 logger.info(f"Generating embeddings for batch of {len(batch_chunks)} chunks...")
#                 embeddings = embedding_service.embed_batch(texts_to_embed)
                
#                 # Insert into ChromaDB
#                 chroma_manager.insert_chunks(
#                     collection_name=collection_name,
#                     chunks=batch_chunks,
#                     embeddings=embeddings
#                 )
                
#                 total_chunks_processed += len(batch_chunks)
#             except Exception as e:
#                 logger.error(f"Error processing batch in {file_path}: {e}")
                
#     logger.info(f"Embedding pipeline completed successfully. Total chunks processed: {total_chunks_processed}")

# if __name__ == "__main__":
#     # Workaround for Python path issues if run directly from scripts folder
#     import sys
#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#     main()


import os
import sys
import glob
import json
import time
import argparse
from typing import List, Dict, Any

# Project root əlavə edilir ki backend modulu tapılsın
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)

from backend.reguaz.services.embeddings.embedding_factory import EmbeddingFactory
from backend.reguaz.services.embeddings.embedding_writer import EmbeddingWriter
from backend.reguaz.utils.logger import setup_logger


def load_jsonl_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Load chunks from a JSONL file.
    """
    chunks = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks


def main():

    parser = argparse.ArgumentParser(
        description="Generate embeddings for chunk files."
    )

    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Embedding model (bge_m3, e5, jina_v3, qwen)"
    )

    parser.add_argument(
        "--chunks-dir",
        default="data/processed/chunks",
        type=str,
        help="Directory containing chunk JSONL files."
    )

    parser.add_argument(
        "--embeddings-dir",
        default="data/processed/embeddings",
        type=str,
        help="Directory where embeddings will be stored."
    )

    parser.add_argument(
        "--batch-size",
        default=100,
        type=int,
        help="Embedding batch size."
    )

    args = parser.parse_args()

    logger = setup_logger(args.model)

    pipeline_start = time.perf_counter()

    logger.info("=" * 80)
    logger.info("Embedding pipeline started")
    logger.info(f"Embedding model : {args.model}")
    logger.info(f"Chunks directory: {args.chunks_dir}")
    logger.info(f"Output directory: {args.embeddings_dir}")
    logger.info("=" * 80)

    # Initialize embedding model
    try:
        embedding_service = EmbeddingFactory.get_service(args.model)
    except Exception:
        logger.exception("Failed to initialize embedding model.")
        return

    # Discover chunk files
    jsonl_files = glob.glob(
        os.path.join(args.chunks_dir, "**/*.jsonl"),
        recursive=True,
    )

    if not jsonl_files:
        logger.warning(
            f"No chunk files found under {args.chunks_dir}"
        )
        return

    logger.info(f"Discovered {len(jsonl_files)} JSONL files.")

    total_chunks_processed = 0

    for file_path in jsonl_files:

        file_start = time.perf_counter()

        logger.info("-" * 80)
        logger.info(f"Processing file: {file_path}")

        try:
            chunks = load_jsonl_chunks(file_path)

            if not chunks:
                logger.warning(
                    f"Skipping empty file: {file_path}"
                )
                continue

            output_file = EmbeddingWriter.get_output_path(
                chunk_file=file_path,
                chunks_root=args.chunks_dir,
                embeddings_root=args.embeddings_dir,
                model_name=args.model,
            )

            logger.info(f"Output file: {output_file}")

            total_batches = (
                len(chunks) + args.batch_size - 1
            ) // args.batch_size
            
            for batch_index, i in enumerate(
                range(0, len(chunks), args.batch_size),
                start=1,
            ):

                batch_start = time.perf_counter()

                batch_chunks = chunks[i:i + args.batch_size]

                texts = [
                    chunk.get("text", "")
                    for chunk in batch_chunks
                ]

                logger.info(
                    f"Batch {batch_index}/{total_batches} "
                    f"| Size: {len(batch_chunks)}"
                )

                embeddings = embedding_service.embed_batch(texts)

                EmbeddingWriter.append_batch(
                    output_file=output_file,
                    chunks=batch_chunks,
                    embeddings=embeddings,
                    model_name=args.model,
                )

                total_chunks_processed += len(batch_chunks)

                batch_elapsed = (
                    time.perf_counter() - batch_start
                )

                logger.info(
                    f"Batch completed in "
                    f"{batch_elapsed:.2f} sec."
                )

            file_elapsed = (
                time.perf_counter() - file_start
            )

            logger.info(
                f"Finished processing: {file_path}"
            )

            logger.info(
                f"Processing time: "
                f"{file_elapsed:.2f} sec."
            )

        except Exception:

            logger.exception(
                f"Failed while processing file: "
                f"{file_path}"
            )

            continue

    total_elapsed = (
        time.perf_counter() - pipeline_start
    )

    logger.info("=" * 80)
    logger.info("Embedding pipeline completed successfully.")
    logger.info(
        f"Total processed chunks : "
        f"{total_chunks_processed}"
    )
    logger.info(
        f"Total processing time : "
        f"{total_elapsed:.2f} sec."
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()