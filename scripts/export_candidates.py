from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import pandas as pd

# Bootstrap: make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.reguaz.retrieval.retriever import ChromaRetriever
from backend.reguaz.services.embeddings.embedding_factory import EmbeddingFactory

def main():
    dataset_path = Path("data/evaluation/gold_dataset_for_embedding_excel.xlsx")
    chroma_dir = "data/chroma"
    output_path = Path("data/evaluation/chunk_candidates.json")

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_excel(dataset_path)

    print("Initializing embedding service (E5)...")
    embedder = EmbeddingFactory.get_service("e5")

    print("Initializing ChromaRetriever...")
    retriever = ChromaRetriever(
        persist_directory=chroma_dir,
        collection_name="reguaz_e5",
    )

    query_prefix = "query: "
    results_list = []

    print(f"Retrieving top-10 candidates for {len(df)} questions...")
    for idx, row in df.iterrows():
        question = str(row["question"])
        old_chunk_ids = str(row.get("relevant_chunk_ids", ""))
        old_doc_ids = str(row.get("expected_document_ids", ""))

        # Generate embedding
        prefixed_query = query_prefix + question
        embedding = embedder.embed_text(prefixed_query)

        # Search ChromaDB
        search_res = retriever.search(query_embedding=embedding, top_k=10)

        candidates = []
        for c_id, doc_text, meta, dist in zip(
            search_res["ids"],
            search_res["documents"],
            search_res["metadatas"],
            search_res["distances"]
        ):
            candidates.append({
                "chunk_id": c_id,
                "text": doc_text,
                "document_id": meta.get("document_id", ""),
                "distance": float(dist),
            })

        results_list.append({
            "row_index": int(idx),
            "question": question,
            "old_relevant_chunk_ids": old_chunk_ids,
            "old_expected_document_ids": old_doc_ids,
            "candidates": candidates
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
            print(f"Processed {idx + 1}/{len(df)} questions...")

    print(f"Saving candidates to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
