import json
from pathlib import Path
from typing import Any


class EmbeddingWriter:
    """
    Writes generated embeddings to JSONL files.

    Directory structure:

    data/processed/embeddings/
        bge_m3/
            laws/
            aml_kyc/
            ...
        e5/
        jina_v3/
    """

    @staticmethod
    def get_output_path(
        chunk_file: str,
        chunks_root: str,
        embeddings_root: str,
        model_name: str,
    ) -> Path:
        """
        Creates the output path for an embedding JSONL file.

        Example:

        data/processed/chunks/laws/Banklar haqqında Qanun.jsonl

        →

        data/processed/embeddings/bge_m3/laws/Banklar haqqında Qanun_embeddings.jsonl
        """

        chunk_path = Path(chunk_file)

        relative_path = chunk_path.relative_to(chunks_root)

        output_dir = Path(embeddings_root) / model_name / relative_path.parent

        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{chunk_path.stem}_embeddings.jsonl"

        return output_file

    @staticmethod
    def write_batch(
        output_file: Path,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        model_name: str,
    ) -> None:
        """
        Writes one batch of embeddings.

        Appends to the JSONL file.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must be equal."
            )

        with output_file.open("a", encoding="utf-8") as f:

            for chunk, embedding in zip(chunks, embeddings):

                record = {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "embedding_model": model_name,
                    "embedding": embedding,
                }

                json.dump(record, f, ensure_ascii=False)

                f.write("\n")

    @staticmethod
    def append_batch(
        output_file: Path,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        model_name: str,
    ) -> None:
        EmbeddingWriter.write_batch(
            output_file,
            chunks,
            embeddings,
            model_name,
        )