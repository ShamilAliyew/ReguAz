from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .v2_models import SparseEmbedding
from backend.reguaz.utils.devices import select_inference_device


@dataclass(slots=True)
class DenseSparseBatch:
    dense: list[list[float]]
    sparse: list[SparseEmbedding]


class DenseSparseEncoder(Protocol):
    model_id: str
    model_revision: str
    dense_dimension: int
    normalized: bool

    def encode_documents(
        self, texts: list[str], *, batch_size: int, max_length: int
    ) -> DenseSparseBatch: ...

    def encode_queries(
        self, texts: list[str], *, batch_size: int, max_length: int
    ) -> DenseSparseBatch: ...


class BgeM3DenseSparseEncoder:
    """Official BGE-M3 dense+sparse inference adapter for V2 artifacts."""

    model_id = "BAAI/bge-m3"
    dense_dimension = 1024
    normalized = True

    def __init__(
        self,
        *,
        device: str | None = None,
        use_fp16: bool = False,
        local_files_only: bool = False,
        model_revision: str | None = None,
    ) -> None:
        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel
            from huggingface_hub import snapshot_download
            from transformers import AutoConfig
        except ImportError as exc:
            raise RuntimeError(
                "V2 dense+sparse embedding requires FlagEmbedding. Install project dependencies."
            ) from exc
        config = AutoConfig.from_pretrained(
            self.model_id,
            revision=model_revision,
            local_files_only=local_files_only,
        )
        resolved_revision = getattr(config, "_commit_hash", None) or model_revision
        if not resolved_revision:
            raise RuntimeError("could not resolve an immutable BGE-M3 model revision")
        self.model_revision = str(resolved_revision)

        # FlagEmbedding loads BGE-M3's separately stored sparse/ColBERT heads only
        # when model_name_or_path is an actual directory. Passing the Hub ID here
        # makes current FlagEmbedding releases silently initialize those heads at
        # random, which produces invalid and batch-dependent lexical weights.
        snapshot_path = Path(
            snapshot_download(
                repo_id=self.model_id,
                revision=self.model_revision,
                local_files_only=local_files_only,
            )
        )
        auxiliary_heads = ("sparse_linear.pt", "colbert_linear.pt")
        missing_heads = [
            name for name in auxiliary_heads if not (snapshot_path / name).is_file()
        ]
        if missing_heads:
            raise RuntimeError(
                "BGE-M3 auxiliary heads are missing from the resolved snapshot: "
                + ", ".join(missing_heads)
            )
        self.device = select_inference_device(device, torch_module=torch)
        self._model = BGEM3FlagModel(
            str(snapshot_path),
            normalize_embeddings=True,
            use_fp16=use_fp16,
            devices=self.device,
            passage_max_length=512,
            query_max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def encode_documents(
        self, texts: list[str], *, batch_size: int, max_length: int
    ) -> DenseSparseBatch:
        return self._encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            query=False,
        )

    def encode_queries(
        self, texts: list[str], *, batch_size: int = 1, max_length: int = 512
    ) -> DenseSparseBatch:
        """Encode queries once into matching dense and learned sparse vectors."""
        return self._encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            query=True,
        )

    def encode_query(
        self, text: str, *, max_length: int = 512
    ) -> tuple[list[float], SparseEmbedding]:
        batch = self.encode_queries([text], batch_size=1, max_length=max_length)
        return batch.dense[0], batch.sparse[0]

    def _encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        max_length: int,
        query: bool,
    ) -> DenseSparseBatch:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("embedding input texts must be non-empty")
        encode = self._model.encode_queries if query else self._model.encode_corpus
        output = encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = [[float(value) for value in vector] for vector in output["dense_vecs"]]
        sparse: list[SparseEmbedding] = []
        for weights in output["lexical_weights"]:
            ordered = sorted(
                (int(index), float(weight)) for index, weight in weights.items()
            )
            sparse.append(
                SparseEmbedding(
                    indices=[index for index, _ in ordered],
                    values=[weight for _, weight in ordered],
                )
            )
        result = DenseSparseBatch(dense=dense, sparse=sparse)
        self._validate(result, len(texts))
        return result

    def _validate(self, batch: DenseSparseBatch, expected_count: int) -> None:
        if len(batch.dense) != expected_count or len(batch.sparse) != expected_count:
            raise ValueError("BGE-M3 returned a batch with inconsistent length")
        for vector in batch.dense:
            if len(vector) != self.dense_dimension:
                raise ValueError(
                    f"BGE-M3 dense dimension mismatch: expected {self.dense_dimension}, got {len(vector)}"
                )
            if any(not math.isfinite(value) for value in vector):
                raise ValueError("dense vector contains NaN or infinity")
            norm = math.sqrt(sum(value * value for value in vector))
            if not 0.995 <= norm <= 1.005:
                raise ValueError(f"dense vector is not normalized (L2 norm={norm:.6f})")
        for vector in batch.sparse:
            if not vector.indices:
                raise ValueError("BGE-M3 returned an empty sparse vector")
            if any(not math.isfinite(value) for value in vector.values):
                raise ValueError("sparse vector contains NaN or infinity")
