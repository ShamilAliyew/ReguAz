from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SparseEmbedding(StrictModel):
    indices: list[int]
    values: list[float]

    @model_validator(mode="after")
    def aligned_unique_indices(self) -> "SparseEmbedding":
        if len(self.indices) != len(self.values):
            raise ValueError("sparse indices and values must be aligned")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("sparse indices must be unique")
        if any(index < 0 for index in self.indices):
            raise ValueError("sparse indices must be non-negative")
        if any(value <= 0 for value in self.values):
            raise ValueError("sparse values must be positive")
        return self


class EmbeddingRecord(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    chunk_id: str
    logical_chunk_id: str
    document_id: str
    document_version_id: str
    embedding_input_sha256: str
    model_id: str
    model_revision: str
    dense: list[float]
    sparse: SparseEmbedding

    @field_validator("dense")
    @classmethod
    def dense_not_empty(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("dense vector must not be empty")
        return value


class EmbeddingManifest(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    corpus_version: str
    source_chunk_manifest: str
    source_chunk_manifest_sha256: str
    source_child_count: int = Field(ge=0)
    embedded_chunk_count: int = Field(ge=0)
    model_id: str
    model_revision: str
    model_library: Literal["FlagEmbedding"] = "FlagEmbedding"
    model_library_version: str
    transformers_version: str
    torch_version: str
    vector_modes: list[Literal["dense", "sparse"]]
    dense_vector_name: Literal["dense"] = "dense"
    dense_dimension: int = Field(gt=0)
    dense_normalized: bool
    dense_dtype: Literal["float32"] = "float32"
    sparse_vector_name: Literal["sparse"] = "sparse"
    sparse_weighting: Literal["bge_m3_learned_lexical_weights"] = (
        "bge_m3_learned_lexical_weights"
    )
    input_field: Literal["embedding_text"] = "embedding_text"
    max_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    shard_size: int = Field(gt=0)
    shard_count: int = Field(ge=0)
    shard_sha256: dict[str, str]
    category_allowlist: list[str]
    approximate: Literal[False] = False
    build_timestamp: datetime
