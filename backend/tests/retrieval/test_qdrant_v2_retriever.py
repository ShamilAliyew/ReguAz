from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import models

from backend.reguaz.retrieval.qdrant_v2_retriever import QdrantV2Retriever
from backend.reguaz.retrieval.v2_contract import V2RetrievalContract
from backend.reguaz.services.embeddings.v2_models import SparseEmbedding
from backend.reguaz.services.ingestion.point_ids import v2_point_id

from .conftest import first_child_payload


class FakeQdrantClient:
    def __init__(
        self,
        *,
        contract: V2RetrievalContract,
        payload: dict[str, object],
        metadata_revision: str = "test-revision",
        alias_exists: bool = True,
    ) -> None:
        self.contract = contract
        self.payload = payload
        self.metadata_revision = metadata_revision
        self.alias_exists = alias_exists
        self.closed = False
        self.calls: list[dict[str, object]] = []

    def get_aliases(self) -> object:
        aliases = []
        if self.alias_exists:
            aliases.append(
                SimpleNamespace(
                    alias_name="reguaz_v2_current",
                    collection_name="physical-v2",
                )
            )
        return SimpleNamespace(aliases=aliases)

    def collection_exists(self, name: str) -> bool:
        return name == "physical-v2"

    def get_collection(self, name: str) -> object:
        assert name == "physical-v2"
        metadata = {
            "corpus_version": self.contract.chunk_manifest.corpus_version,
            "embedding_model_id": "BAAI/bge-m3",
            "embedding_model_revision": self.metadata_revision,
        }
        params = SimpleNamespace(
            vectors={
                "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
            },
            sparse_vectors={"sparse": models.SparseVectorParams()},
        )
        return SimpleNamespace(config=SimpleNamespace(metadata=metadata, params=params))

    def count(self, *, collection_name: str, exact: bool) -> object:
        assert collection_name == "physical-v2" and exact is True
        return SimpleNamespace(count=self.contract.chunk_manifest.child_count)

    def query_points(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        point = SimpleNamespace(
            id="00000000-0000-0000-0000-000000000001", score=0.9, payload=self.payload
        )
        return SimpleNamespace(points=[point])

    def query_batch_points(self, **kwargs: object) -> list[object]:
        self.calls.append(kwargs)
        point = SimpleNamespace(
            id=v2_point_id(str(self.payload["chunk_id"])),
            score=0.9,
            payload={"chunk_id": self.payload["chunk_id"]},
        )
        return [SimpleNamespace(points=[point]), SimpleNamespace(points=[point])]

    def retrieve(self, **kwargs: object) -> list[object]:
        self.calls.append(kwargs)
        return [
            SimpleNamespace(id=point_id, payload=self.payload)
            for point_id in kwargs["ids"]
        ]

    def close(self) -> None:
        self.closed = True


def test_named_dense_sparse_search_and_filters(v2_root: Path) -> None:
    contract = V2RetrievalContract.load(v2_root)
    client = FakeQdrantClient(contract=contract, payload=first_child_payload(v2_root))
    retriever = QdrantV2Retriever(
        contract=contract,
        qdrant_path=v2_root / "qdrant",
        client_factory=lambda _: client,
    )

    dense = retriever.dense_search(
        [1.0] + [0.0] * 1023,
        top_k=30,
        filters={"category": "laws", "document_id": "doc-filter"},
    )
    sparse = retriever.sparse_search(
        SparseEmbedding(indices=[17], values=[0.5]), top_k=30
    )

    assert client.calls[0]["using"] == "dense"
    assert client.calls[1]["using"] == "sparse"
    assert client.calls[0]["collection_name"] == "reguaz_v2_current"
    assert client.calls[0]["limit"] == 30
    assert len(client.calls[0]["query_filter"].must) == 3
    assert dense[0]["chunk_id"] == client.payload["chunk_id"]
    assert dense[0]["point_id"].endswith("0001")
    assert set(sparse[0]) == {"chunk_id", "point_id", "score", "rank"}
    assert client.calls[0]["with_payload"] == ["chunk_id"]

    retriever.close()
    assert client.closed is True


def test_manifest_collection_mismatch_fails_fast_and_releases_client(
    v2_root: Path,
) -> None:
    contract = V2RetrievalContract.load(v2_root)
    client = FakeQdrantClient(
        contract=contract,
        payload=first_child_payload(v2_root),
        metadata_revision="wrong-revision",
    )

    with pytest.raises(ValueError, match="embedding_model_revision"):
        QdrantV2Retriever(
            contract=contract,
            qdrant_path=v2_root / "qdrant",
            client_factory=lambda _: client,
        )

    assert client.closed is True


def test_batch_search_is_lean_and_final_payloads_use_one_batch(v2_root: Path) -> None:
    contract = V2RetrievalContract.load(v2_root)
    payload = first_child_payload(v2_root)
    client = FakeQdrantClient(contract=contract, payload=payload)
    retriever = QdrantV2Retriever(
        contract=contract,
        qdrant_path=v2_root / "qdrant",
        client_factory=lambda _: client,
    )
    dense, sparse = retriever.batch_search(
        [1.0] + [0.0] * 1023,
        SparseEmbedding(indices=[17], values=[0.5]),
    )
    request_call = client.calls[-1]
    assert [request.using for request in request_call["requests"]] == [
        "dense",
        "sparse",
    ]
    assert all(
        request.with_payload == ["chunk_id"] for request in request_call["requests"]
    )
    assert set(dense[0]) == set(sparse[0]) == {"chunk_id", "point_id", "score", "rank"}

    loaded = retriever.fetch_full_payloads([dense[0]])
    retrieve_call = client.calls[-1]
    assert retrieve_call["ids"] == [v2_point_id(str(payload["chunk_id"]))]
    assert retrieve_call["with_payload"] is True
    assert loaded[str(payload["chunk_id"])]["content"] == payload["content"]
    retriever.close()


def test_missing_alias_fails_fast(v2_root: Path) -> None:
    contract = V2RetrievalContract.load(v2_root)
    client = FakeQdrantClient(
        contract=contract,
        payload=first_child_payload(v2_root),
        alias_exists=False,
    )
    with pytest.raises(ValueError, match="alias does not exist"):
        QdrantV2Retriever(
            contract=contract,
            qdrant_path=v2_root / "qdrant",
            client_factory=lambda _: client,
        )
    assert client.closed is True


def test_remote_qdrant_uses_url_without_opening_local_storage(v2_root: Path) -> None:
    contract = V2RetrievalContract.load(v2_root)
    client = FakeQdrantClient(contract=contract, payload=first_child_payload(v2_root))
    targets: list[str] = []

    retriever = QdrantV2Retriever(
        contract=contract,
        qdrant_url="https://example.qdrant.io:6333",
        qdrant_api_key="secret-not-forwarded-to-test-factory",
        client_factory=lambda target: targets.append(target) or client,
    )

    assert targets == ["https://example.qdrant.io:6333"]
    assert retriever.mode == "remote"
    retriever.close()
    assert client.closed is True


def test_remote_qdrant_rejects_invalid_url_before_client_creation(
    v2_root: Path,
) -> None:
    with pytest.raises(ValueError, match="must use http"):
        QdrantV2Retriever(
            contract=V2RetrievalContract.load(v2_root),
            qdrant_url="example.qdrant.io:6333",
            client_factory=lambda _: pytest.fail("client must not be created"),
        )


def test_contract_detects_manifest_count_mismatch(v2_root: Path) -> None:
    manifest_path = v2_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["child_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="child counts differ"):
        V2RetrievalContract.load(v2_root)


def test_v2_point_id_remains_compatible_with_ingestion_namespace() -> None:
    assert v2_point_id("chunk_example") == "eb66d143-cc1f-5546-90af-0e179033a98c"
