"""
backend/reguaz/retrieval/hybrid_qdrant.py

Hybrid retriever that combines Qdrant semantic search with BM25 keyword
search via Reciprocal Rank Fusion (RRF).

This module is the Qdrant equivalent of hybrid_retriever.py (HybridRetriever).
The pipeline structure, component wiring, public API, result format, logging
style, and docstring conventions deliberately mirror HybridRetriever.  The
only structural difference is that ChromaRetriever has been replaced by
QdrantRetriever and the ``chroma_dir`` parameter has been renamed to
``qdrant_dir`` to reflect the new backend.

Each component (embedding service, vector store, BM25 index, fusion) is
wired up independently so that any of them can be swapped out without
affecting the others.

Usage example
-------------
    from backend.reguaz.retrieval.hybrid_qdrant import HybridQdrantRetriever

    retriever = HybridQdrantRetriever(
        model_name="bge_m3",
        qdrant_dir="data/qdrant",
        chunks_dir="data/processed/chunks",
        top_k_semantic=20,
        top_k_bm25=20,
        rrf_k=60,
    )
    results = retriever.retrieve("What are the capital adequacy requirements?", top_k=10)
    # results → [{"id": "chunk_id_...", "rrf_score": 0.031, "rank": 1}, ...]
"""

from __future__ import annotations

import logging
import time
from typing import Any

from backend.reguaz.retrieval.bm25_retriever import BM25Retriever
from backend.reguaz.retrieval.fusion import compute_rrf_scores, reciprocal_rank_fusion
from backend.reguaz.retrieval.qdrant_retriever import QdrantRetriever
from backend.reguaz.retrieval.reranker import CrossEncoderReranker
from backend.reguaz.services.embeddings.embedding_factory import EmbeddingFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model-specific constants (mirrors hybrid_retriever.py)
# ---------------------------------------------------------------------------

_COLLECTION_MAP: dict[str, str] = {
    "bge_m3": "reguaz_bge_m3",
}

# BGE-M3 does not require an asymmetric query/passage prefix.
# This dict mirrors the _QUERY_PREFIX structure in hybrid_retriever.py so
# that adding further models in the future follows the same pattern.
_QUERY_PREFIX: dict[str, str] = {
    "bge_m3": "",
}


# ---------------------------------------------------------------------------
# HybridQdrantRetriever
# ---------------------------------------------------------------------------

class HybridQdrantRetriever:
    """
    Orchestrates Qdrant semantic search + BM25 search + RRF + Cross-Encoder reranking.

    This class is the Qdrant equivalent of HybridRetriever, extended with a
    Cross-Encoder reranking stage using BAAI/bge-reranker-v2-m3.

    Parameters
    ----------
    model_name : str
        Embedding model identifier.  Must be ``"bge_m3"``.
    qdrant_dir : str
        Path to the local Qdrant storage directory (e.g. ``"data/qdrant"``).
    chunks_dir : str
        Root directory containing chunk ``.jsonl`` files (e.g.
        ``"data/processed/chunks"``).
    top_k_semantic : int
        Number of candidate results to fetch from Qdrant before fusion.
        Default: 30.
    top_k_bm25 : int
        Number of candidate results to fetch from BM25 before fusion.
        Default: 30.
    rerank_top_k : int
        Number of candidates to keep after RRF fusion for the reranker.
        Default: 30.
    final_top_k : int
        Number of candidates to return after reranking.
        Default: 10.
    rrf_k : int
        The RRF smoothing constant.  Canonical value is 60.
    reranker_model : str
        Name/path of the Cross-Encoder model to load.
        Default: ``"BAAI/bge-reranker-v2-m3"``.

    Raises
    ------
    ValueError
        If *model_name* is not supported.
    FileNotFoundError
        If *chunks_dir* does not exist.
    Exception
        If the Qdrant collection cannot be opened (collection not found
        or empty collection) or the reranker fails to load.
    """

    def __init__(
        self,
        model_name: str,
        qdrant_dir: str,
        chunks_dir: str,
        top_k_semantic: int = 30,
        top_k_bm25: int = 30,
        rerank_top_k: int = 15,
        final_top_k: int = 5,
        rrf_k: int = 60,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ) -> None:
        # Normalise model name the same way the factory does.
        self._model_name = model_name.lower().replace("-", "_")

        if self._model_name not in _COLLECTION_MAP:
            raise ValueError(
                f"HybridQdrantRetriever: unsupported model '{model_name}'. "
                f"Supported: {list(_COLLECTION_MAP.keys())}"
            )

        self._top_k_semantic = top_k_semantic
        self._top_k_bm25 = top_k_bm25
        self._rerank_top_k = rerank_top_k
        self._final_top_k = final_top_k
        self._rrf_k = rrf_k
        self._reranker_model = reranker_model
        self._query_prefix: str = _QUERY_PREFIX[self._model_name]
        self._collection_name: str = _COLLECTION_MAP[self._model_name]

        logger.info(
            "HybridQdrantRetriever: initialising for model='%s' | collection='%s' | "
            "top_k_semantic=%d | top_k_bm25=%d | rerank_top_k=%d | final_top_k=%d | "
            "rrf_k=%d | reranker_model='%s'",
            self._model_name,
            self._collection_name,
            self._top_k_semantic,
            self._top_k_bm25,
            self._rerank_top_k,
            self._final_top_k,
            self._rrf_k,
            self._reranker_model,
        )

        # --- Component 1: Embedding service ---
        logger.info(
            "HybridQdrantRetriever: loading embedding service for model '%s' …",
            self._model_name,
        )
        self._embedding_service = EmbeddingFactory.get_service(self._model_name)
        logger.info("HybridQdrantRetriever: embedding service ready.")

        # --- Component 2: Qdrant semantic retriever ---
        logger.info(
            "HybridQdrantRetriever: opening Qdrant collection '%s' from '%s' …",
            self._collection_name,
            qdrant_dir,
        )
        self._qdrant_retriever = QdrantRetriever(
            persist_directory=qdrant_dir,
            collection_name=self._collection_name,
        )
        logger.info("HybridQdrantRetriever: Qdrant collection ready.")

        # --- Component 3: BM25 retriever ---
        logger.info(
            "HybridQdrantRetriever: building BM25 index from '%s' …", chunks_dir
        )
        self._bm25_retriever = BM25Retriever(chunks_dir=chunks_dir)
        logger.info(
            "HybridQdrantRetriever: BM25 index ready (%d chunks).",
            self._bm25_retriever.corpus_size,
        )

        # --- Component 4: Cross-Encoder Reranker ---
        logger.info(
            "HybridQdrantRetriever: loading Cross-Encoder reranker model '%s' …",
            self._reranker_model,
        )
        self._reranker = CrossEncoderReranker(
            model_name=self._reranker_model
        )
        logger.info("HybridQdrantRetriever: Cross-Encoder reranker ready.")

        logger.info(
            "HybridQdrantRetriever: all components initialised for model '%s'.",
            self._model_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Normalised model name (e.g. ``"bge_m3"``)."""
        return self._model_name

    @property
    def collection_name(self) -> str:
        """Qdrant collection name being queried."""
        return self._collection_name

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run the full hybrid retrieval pipeline with Cross-Encoder reranking.

        Steps
        -----
        1. Apply model-specific query prefix.
        2. Generate query embedding via the embedding service.
        3. Retrieve ``top_k_semantic`` results from Qdrant.
        4. Retrieve ``top_k_bm25`` results from BM25.
        5. Merge both ranked lists with RRF.
        6. Keep the top ``rerank_top_k`` fused results as candidates.
        7. Retrieve candidate document texts (in-memory) and rerank with Cross-Encoder.
        8. Sort candidates by rerank score descending and return the top ``top_k``.

        Parameters
        ----------
        query : str
            Raw query string (no prefix required; the retriever applies it).
        top_k : int | None
            Maximum number of final results to return. If None, defaults to the
            constructed ``final_top_k``.

        Returns
        -------
        list[dict]
            Ordered list (rank 1 first) of reranked result dicts, each containing:

            ``id``
                The matched chunk ID.
            ``rerank_score``
                The score assigned by the Cross-Encoder reranker.
            ``rank``
                1-indexed position in the final reranked ranking.
            ``rrf_score``
                The RRF score prior to reranking.
            ``semantic_rank``
                1-indexed position in the semantic-only ranking
                (``None`` if not in semantic results).
            ``bm25_rank``
                1-indexed position in the BM25-only ranking
                (``None`` if not in BM25 results).
        """
        t_total = time.perf_counter()

        if top_k is None:
            top_k = self._final_top_k

        # Step 1 – Apply query prefix
        prefixed_query = self._query_prefix + query
        logger.debug(
            "HybridQdrantRetriever.retrieve: query='%s…' (prefix='%s')",
            query[:60],
            self._query_prefix or "<none>",
        )

        # Step 2 – Generate embedding
        t0 = time.perf_counter()
        raw_embedding = self._embedding_service.embed_text(prefixed_query)
        # Coerce to list[float] — embed_text() may return numpy.ndarray or Tensor.
        if isinstance(raw_embedding, list):
            query_embedding: list[float] = raw_embedding
        else:
            query_embedding = [float(v) for v in raw_embedding]
        logger.info(
            "HybridQdrantRetriever.retrieve: embedding generated in %.3f s (dim=%d).",
            time.perf_counter() - t0,
            len(query_embedding),
        )

        # Step 3 – Semantic retrieval (Qdrant)
        t0 = time.perf_counter()
        semantic_results = self._qdrant_retriever.search(
            query_embedding=query_embedding,
            top_k=self._top_k_semantic,
        )
        semantic_ids: list[str] = semantic_results["ids"]
        logger.info(
            "HybridQdrantRetriever.retrieve: semantic search returned %d results in %.3f s.",
            len(semantic_ids),
            time.perf_counter() - t0,
        )

        # Step 4 – BM25 retrieval
        t0 = time.perf_counter()
        bm25_results = self._bm25_retriever.search(
            query=query,   # no prefix for BM25 — it tokenises the raw text
            top_k=self._top_k_bm25,
        )
        bm25_ids: list[str] = [r["id"] for r in bm25_results]
        logger.info(
            "HybridQdrantRetriever.retrieve: BM25 search returned %d results in %.3f s.",
            len(bm25_ids),
            time.perf_counter() - t0,
        )

        # Step 5 – Reciprocal Rank Fusion
        t0 = time.perf_counter()
        rrf_scores = compute_rrf_scores(
            ranked_lists=[semantic_ids, bm25_ids],
            k=self._rrf_k,
        )
        fused_ids = reciprocal_rank_fusion(
            ranked_lists=[semantic_ids, bm25_ids],
            k=self._rrf_k,
        )
        logger.info(
            "HybridQdrantRetriever.retrieve: RRF fusion completed in %.3f s "
            "(%d unique candidates).",
            time.perf_counter() - t0,
            len(fused_ids),
        )

        # Step 6 – Keep top rerank_top_k candidates
        fused_candidates = fused_ids[:self._rerank_top_k]

        # Step 7 – Retrieve candidate document texts (in-memory) and rerank
        t0 = time.perf_counter()
        texts_to_rerank = [
            self._bm25_retriever.get_text(cid) for cid in fused_candidates
        ]

        rerank_scores = self._reranker.rerank(query, texts_to_rerank)
        logger.info(
            "HybridQdrantRetriever.retrieve: Cross-Encoder reranked %d candidates in %.3f s.",
            len(fused_candidates),
            time.perf_counter() - t0,
        )

        # Build rank-lookup dictionaries for transparency.
        semantic_rank_map: dict[str, int] = {
            cid: rank for rank, cid in enumerate(semantic_ids, start=1)
        }
        bm25_rank_map: dict[str, int] = {
            cid: rank for rank, cid in enumerate(bm25_ids, start=1)
        }

        # Pair candidate IDs with their scores and rank metadata.
        scored_candidates = []
        for cid, score in zip(fused_candidates, rerank_scores):
            scored_candidates.append({
                "id": cid,
                "rerank_score": score,
                "rrf_score": rrf_scores.get(cid, 0.0),
                "semantic_rank": semantic_rank_map.get(cid),
                "bm25_rank": bm25_rank_map.get(cid),
            })

        # Step 8 – Sort candidates by rerank score descending (break ties lexicographically)
        scored_candidates.sort(key=lambda x: (-x["rerank_score"], x["id"]))

        results: list[dict[str, Any]] = []
        for rank, cand in enumerate(scored_candidates[:top_k], start=1):
            results.append(
                {
                    "id": cand["id"],
                    "rerank_score": cand["rerank_score"],
                    "rank": rank,
                    "rrf_score": cand["rrf_score"],
                    "semantic_rank": cand["semantic_rank"],
                    "bm25_rank": cand["bm25_rank"],
                }
            )

        logger.info(
            "HybridQdrantRetriever.retrieve: total retrieval time %.3f s.",
            time.perf_counter() - t_total,
        )
        return results
