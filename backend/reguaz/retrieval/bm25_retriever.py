"""
backend/reguaz/retrieval/bm25_retriever.py

Standalone BM25 retriever.

Loads every chunk JSONL file from a given root directory, tokenises the
text of each chunk, and builds a BM25Okapi index.  Retrieval is completely
independent from ChromaDB — the only shared dependency is ChunkReader.

Usage example
-------------
    from backend.reguaz.retrieval.bm25_retriever import BM25Retriever

    retriever = BM25Retriever(chunks_dir="data/processed/chunks")
    results = retriever.search("capital adequacy requirements", top_k=10)
    # results → [{"id": "chunk_id_...", "score": 3.14, "rank": 1}, ...]
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """
    Language-agnostic whitespace tokeniser.

    Lowercases the input and splits on whitespace.  Works well for
    Azerbaijani, Russian, and English mixed-language corpora without
    requiring language-specific libraries.

    Parameters
    ----------
    text : str
        Raw text to tokenise.

    Returns
    -------
    list[str]
        List of lowercase tokens.  Empty strings are filtered out.
    """
    return [token for token in text.lower().split() if token]


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    BM25-based retriever over the chunk corpus.

    Reads all ``.jsonl`` files under *chunks_dir* (recursively), builds a
    ``BM25Okapi`` index from each chunk's ``text`` field, and exposes a
    :meth:`search` method that returns top-K chunks by BM25 score.

    Parameters
    ----------
    chunks_dir : str
        Root directory that contains the chunked ``.jsonl`` files, e.g.
        ``"data/processed/chunks"``.

    Raises
    ------
    FileNotFoundError
        If *chunks_dir* does not exist.
    ValueError
        If no chunks are found after scanning *chunks_dir*.
    """

    def __init__(self, chunks_dir: str) -> None:
        self._chunks_dir = Path(chunks_dir)

        if not self._chunks_dir.exists():
            raise FileNotFoundError(
                f"BM25Retriever: chunks directory not found: '{self._chunks_dir}'"
            )

        logger.info(
            "BM25Retriever: loading corpus from '%s' …", self._chunks_dir
        )
        t0 = time.perf_counter()

        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._chunk_metadata: list[dict[str, Any]] = []

        self._load_corpus()

        if not self._chunk_ids:
            raise ValueError(
                f"BM25Retriever: no chunks found under '{self._chunks_dir}'. "
                "Ensure the directory contains valid .jsonl files."
            )

        logger.info(
            "BM25Retriever: indexing %d chunks with BM25Okapi …", len(self._chunk_ids)
        )
        tokenised_corpus = [_tokenize(text) for text in self._chunk_texts]
        self._index = BM25Okapi(tokenised_corpus)

        # Build in-memory lookup map for direct O(1) text retrieval.
        self._chunk_id_to_text = {
            cid: text for cid, text in zip(self._chunk_ids, self._chunk_texts)
        }

        elapsed = time.perf_counter() - t0
        logger.info(
            "BM25Retriever: index built in %.2f s  (%d chunks).",
            elapsed,
            len(self._chunk_ids),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_corpus(self) -> None:
        """
        Discover all ``.jsonl`` files and populate internal parallel lists.

        Skips any record that lacks a ``chunk_id`` or a non-empty ``text``
        field, logging a warning for each skipped record.
        """
        jsonl_files = sorted(self._chunks_dir.rglob("*.jsonl"))
        logger.info(
            "BM25Retriever: discovered %d JSONL file(s).", len(jsonl_files)
        )

        skipped = 0
        for file_path in jsonl_files:
            try:
                with file_path.open(encoding="utf-8") as fh:
                    for line_number, raw_line in enumerate(fh, start=1):
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        try:
                            chunk: dict[str, Any] = json.loads(raw_line)
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                "BM25Retriever: JSON decode error in '%s' line %d: %s — skipping.",
                                file_path.name,
                                line_number,
                                exc,
                            )
                            skipped += 1
                            continue

                        chunk_id: str | None = chunk.get("chunk_id")
                        text: str = chunk.get("text") or chunk.get("content") or ""

                        if not chunk_id:
                            logger.warning(
                                "BM25Retriever: record in '%s' line %d has no chunk_id — skipping.",
                                file_path.name,
                                line_number,
                            )
                            skipped += 1
                            continue

                        if not text.strip():
                            logger.warning(
                                "BM25Retriever: chunk '%s' has empty text — skipping.",
                                chunk_id,
                            )
                            skipped += 1
                            continue

                        self._chunk_ids.append(chunk_id)
                        self._chunk_texts.append(text)
                        self._chunk_metadata.append(chunk)

            except OSError as exc:
                logger.error(
                    "BM25Retriever: cannot read file '%s': %s — skipping file.",
                    file_path,
                    exc,
                )

        if skipped:
            logger.warning(
                "BM25Retriever: skipped %d record(s) during corpus loading.", skipped
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunk_ids)

    def get_text(self, chunk_id: str) -> str:
        """
        Retrieve the raw text content of a chunk by its ID from the in-memory corpus.

        Parameters
        ----------
        chunk_id : str
            The chunk ID to look up.

        Returns
        -------
        str
            The raw text content of the chunk, or empty string if not found.
        """
        return self._chunk_id_to_text.get(chunk_id, "")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-K chunks most relevant to *query* by BM25 score.

        Parameters
        ----------
        query : str
            Raw query string (no prefix required; the tokeniser handles it).
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[dict]
            Ordered list of result dicts (rank 1 first), each containing:

            ``id``
                The ``chunk_id`` of the matched chunk.
            ``score``
                The raw BM25 score (higher = more relevant).
            ``rank``
                1-indexed position in the ranked list.
            ``text``
                The chunk text (for debugging / re-ranking).
            ``metadata``
                Full chunk metadata dict as loaded from the JSONL file.
        """
        if not query.strip():
            logger.warning("BM25Retriever.search: received empty query — returning [].")
            return []

        tokens = _tokenize(query)
        if not tokens:
            logger.warning(
                "BM25Retriever.search: query '%s' produced no tokens — returning [].",
                query[:80],
            )
            return []

        scores: list[float] = self._index.get_scores(tokens).tolist()

        # Pair each score with its corpus index and sort descending.
        indexed_scores = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        results: list[dict[str, Any]] = []
        for rank, (corpus_idx, score) in enumerate(
            indexed_scores[:top_k], start=1
        ):
            results.append(
                {
                    "id": self._chunk_ids[corpus_idx],
                    "score": score,
                    "rank": rank,
                    "text": self._chunk_texts[corpus_idx],
                    "metadata": self._chunk_metadata[corpus_idx],
                }
            )

        logger.debug(
            "BM25Retriever.search: query='%s…' → %d results (top_k=%d).",
            query[:60],
            len(results),
            top_k,
        )
        return results
