"""
backend/reguaz/evaluation/context_enricher.py

Context Enrichment Pipeline for RAGAS Preparation.

Takes a lightweight ``answers.csv`` (produced by GenerationEvaluator) and
enriches it with full chunk text and metadata by resolving the
``Retrieved Chunk IDs`` column against the on-disk chunk JSONL corpus.

Architecture
------------
::

    answers.csv  ──►  ContextEnricher  ──►  answers_with_contexts.csv
                           │
                     ChunkReader.build_lookup()
                     (single batch load from disk)

The enricher deliberately avoids per-chunk Qdrant queries.  Instead it
batch-loads every chunk from the JSONL files once via
:class:`~backend.reguaz.services.chunks.chunk_reader.ChunkReader`, then
performs in-memory lookups for each question's chunk IDs.  This is
O(total_chunks) for the load step and O(1) per lookup — far faster than
N×K individual Qdrant point reads.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from backend.reguaz.services.chunks.chunk_reader import ChunkReader
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "context_enrichment.log")

# Canonical fields to expose in the enriched context objects.
_CONTEXT_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "document_id",
    "title",
    "category",
    "chapter",
    "article",
    "section",
    "subsection",
    "page_start",
    "page_end",
    "source_file",
    "text",
)

# Metadata-only fields (everything except text) for the separate column.
_METADATA_FIELDS: tuple[str, ...] = tuple(
    f for f in _CONTEXT_FIELDS if f != "text"
)


class ContextEnricher:
    """Enriches generation evaluation results with full chunk contexts.

    Parameters
    ----------
    chunks_dir : str | Path
        Root directory containing chunk ``.jsonl`` files
        (e.g. ``data/processed/chunks``).
    """

    def __init__(self, chunks_dir: str | Path) -> None:
        self._chunks_dir = Path(chunks_dir)

        logger.info(
            "ContextEnricher: building chunk lookup from '%s' …",
            self._chunks_dir,
        )
        t0 = time.perf_counter()
        reader = ChunkReader()
        self._lookup: dict[str, dict[str, Any]] = reader.build_lookup(
            self._chunks_dir
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "ContextEnricher: chunk lookup ready — %d chunks loaded in %.3f s.",
            len(self._lookup),
            elapsed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
    ) -> Path:
        """Enrich an ``answers.csv`` with full chunk contexts.

        Parameters
        ----------
        input_csv : str | Path
            Path to the generation evaluation CSV (must contain a
            ``Retrieved Chunk IDs`` column with JSON-encoded chunk ID
            lists).
        output_csv : str | Path
            Destination path for the enriched CSV.

        Returns
        -------
        Path
            The resolved *output_csv* path.
        """
        input_path = Path(input_csv)
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise FileNotFoundError(
                f"ContextEnricher: input CSV not found: '{input_path}'"
            )

        logger.info("ContextEnricher: loading '%s' …", input_path)
        df = pd.read_csv(input_path)
        total_questions = len(df)
        logger.info(
            "ContextEnricher: loaded %d rows from answers CSV.", total_questions
        )

        # Collect all unique chunk IDs across the dataset for stats.
        all_chunk_ids: set[str] = set()
        missing_chunk_ids: set[str] = set()

        enriched_rows: list[dict[str, Any]] = []

        for idx, row in df.iterrows():
            question_id = row.get("Question ID", idx)

            # Parse the JSON-encoded chunk ID list.
            raw_ids = row.get("Retrieved Chunk IDs")
            chunk_ids = self._parse_chunk_ids(raw_ids, question_id)
            all_chunk_ids.update(chunk_ids)

            # Look up each chunk and build context / metadata lists.
            contexts: list[dict[str, Any]] = []
            metadata_list: list[dict[str, Any]] = []

            for cid in chunk_ids:
                chunk = self._lookup.get(cid)
                if chunk is None:
                    missing_chunk_ids.add(cid)
                    logger.warning(
                        "ContextEnricher: chunk '%s' (Question %s) not found "
                        "in lookup — skipping.",
                        cid,
                        question_id,
                    )
                    continue

                contexts.append(
                    {field: chunk.get(field) for field in _CONTEXT_FIELDS}
                )
                metadata_list.append(
                    {field: chunk.get(field) for field in _METADATA_FIELDS}
                )

            enriched_rows.append(
                {
                    "Question ID": row.get("Question ID"),
                    "Question": row.get("Question"),
                    "Ground Truth": row.get("Ground Truth"),
                    "Generated Answer": row.get("Generated Answer"),
                    "Retrieved Contexts": json.dumps(
                        contexts, ensure_ascii=False
                    ),
                    "Retrieved Metadata": json.dumps(
                        metadata_list, ensure_ascii=False
                    ),
                }
            )

        out_df = pd.DataFrame(enriched_rows)
        out_df.to_csv(output_path, index=False)

        # Summary logging.
        logger.info(
            "ContextEnricher: enrichment complete.\n"
            "  Questions processed : %d\n"
            "  Unique chunk IDs    : %d\n"
            "  Missing chunk IDs   : %d%s\n"
            "  Output saved to     : %s",
            total_questions,
            len(all_chunk_ids),
            len(missing_chunk_ids),
            f" ({sorted(missing_chunk_ids)})" if missing_chunk_ids else "",
            output_path,
        )

        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_chunk_ids(
        raw: Any, question_id: Any
    ) -> list[str]:
        """Safely parse a JSON-encoded list of chunk IDs."""
        if pd.isna(raw) or raw is None:
            logger.warning(
                "ContextEnricher: no chunk IDs for Question %s.", question_id
            )
            return []

        try:
            ids = json.loads(str(raw))
            if isinstance(ids, list):
                return [str(i) for i in ids]
            logger.warning(
                "ContextEnricher: unexpected chunk ID format for Question %s: %s",
                question_id,
                type(ids),
            )
            return []
        except json.JSONDecodeError:
            logger.warning(
                "ContextEnricher: malformed JSON in chunk IDs for Question %s.",
                question_id,
            )
            return []
