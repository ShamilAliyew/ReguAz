"""Chunk reading utilities.

This module is responsible solely for reading pre-existing chunk JSONL
files from disk and building a ``chunk_id -> chunk`` lookup table. It
does not perform any chunking, mutation, or metadata generation — it
assumes chunk files already exist in their final form and simply loads
them as-is.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "chunk_reader.log")

CHUNK_FILE_SUFFIX = ".jsonl"
CHUNK_ID_KEY = "chunk_id"


class ChunkReader:
    """Reads chunk JSONL files and builds a chunk_id lookup table.

    This class only reads chunk files that already exist on disk
    (typically produced by an earlier, separate chunking pipeline). It
    performs no chunking, no mutation of chunk content, and no
    metadata generation.

    Each JSONL line is expected to represent a single chunk object with
    (at minimum) a unique ``chunk_id`` field. The full chunk object is
    preserved exactly as stored.
    """

    def discover_files(self, chunks_root: str | Path) -> list[Path]:
        """Discovers all chunk JSONL files under a root directory.

        Args:
            chunks_root: Root directory under which chunk files are
                stored (e.g. ``data/processed/chunks``). The directory
                is searched recursively.

        Returns:
            A sorted list of paths to all discovered ``.jsonl`` files.

        Raises:
            FileNotFoundError: If ``chunks_root`` does not exist.
            NotADirectoryError: If ``chunks_root`` is not a directory.
        """
        root = Path(chunks_root)

        if not root.exists():
            message = f"Chunks root directory does not exist: '{root}'."
            logger.error(message)
            raise FileNotFoundError(message)

        if not root.is_dir():
            message = f"Chunks root path is not a directory: '{root}'."
            logger.error(message)
            raise NotADirectoryError(message)

        files = sorted(root.rglob(f"*{CHUNK_FILE_SUFFIX}"))

        logger.info("Discovered %d chunk file(s) under '%s'.", len(files), root)
        return files

    def read_file(self, file_path: Path | str) -> Generator[dict[str, Any], None, None]:
        """Reads a single JSONL chunk file line by line.

        Args:
            file_path: Path to the JSONL file to read.

        Yields:
            One chunk dictionary per non-empty line, preserved exactly
            as stored in the file.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
            json.JSONDecodeError: If a line contains malformed JSON.
        """
        path = Path(file_path)

        if not path.exists():
            message = f"Chunk file does not exist: '{path}'."
            logger.error(message)
            raise FileNotFoundError(message)

        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.exception(
                        "Malformed JSON in file '%s' at line %d.", path, line_number
                    )
                    raise

    def build_lookup(self, chunks_root: str | Path) -> dict[str, dict[str, Any]]:
        """Builds a chunk_id -> chunk lookup table from all chunk files.

        If a duplicate ``chunk_id`` is encountered, the first occurrence
        is kept and every subsequent occurrence is skipped with a
        logged warning. This method never raises on duplicates; use the
        logs to identify and fix the underlying chunking issue.

        Args:
            chunks_root: Root directory under which chunk files are
                stored (e.g. ``data/processed/chunks``).

        Returns:
            A dictionary mapping each unique ``chunk_id`` to its full,
            unmodified chunk object.

        Raises:
            FileNotFoundError: If ``chunks_root`` does not exist, or a
                discovered file disappears before it can be read.
            NotADirectoryError: If ``chunks_root`` is not a directory.
            ValueError: If a chunk is missing a ``chunk_id`` field.
            json.JSONDecodeError: If any chunk file contains malformed
                JSON.
        """
        files = self.discover_files(chunks_root)
        lookup: dict[str, dict[str, Any]] = {}
        duplicate_count = 0

        for file_path in files:
            duplicate_count += self._merge_file_into_lookup(file_path, lookup)

        logger.info(
            "Built chunk lookup with %d unique chunk(s) from %d file(s) "
            "(%d duplicate chunk_id(s) skipped).",
            len(lookup),
            len(files),
            duplicate_count,
        )
        return lookup

    def _merge_file_into_lookup(
        self,
        file_path: Path,
        lookup: dict[str, dict[str, Any]],
    ) -> int:
        """Reads a single chunk file and merges its chunks into a lookup.

        Chunks whose ``chunk_id`` already exists in ``lookup`` are
        skipped (the first occurrence wins) and logged as a warning.

        Args:
            file_path: Path to the JSONL file to read.
            lookup: Existing ``chunk_id -> chunk`` lookup dictionary to
                merge new chunks into, in place.

        Returns:
            The number of duplicate chunks skipped while reading this
            file.

        Raises:
            ValueError: If a chunk is missing a ``chunk_id`` field.
        """
        chunk_count = 0
        duplicate_count = 0

        for chunk in self.read_file(file_path):
            chunk_id = self._extract_chunk_id(chunk, file_path)

            if self._is_duplicate(chunk_id, lookup, file_path):
                duplicate_count += 1
                continue

            lookup[chunk_id] = chunk
            chunk_count += 1

        logger.info(
            "Read %d chunk(s) from '%s' (%d duplicate(s) skipped).",
            chunk_count,
            file_path,
            duplicate_count,
        )
        return duplicate_count

    @staticmethod
    def _extract_chunk_id(chunk: dict[str, Any], file_path: Path) -> str:
        """Extracts and validates the chunk_id field from a chunk.

        Args:
            chunk: Chunk dictionary parsed from a JSONL line.
            file_path: Path of the file the chunk was read from, used
                for error context.

        Returns:
            The chunk's ``chunk_id`` value.

        Raises:
            ValueError: If the chunk has no ``chunk_id`` field.
        """
        chunk_id = chunk.get(CHUNK_ID_KEY)

        if not chunk_id:
            message = f"Chunk missing '{CHUNK_ID_KEY}' field in file '{file_path}'."
            logger.error(message)
            raise ValueError(message)

        return chunk_id

    @staticmethod
    def _is_duplicate(
        chunk_id: str,
        lookup: dict[str, dict[str, Any]],
        file_path: Path,
    ) -> bool:
        """Checks whether a chunk_id has already been seen, warning if so.

        Args:
            chunk_id: Chunk id to check.
            lookup: Existing lookup dictionary to check against.
            file_path: Path of the file the chunk was read from, used
                for warning context.

        Returns:
            True if ``chunk_id`` already exists in ``lookup`` (and the
            duplicate has been logged as a warning), False otherwise.
        """
        if chunk_id in lookup:
            logger.warning(
                "Duplicate chunk_id '%s' encountered in file '%s'; "
                "keeping the first occurrence and skipping this one.",
                chunk_id,
                file_path,
            )
            return True
        return False