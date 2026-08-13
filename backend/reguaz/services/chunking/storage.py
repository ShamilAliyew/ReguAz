from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel


PROTECTED_V1_NAMES = {"chunks", "metadata", "embeddings", "qdrant", "cleaned_documents"}


def write_json(path: Path, value: BaseModel | dict) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Iterable[BaseModel]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value.model_dump(mode="json"), ensure_ascii=False) + "\n")


class AtomicBuildDirectory:
    def __init__(self, output: Path, *, force: bool) -> None:
        self.output = output.resolve()
        self.force = force
        self.temporary: Path | None = None

    def __enter__(self) -> Path:
        if self.output.name != "v2" or self.output.parent.name != "processed":
            raise ValueError("V2 output must resolve to an exact data/processed/v2-style target")
        if self.output.name in PROTECTED_V1_NAMES:
            raise ValueError("refusing to target a protected V1 directory")
        if self.output.exists() and not self.force:
            raise FileExistsError(f"{self.output} already exists; pass --force to replace only V2")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = Path(tempfile.mkdtemp(prefix=".v2-build-", dir=self.output.parent))
        return self.temporary

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.temporary is None:
            return False
        if exc_type is not None:
            shutil.rmtree(self.temporary, ignore_errors=True)
            return False
        backup = self.output.parent / f".v2-replaced-{os.getpid()}"
        if self.output.exists():
            if backup.exists():
                raise RuntimeError(f"refusing to overwrite unexpected backup {backup}")
            self.output.rename(backup)
        try:
            self.temporary.rename(self.output)
        except Exception:
            if backup.exists() and not self.output.exists():
                backup.rename(self.output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return False
