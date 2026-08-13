import json
import glob
from pathlib import Path
from typing import Generator, Dict, Any, List

class EmbeddingReader:
    """
    Reads generated embeddings from JSONL files.
    """

    @staticmethod
    def discover_files(embeddings_root: str, model_name: str) -> List[Path]:
        """
        Discovers all embedding JSONL files for a specific model.
        """
        root_path = Path(embeddings_root) / model_name
        if not root_path.exists():
            return []
            
        pattern = str(root_path / "**" / "*.jsonl")
        return [Path(p) for p in glob.glob(pattern, recursive=True)]

    @staticmethod
    def read_file(file_path: Path | str) -> Generator[Dict[str, Any], None, None]:
        """
        Reads a single JSONL file and yields embedding records.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def read_all(embeddings_root: str, model_name: str) -> Generator[Dict[str, Any], None, None]:
        """
        Reads all embedding files for a model and yields all records.
        """
        files = EmbeddingReader.discover_files(embeddings_root, model_name)
        for file_path in files:
            yield from EmbeddingReader.read_file(file_path)
