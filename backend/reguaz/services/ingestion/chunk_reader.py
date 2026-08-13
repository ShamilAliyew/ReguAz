import json
import glob
from pathlib import Path
from typing import Dict, Any, List

class ChunkReader:
    """
    Reads chunk JSONL files and builds a lookup mapping for ingestion.
    """

    @staticmethod
    def discover_files(chunks_root: str) -> List[Path]:
        """
        Discovers all chunk JSONL files.
        """
        root_path = Path(chunks_root)
        if not root_path.exists():
            return []
            
        pattern = str(root_path / "**" / "*.jsonl")
        return [Path(p) for p in glob.glob(pattern, recursive=True)]

    @staticmethod
    def build_lookup(chunks_root: str) -> Dict[str, Dict[str, Any]]:
        """
        Reads all chunk JSONL files and builds a lookup mapping of chunk_id to full chunk metadata.
        """
        lookup = {}
        files = ChunkReader.discover_files(chunks_root)
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunk = json.loads(line)
                        chunk_id = chunk.get("chunk_id")
                        if chunk_id:
                            lookup[chunk_id] = chunk
                            
        return lookup
