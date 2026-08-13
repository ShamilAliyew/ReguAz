import logging
from typing import List, Dict, Any

import chromadb
from chromadb.errors import NotFoundError

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """
    Manager for interacting with ChromaDB.
    """

    def __init__(self, persist_directory: str = "data/chroma"):
        logger.info(f"Initializing ChromaDB manager at {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)

    def get_or_create_collection(
        self,
        collection_name: str,
        metadata: Dict[str, Any] | None = None,
    ):
        """
        Returns an existing collection or creates a new one.
        """

        logger.info(f"Opening collection: {collection_name}")

        metadata = metadata or {
            "hnsw:space": "cosine"
        }

        return self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
        )

    def reset_collection(
        self,
        collection_name: str,
        metadata: Dict[str, Any] | None = None,
    ):
        """
        Deletes collection if it exists and recreates it.
        """

        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")

        except NotFoundError:
            logger.info(
                f"Collection {collection_name} does not exist. Creating a new one."
            )

        return self.get_or_create_collection(
            collection_name,
            metadata,
        )

    def get_collection_count(self, collection_name: str) -> int:
        """
        Returns number of vectors inside collection.
        """

        try:
            collection = self.client.get_collection(collection_name)
            return collection.count()

        except NotFoundError:
            return 0

    def insert_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ):
        """
        Inserts chunk batch into ChromaDB.
        """

        if not chunks:
            logger.warning("No chunks to insert.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) != Embeddings ({len(embeddings)})"
            )

        collection = self.client.get_collection(collection_name)

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            chunk_id = chunk.get("chunk_id") or chunk.get("id")

            if chunk_id is None:
                raise ValueError("Chunk has no chunk_id.")

            ids.append(str(chunk_id))

            documents.append(
                chunk.get("text")
                or chunk.get("content")
                or ""
            )

            metadata = {}

            for key, value in chunk.items():

                if key in ("text", "content"):
                    continue

                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value

                elif value is None:
                    metadata[key] = ""

                else:
                    metadata[key] = str(value)

            metadatas.append(metadata)

        logger.info(
            f"Inserting batch of {len(ids)} vectors into {collection_name}"
        )

        # Safety split.
        INTERNAL_BATCH_SIZE = 10000

        for start in range(0, len(ids), INTERNAL_BATCH_SIZE):

            end = start + INTERNAL_BATCH_SIZE

            collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info(
            f"Successfully inserted {len(ids)} vectors."
        )