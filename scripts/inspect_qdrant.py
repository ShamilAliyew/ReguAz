#!/usr/bin/env python3
"""
Inspect the local Qdrant database.

This utility connects to the local Qdrant instance and allows inspecting
collections, vector configurations, and stored points without modifying
the database.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Add project root to path so we can import backend
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient

from backend.reguaz.config import QDRANT_PATH
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "inspect_qdrant.log")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Inspect Qdrant database contents.")
    parser.add_argument(
        "--collection",
        type=str,
        help="Name of the collection to inspect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of points to inspect (default: 5).",
    )
    return parser.parse_args()


def _print_vectors_config(vectors_config: Any) -> None:
    """Helper to cleanly print vector configuration."""
    if hasattr(vectors_config, "size"):
        print(f"Vector dimension: {vectors_config.size}")
        print(f"Distance metric: {vectors_config.distance}")
    elif isinstance(vectors_config, dict):
        print("Vector components:")
        for name, config in vectors_config.items():
            size = getattr(config, "size", "N/A")
            dist = getattr(config, "distance", "N/A")
            print(f"  - '{name}': dimension={size}, distance={dist}")
    else:
        print(f"Vectors config: {vectors_config}")


def main() -> None:
    """Main execution function."""
    args = parse_args()

    logger.info("Connecting to Qdrant at %s", QDRANT_PATH)
    try:
        client = QdrantClient(path=str(QDRANT_PATH))
    except Exception as exc:
        logger.error("Failed to initialize QdrantClient: %s", exc)
        sys.exit(1)

    try:
        collections_response = client.get_collections()
        collections = [c.name for c in collections_response.collections]
    except Exception as exc:
        logger.error("Failed to list collections: %s", exc)
        client.close()
        sys.exit(1)

    if not collections:
        logger.info("No collections found in Qdrant.")
        print("No collections found in Qdrant database.")
        client.close()
        return

    collection_name = args.collection

    if not collection_name:
        if len(collections) == 1:
            collection_name = collections[0]
            logger.info(
                "Only one collection found. Automatically selecting '%s'.",
                collection_name,
            )
            print(f"Automatically selected collection: {collection_name}")
        else:
            print("Available collections:")
            for c in collections:
                print(f"  - {c}")
            print("\nPlease specify a collection using: --collection <name>")
            client.close()
            sys.exit(0)

    if collection_name not in collections:
        logger.error(
            "Collection '%s' does not exist. Available: %s",
            collection_name,
            ", ".join(collections),
        )
        print(f"Error: Collection '{collection_name}' does not exist.")
        client.close()
        sys.exit(1)

    try:
        info = client.get_collection(collection_name)
    except Exception as exc:
        logger.error("Failed to get info for collection '%s': %s", collection_name, exc)
        client.close()
        sys.exit(1)

    print(f"\n--- Collection: {collection_name} ---")
    print(f"Status: {info.status}")
    print(f"Number of vectors: {info.points_count}")
    
    vectors_config = info.config.params.vectors
    _print_vectors_config(vectors_config)

    if info.points_count == 0:
        print("\nCollection is empty.")
        client.close()
        return

    print(f"\n--- Inspecting up to {args.limit} points ---")
    
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=args.limit,
            with_payload=True,
            with_vectors=True,
        )
    except Exception as exc:
        logger.error("Failed to retrieve points: %s", exc)
        client.close()
        sys.exit(1)

    for i, point in enumerate(points, 1):
        print(f"\n[Point {i}]")
        print(f"ID: {point.id}")
        
        if point.vector is not None:
            if isinstance(point.vector, list):
                print(f"Vector dimension: {len(point.vector)}")
            elif isinstance(point.vector, dict):
                print(f"Vector named components: {list(point.vector.keys())}")
                for k, v in point.vector.items():
                    print(f"  - '{k}' dimension: {len(v)}")
        
        payload = point.payload or {}
        print(f"Payload keys: {list(payload.keys())}")
        print("Complete payload:")
        for k, v in payload.items():
            print(f"  {k}: {v}")

    client.close()


if __name__ == "__main__":
    main()
