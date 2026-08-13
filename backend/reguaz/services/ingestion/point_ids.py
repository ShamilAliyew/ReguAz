from __future__ import annotations

import uuid


POINT_ID_NAMESPACE = uuid.UUID("266db0a3-86b4-4d21-a61e-dbb1d37dc7f5")


def v2_point_id(chunk_id: str) -> str:
    """Return the stable point ID used by the existing V2 collection."""
    if not chunk_id.strip():
        raise ValueError("chunk_id must not be blank")
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))
