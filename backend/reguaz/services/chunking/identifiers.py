from __future__ import annotations

import hashlib
import re
import unicodedata


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prefixed(prefix: str, *parts: str, length: int = 24) -> str:
    canonical = "\x1f".join(unicodedata.normalize("NFC", p).strip() for p in parts)
    return f"{prefix}_{sha256_text(canonical)[:length]}"


def normalized_identity_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold()
    return " ".join(re.findall(r"\w+", value, re.UNICODE))


def document_id(
    *, title: str, document_type: str, category: str, authority: str | None, act_number: str | None,
    relative_path: str,
) -> tuple[str, bool]:
    title_key = normalized_identity_text(title)
    if authority and act_number and title_key:
        return prefixed("doc", authority, act_number, title_key), False
    if document_type and title_key:
        return prefixed("doc", document_type, title_key), False
    if category and title_key:
        return prefixed("doc", category, title_key), False
    return prefixed("doc", unicodedata.normalize("NFC", relative_path)), True


def document_version_id(doc_id: str, canonical_content: str) -> str:
    return prefixed("dver", doc_id, sha256_text(canonical_content))


def logical_unit_id(doc_id: str, node_type: str, locator: str) -> str:
    return prefixed("unit", doc_id, node_type, locator)


def logical_chunk_id(doc_id: str, unit_interval: str, split_index: int) -> str:
    return prefixed("lchunk", doc_id, unit_interval, str(split_index))


def chunk_id(doc_version_id: str, logical_id: str, content_hash: str) -> str:
    return prefixed("chunk", doc_version_id, logical_id, content_hash)


def parent_id(doc_version_id: str, root_locator: str, segment: int) -> str:
    return prefixed("parent", doc_version_id, root_locator, str(segment))
