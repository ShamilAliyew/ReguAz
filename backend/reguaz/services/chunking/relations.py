from __future__ import annotations

import re
from collections import defaultdict

from .models import ChildChunk, Reference


REFERENCE_PATTERNS = (
    re.compile(r"(?P<raw>(?:bu\s+\w+ın\s+)?(?P<locator>\d+(?:[-.]\d+)+)-ci\s+(?:yarım)?bənd(?:i|də|in)?)", re.I),
    re.compile(r"(?P<raw>Maddə\s+(?P<locator>\d+(?:-\d+)?))", re.I),
    re.compile(r"(?P<raw>(?P<locator>\d+(?:-\d+)?)-ci\s+maddə)", re.I),
    re.compile(r"(?P<raw>Əlavə\s+(?P<locator>\d+))", re.I),
)


def wire_relations(chunks: list[ChildChunk]) -> int:
    """Add sibling and bidirectional cross-reference relations in-place."""
    by_parent: dict[str, list[ChildChunk]] = defaultdict(list)
    for chunk in chunks:
        by_parent[chunk.parent_chunk_id].append(chunk)
    for siblings in by_parent.values():
        for index, chunk in enumerate(siblings):
            chunk.relations.previous_sibling = siblings[index - 1].chunk_id if index else None
            chunk.relations.next_sibling = siblings[index + 1].chunk_id if index + 1 < len(siblings) else None

    locator_index: dict[str, list[ChildChunk]] = defaultdict(list)
    for chunk in chunks:
        for atom in chunk.atomic_units:
            for locator in _locator_keys(atom.canonical_locator):
                locator_index[locator].append(chunk)

    unresolved = 0
    for chunk in chunks:
        seen: set[tuple[str, str]] = set()
        for pattern in REFERENCE_PATTERNS:
            for match in pattern.finditer(chunk.content):
                raw = match.group("raw")
                locator = match.group("locator")
                key = (raw, locator)
                if key in seen:
                    continue
                seen.add(key)
                candidates = [candidate for candidate in locator_index.get(locator, []) if candidate.chunk_id != chunk.chunk_id]
                unique = {candidate.chunk_id: candidate for candidate in candidates}
                if len(unique) == 1:
                    target = next(iter(unique.values()))
                    reference = Reference(
                        raw_reference=raw,
                        target_locator=locator,
                        target_logical_chunk_id=target.logical_chunk_id,
                        target_chunk_id=target.chunk_id,
                        resolved=True,
                        confidence=1.0,
                    )
                    chunk.relations.references.append(reference)
                    target.relations.referenced_by.append(
                        Reference(
                            raw_reference=raw,
                            target_locator=chunk.canonical_locator,
                            target_logical_chunk_id=chunk.logical_chunk_id,
                            target_chunk_id=chunk.chunk_id,
                            resolved=True,
                            confidence=1.0,
                        )
                    )
                else:
                    chunk.relations.references.append(
                        Reference(raw_reference=raw, target_locator=locator, resolved=False)
                    )
                    if "unresolved_reference" not in chunk.quality.quality_flags:
                        chunk.quality.quality_flags.append("unresolved_reference")
                    unresolved += 1
    return unresolved


def _locator_keys(locator: str) -> set[str]:
    keys = set(re.findall(r"\d+(?:[-.]\d+)+|(?<=Maddə\s)\d+(?:-\d+)?|(?<=Əlavə\s)\d+", locator))
    return keys
