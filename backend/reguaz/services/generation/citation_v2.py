from __future__ import annotations

import hashlib
from typing import Any

from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore
from backend.reguaz.services.generation.v2_models import (
    EvidenceItem,
    StructuredGeneration,
)


class CitationResolutionError(ValueError):
    pass


class BackendCitationResolver:
    def __init__(self, artifacts: V2ArtifactStore) -> None:
        self.artifacts = artifacts

    def resolve(
        self,
        generated: StructuredGeneration,
        evidence: list[EvidenceItem],
    ) -> dict[str, Any]:
        evidence_map = {item.evidence_id: item for item in evidence}
        citation_numbers: dict[str, int] = {}
        citations: list[dict[str, Any]] = []
        rendered_blocks: list[dict[str, Any]] = []

        for block in generated.answer_blocks:
            block_numbers: list[int] = []
            for evidence_id in block.evidence_ids:
                if evidence_id not in evidence_map:
                    raise CitationResolutionError(
                        f"unknown model evidence ID: {evidence_id}"
                    )
                if evidence_id not in citation_numbers:
                    number = len(citation_numbers) + 1
                    citation_numbers[evidence_id] = number
                    citations.append(
                        self._resolve_item(number, evidence_map[evidence_id])
                    )
                block_numbers.append(citation_numbers[evidence_id])
            rendered_blocks.append(
                {
                    "text": block.text,
                    "evidence_ids": list(block.evidence_ids),
                    "citation_numbers": block_numbers,
                }
            )

        answer = "\n\n".join(
            f"{block['text']} "
            + "".join(f"[{number}]" for number in block["citation_numbers"])
            for block in rendered_blocks
        ).strip()
        return {
            "answer": answer,
            "answer_blocks": rendered_blocks,
            "citations": citations,
            "sources": [self._legacy_source(citation) for citation in citations],
        }

    def _resolve_item(self, citation: int, item: EvidenceItem) -> dict[str, Any]:
        artifact_id = item.chunk_id or item.parent_chunk_id
        if not artifact_id:
            raise CitationResolutionError("evidence artifact ID is missing")
        resolved = self.artifacts.get_artifact(artifact_id)
        if resolved is None:
            raise CitationResolutionError("evidence artifact no longer exists")
        artifact_type, record = resolved
        if artifact_type != item.artifact_type:
            raise CitationResolutionError("evidence artifact type mismatch")
        if record.get("document_version_id") != item.document_version_id:
            raise CitationResolutionError("evidence document version mismatch")
        if record.get("content_sha256") != item.content_sha256:
            raise CitationResolutionError("evidence content hash mismatch")

        source = self.artifacts.get_source_representation(item.document_id)
        document_metadata = self.artifacts.get_document_metadata(item.document_id)
        if source is None or source["source_sha256"] != item.source_sha256:
            raise CitationResolutionError("evidence source representation mismatch")
        actual_source_hash = hashlib.sha256(source["text"].encode("utf-8")).hexdigest()
        if actual_source_hash != item.source_sha256:
            raise CitationResolutionError("evidence source hash mismatch")

        selectors: list[dict[str, Any]] = []
        exact_quotes: list[str] = []
        for selector in item.selectors:
            start = selector.position.start
            end = selector.position.end
            text = source["text"]
            exact = selector.quote.exact
            mode = "position"
            resolved_start, resolved_end = start, end
            if text[start:end] != exact:
                matches = _quote_matches(
                    text,
                    exact,
                    selector.quote.prefix,
                    selector.quote.suffix,
                )
                if len(matches) != 1:
                    raise CitationResolutionError(
                        "evidence selector cannot be resolved uniquely"
                    )
                resolved_start, resolved_end = matches[0]
                mode = "quote_fallback"
            selector_data = selector.model_dump(mode="json")
            selector_data["resolved_position"] = {
                "start": resolved_start,
                "end": resolved_end,
            }
            selector_data["resolution_mode"] = mode
            selectors.append(selector_data)
            exact_quotes.append(exact)

        if not selectors:
            raise CitationResolutionError("evidence has no validated selectors")
        provenance = [path.model_dump(mode="json") for path in item.provenance]
        return {
            "citation": citation,
            "evidence_id": item.evidence_id,
            "role": item.role,
            "relation_type": item.relation_type,
            "chunk_id": item.chunk_id,
            "logical_chunk_id": item.logical_chunk_id,
            "parent_chunk_id": item.parent_chunk_id,
            "document_id": item.document_id,
            "document_version_id": item.document_version_id,
            "document_title": item.document_title,
            "category": (document_metadata or {}).get("category", "unknown"),
            "canonical_locator": item.canonical_locator,
            "hierarchy": item.hierarchy,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "exact_quotes": exact_quotes,
            "selectors": selectors,
            "content_sha256": item.content_sha256,
            "source_sha256": item.source_sha256,
            "source_validated": True,
            "claim_support_status": "not_evaluated",
            "provenance": provenance,
        }

    @staticmethod
    def _legacy_source(citation: dict[str, Any]) -> dict[str, Any]:
        hierarchy = citation.get("hierarchy") or {}
        chapter = _hierarchy_label(hierarchy.get("chapter"))
        article = _hierarchy_label(hierarchy.get("article"))
        preview = "\n".join(citation.get("exact_quotes") or [])[:300]
        return {
            "citation": citation["citation"],
            "chunk_id": citation.get("chunk_id")
            or citation.get("parent_chunk_id")
            or "",
            "document_id": citation["document_id"],
            "document_name": citation["document_title"],
            "category": citation.get("category", "unknown"),
            "chapter": chapter,
            "article": article,
            "page": citation.get("page_start"),
            "chunk_preview": preview,
            "canonical_locator": citation.get("canonical_locator"),
            "role": citation.get("role"),
            "relation_type": citation.get("relation_type"),
            "selectors": citation.get("selectors") or [],
        }


def _quote_matches(
    text: str, exact: str, prefix: str, suffix: str
) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        position = text.find(exact, start)
        if position < 0:
            break
        end = position + len(exact)
        prefix_ok = (
            not prefix or text[max(0, position - len(prefix)) : position] == prefix
        )
        suffix_ok = not suffix or text[end : end + len(suffix)] == suffix
        if prefix_ok and suffix_ok:
            matches.append((position, end))
        start = position + 1
    return matches


def _hierarchy_label(value: object) -> str | None:
    if not isinstance(value, dict):
        return str(value) if value else None
    values = [str(value[key]) for key in ("number", "title") if value.get(key)]
    return " — ".join(values) or None
