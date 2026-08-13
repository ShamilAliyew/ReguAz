from __future__ import annotations

import hashlib
import re
from typing import Any

from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore
from backend.reguaz.services.generation.v2_models import (
    EvidenceItem,
    PositionSelector,
    QuoteSelector,
    SourceSelector,
    V2GenerationSettings,
)


PAGE_MARKER = re.compile(r"<!--\s*PAGE:\s*(\d+)\s*-->", re.IGNORECASE)


class EvidenceSelector:
    """Build, validate and deterministically deduplicate authoritative evidence."""

    def __init__(
        self,
        artifacts: V2ArtifactStore,
        settings: V2GenerationSettings,
    ) -> None:
        self.artifacts = artifacts
        self.settings = settings

    def select(
        self, candidates: list[dict[str, Any]]
    ) -> tuple[list[EvidenceItem], list[dict[str, str]]]:
        built: list[EvidenceItem] = []
        diagnostics: list[dict[str, str]] = []
        for candidate in candidates:
            try:
                built.append(self._build(candidate))
            except ValueError as exc:
                record = candidate["record"]
                diagnostics.append(
                    {
                        "code": "invalid_evidence_candidate",
                        "artifact_id": str(
                            record.get("chunk_id") or record.get("parent_chunk_id")
                        ),
                        "reason": str(exc),
                    }
                )

        built.sort(key=self._sort_key)
        selected: list[EvidenceItem] = []
        by_artifact: dict[str, EvidenceItem] = {}
        by_hash: dict[str, EvidenceItem] = {}
        for item in built:
            artifact_id = item.chunk_id or item.parent_chunk_id or ""
            duplicate = by_artifact.get(artifact_id) or by_hash.get(
                item.evidence_text_sha256
            )
            if duplicate is not None:
                known = {path.model_dump_json() for path in duplicate.provenance}
                duplicate.provenance.extend(
                    path
                    for path in item.provenance
                    if path.model_dump_json() not in known
                )
                diagnostics.append(
                    {
                        "code": "duplicate_evidence",
                        "artifact_id": artifact_id,
                        "reason": f"deduplicated into {duplicate.chunk_id or duplicate.parent_chunk_id}",
                    }
                )
                continue
            selected.append(item)
            by_artifact[artifact_id] = item
            by_hash[item.evidence_text_sha256] = item
        return selected, diagnostics

    def _build(self, candidate: dict[str, Any]) -> EvidenceItem:
        if not candidate.get("provenance"):
            raise ValueError("evidence provenance is empty")
        record = dict(candidate["record"])
        artifact_type = str(candidate["artifact_type"])
        document_id = str(record["document_id"])
        metadata = self.artifacts.get_document_metadata(document_id)
        source = self.artifacts.get_source_representation(document_id)
        if metadata is None or source is None:
            raise ValueError("authoritative cleaned source is missing")
        if record.get("document_version_id") != metadata["document_version_id"]:
            raise ValueError("document version mismatch")

        spans = [dict(span) for span in record.get("source_spans") or []]
        content = str(record.get("content") or "").strip()
        if artifact_type == "parent":
            content, spans = self._parent_excerpt(record, source["text"], spans)
        if not content:
            raise ValueError("evidence content is empty")

        selectors = self._selectors(source, spans)
        if not selectors:
            raise ValueError("no validated source selector")
        content_sha256 = str(record.get("content_sha256") or "")
        if not content_sha256:
            raise ValueError("authoritative content hash is missing")
        quality = record.get("quality") or {}
        return EvidenceItem(
            role=candidate["role"],
            relation_type=candidate.get("relation_type"),
            seed_chunk_id=candidate["provenance"][0].seed_chunk_id,
            artifact_type=artifact_type,  # type: ignore[arg-type]
            chunk_id=record.get("chunk_id"),
            logical_chunk_id=record.get("logical_chunk_id"),
            parent_chunk_id=record.get("parent_chunk_id"),
            document_id=document_id,
            document_version_id=str(record["document_version_id"]),
            document_title=str(record["document_title"]),
            canonical_locator=str(record.get("canonical_locator") or ""),
            hierarchy=dict(record.get("hierarchy") or {}),
            page_start=record.get("page_start"),
            page_end=record.get("page_end"),
            content=content,
            content_sha256=content_sha256,
            evidence_text_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            source_sha256=str(metadata["source_sha256"]),
            source_spans=spans,
            selectors=selectors,
            selection_reason=str(candidate["selection_reason"]),
            priority=int(candidate["priority"]),
            final_rank=candidate.get("seed_rank"),
            reranker_score=candidate.get("reranker_score"),
            quality_flags=list(quality.get("quality_flags") or []),
            provenance=candidate["provenance"],
        )

    def _parent_excerpt(
        self,
        record: dict[str, Any],
        source_text: str,
        spans: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        if not spans:
            raise ValueError("parent has no source spans")
        first = dict(spans[0])
        start = int(first["char_start"])
        end = min(
            int(first["char_end"]), start + self.settings.parent_excerpt_char_limit
        )
        excerpt = source_text[start:end].strip()
        if not excerpt:
            raise ValueError("parent excerpt is empty")
        left_trim = len(source_text[start:end]) - len(source_text[start:end].lstrip())
        right = start + left_trim + len(excerpt)
        first["char_start"] = start + left_trim
        first["char_end"] = right
        first["line_start"] = source_text.count("\n", 0, first["char_start"]) + 1
        first["line_end"] = source_text.count("\n", 0, first["char_end"]) + 1
        return excerpt, [first]

    @staticmethod
    def _selectors(
        source: dict[str, Any], spans: list[dict[str, Any]]
    ) -> list[SourceSelector]:
        text = str(source["text"])
        page_bounds = _page_bounds(text)
        selectors: list[SourceSelector] = []
        for span in spans:
            start = int(span["char_start"])
            end = int(span["char_end"])
            if start < 0 or end > len(text) or end <= start:
                continue
            exact = text[start:end]
            if not exact:
                continue
            page = span.get("page")
            page_position = None
            if page is not None and int(page) in page_bounds:
                page_start, page_end = page_bounds[int(page)]
                if page_start <= start <= end <= page_end:
                    page_position = PositionSelector(
                        start=start - page_start,
                        end=end - page_start,
                    )
            selectors.append(
                SourceSelector(
                    representation_id=str(source["representation_id"]),
                    document_version_id=str(source["document_version_id"]),
                    source_sha256=str(source["source_sha256"]),
                    page=int(page) if page is not None else None,
                    position=PositionSelector(start=start, end=end),
                    page_position=page_position,
                    quote=QuoteSelector(
                        exact=exact,
                        prefix=text[max(0, start - 48) : start],
                        suffix=text[end : min(len(text), end + 48)],
                    ),
                )
            )
        return selectors

    @staticmethod
    def _sort_key(item: EvidenceItem) -> tuple[int, int, int, str]:
        artifact_id = item.chunk_id or item.parent_chunk_id or ""
        quality_penalty = len(item.quality_flags)
        return (item.priority, item.final_rank or 9999, quality_penalty, artifact_id)


def _page_bounds(text: str) -> dict[int, tuple[int, int]]:
    matches = list(PAGE_MARKER.finditer(text))
    result: dict[int, tuple[int, int]] = {}
    for index, match in enumerate(matches):
        raw_start = match.end()
        raw_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw = text[raw_start:raw_end]
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw) - len(raw.rstrip())
        start = raw_start + left_trim
        end = raw_end - right_trim
        result[int(match.group(1))] = (start, max(start, end))
    return result
