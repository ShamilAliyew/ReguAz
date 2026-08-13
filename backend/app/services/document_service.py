"""
app/services/document_service.py

Service layer for Document and Metadata retrieval.
Decouples raw PDF/markdown parsing and folder scanning from router endpoints.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.schemas.document import (
    DocumentMetadataResponse,
    DocumentPageResponse,
    ArticleInfo,
    DocumentHighlightResponse,
)

logger = logging.getLogger(__name__)


def extract_page_text(markdown_content: str, page_number: int) -> Optional[str]:
    """
    Extracts text content for a specific page from a cleaned markdown file.
    Cleaned markdown files use <!-- PAGE: N --> as page boundaries.
    """
    pattern = rf"<!--\s*PAGE:\s*{page_number}\s*-->"
    matches = list(re.finditer(pattern, markdown_content, re.IGNORECASE))
    if not matches:
        return None

    start_pos = matches[0].end()

    # Find the next page boundary or end of string
    next_pattern = r"<!--\s*PAGE:\s*\d+\s*-->"
    next_matches = list(
        re.finditer(next_pattern, markdown_content[start_pos:], re.IGNORECASE)
    )
    if next_matches:
        end_pos = start_pos + next_matches[0].start()
    else:
        end_pos = len(markdown_content)

    return markdown_content[start_pos:end_pos].strip()


class DocumentService:
    """
    Service responsible for loading document catalog metadata, slicing pages,
    and resolving highlighting targets.
    """

    def __init__(
        self,
        metadata_dir: str | Path,
        cleaned_docs_dir: str | Path,
        chunk_lookup: dict[str, dict[str, Any]],
    ) -> None:
        """
        Initialize the DocumentService by recursively scanning the metadata folder.
        """
        self.metadata_dir = Path(metadata_dir)
        self.cleaned_docs_dir = Path(cleaned_docs_dir)
        self.chunk_lookup = chunk_lookup

        self._doc_metadata_map: Dict[str, Dict[str, Any]] = {}
        self._doc_path_map: Dict[str, Path] = {}
        self._scan_documents()

    def _scan_documents(self) -> None:
        """
        Scans all *_metadata.json files recursively to build local maps.
        """
        if not self.metadata_dir.exists():
            logger.warning("Metadata directory %s does not exist.", self.metadata_dir)
            return

        logger.info("Scanning metadata files in %s...", self.metadata_dir)
        count = 0
        for path in self.metadata_dir.rglob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                doc_id = meta.get("document_id")
                if doc_id:
                    self._doc_metadata_map[doc_id] = meta

                    category = meta.get("category", "")
                    processed_file = meta.get("processed_file", f"{doc_id}.md")
                    md_path = self.cleaned_docs_dir / category / processed_file

                    if not md_path.exists():
                        # Try fallback filename matching doc_id directly
                        md_path = self.cleaned_docs_dir / category / f"{doc_id}.md"

                    self._doc_path_map[doc_id] = md_path
                    count += 1
            except Exception as e:
                logger.warning("Failed to parse metadata file at %s: %s", path, e)
                continue
        logger.info("Successfully indexed metadata and paths for %d documents.", count)

    def get_document_metadata(
        self, document_id: str
    ) -> Optional[DocumentMetadataResponse]:
        """
        Finds document metadata and compiles associated articles.
        """
        meta = self._doc_metadata_map.get(document_id)
        if not meta:
            return None

        # Build list of unique articles associated with this document from chunk_lookup
        related_articles = set()
        for chunk in self.chunk_lookup.values():
            if chunk.get("document_id") == document_id:
                art = chunk.get("article")
                if art:
                    related_articles.add(str(art))

        return DocumentMetadataResponse(
            document_id=meta.get("document_id", document_id),
            title=meta.get("title") or meta.get("document_id") or "Unknown Document",
            category=meta.get("category", "unknown"),
            total_pages=meta.get("total_pages"),
            total_chunks=meta.get("total_chunks"),
            language=meta.get("language"),
            parser=meta.get("parser"),
            publication_date=meta.get("created_at"),
            status="active",
            related_articles=sorted(list(related_articles)),
            document_metadata=meta,
        )

    def get_all_documents(self) -> List[DocumentMetadataResponse]:
        """
        Retrieve all documents indexed in the system.
        """
        results = []
        for doc_id in self._doc_metadata_map:
            meta = self.get_document_metadata(doc_id)
            if meta:
                results.append(meta)
        return results

    def get_document_page(
        self, document_id: str, page_number: int
    ) -> Optional[DocumentPageResponse]:
        """
        Extracts content text and finds related articles for a given document page.
        """
        meta = self._doc_metadata_map.get(document_id)
        if not meta:
            return None

        md_path = self._doc_path_map.get(document_id)
        if not md_path or not md_path.exists():
            logger.warning(
                "Markdown source file not found for document_id %s.", document_id
            )
            return None

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error("Failed to read markdown file at %s: %s", md_path, e)
            return None

        page_text = extract_page_text(content, page_number)
        if page_text is None:
            return None

        # Find any chunks that appear on this page to identify section headings
        articles = []
        for chunk in self.chunk_lookup.values():
            if chunk.get("document_id") == document_id:
                p_start = chunk.get("page_start")
                p_end = chunk.get("page_end")
                if p_start is not None and p_end is not None:
                    if p_start <= page_number <= p_end:
                        articles.append(
                            ArticleInfo(
                                chapter=chunk.get("chapter"),
                                article=chunk.get("article"),
                                section=chunk.get("section"),
                                chunk_id=chunk.get("chunk_id") or chunk.get("id") or "",
                            )
                        )

        return DocumentPageResponse(
            document_id=document_id,
            page_number=page_number,
            page_content=page_text,
            article_information=articles,
            metadata={"source_file": meta.get("source_file")},
        )

    def get_highlight(
        self, document_id: str, chunk_id: str
    ) -> Optional[DocumentHighlightResponse]:
        """
        Finds chunk details to return navigation parameters for highlighting.
        """
        chunk = self.chunk_lookup.get(chunk_id)
        if not chunk:
            return None

        # Fetch basic chunk specs
        page = chunk.get("page_start") or 1
        article = chunk.get("article")
        text = chunk.get("text") or chunk.get("content") or ""

        return DocumentHighlightResponse(
            document_id=document_id,
            page=page,
            article=article,
            chunk_id=chunk_id,
            chunk_start=None,
            chunk_end=None,
            highlighted_text=text,
            offset_status="future_enhancement",
        )


class V2DocumentService:
    """Read-only viewer over authoritative V2 cleaned-Markdown representations."""

    def __init__(self, artifact_store: Any) -> None:
        from backend.reguaz.services.generation.evidence_v2 import _page_bounds

        self.artifacts = artifact_store
        self._page_bounds = _page_bounds
        self._children_by_document: dict[str, list[dict[str, Any]]] = {}
        for child in self.artifacts.iter_children():
            self._children_by_document.setdefault(str(child["document_id"]), []).append(
                child
            )

    def get_document_metadata(
        self, document_id: str
    ) -> Optional[DocumentMetadataResponse]:
        metadata = self.artifacts.get_document_metadata(document_id)
        source = self.artifacts.get_source_representation(document_id)
        if metadata is None or source is None:
            return None
        bounds = self._page_bounds(source["text"])
        articles = {
            _label((child.get("hierarchy") or {}).get("article"))
            for child in self._children_by_document.get(document_id, [])
        }
        return DocumentMetadataResponse(
            document_id=document_id,
            document_version_id=metadata["document_version_id"],
            title=metadata["title"],
            category=metadata["category"],
            total_pages=max(bounds) if bounds else None,
            total_chunks=len(self._children_by_document.get(document_id, [])),
            language=metadata.get("language"),
            parser=metadata.get("parser_name"),
            publication_date=metadata.get("publication_date"),
            status=metadata.get("status", "unknown"),
            related_articles=sorted(value for value in articles if value),
            document_metadata=metadata,
            source_representation="cleaned_markdown",
            source_sha256=metadata["source_sha256"],
        )

    def get_all_documents(self) -> List[DocumentMetadataResponse]:
        output = []
        for metadata in sorted(
            self.artifacts.iter_document_metadata(), key=lambda item: item["title"]
        ):
            value = self.get_document_metadata(str(metadata["document_id"]))
            if value is not None:
                output.append(value)
        return output

    def get_document_page(
        self, document_id: str, page_number: int
    ) -> Optional[DocumentPageResponse]:
        source = self.artifacts.get_source_representation(document_id)
        if source is None:
            return None
        bounds = self._page_bounds(source["text"])
        if page_number not in bounds:
            return None
        start, end = bounds[page_number]
        page_content = source["text"][start:end]
        articles = []
        for child in self._children_by_document.get(document_id, []):
            page_start = child.get("page_start")
            page_end = child.get("page_end")
            if page_start is None or page_end is None:
                continue
            if int(page_start) <= page_number <= int(page_end):
                hierarchy = child.get("hierarchy") or {}
                articles.append(
                    ArticleInfo(
                        chapter=_label(hierarchy.get("chapter")),
                        article=_label(hierarchy.get("article")),
                        section=_label(hierarchy.get("section")),
                        chunk_id=str(child["chunk_id"]),
                    )
                )
        return DocumentPageResponse(
            document_id=document_id,
            document_version_id=source["document_version_id"],
            page_number=page_number,
            page_content=page_content,
            article_information=articles,
            representation_id=source["representation_id"],
            source_sha256=source["source_sha256"],
            page_start_offset=start,
            metadata={
                "representation_type": "cleaned_markdown",
                "viewer_notice": "Authoritative cleaned Markdown; not original PDF.",
            },
        )

    def get_highlight(
        self, document_id: str, chunk_id: str
    ) -> Optional[DocumentHighlightResponse]:
        child = self.artifacts.get_child(chunk_id)
        if child is None or child.get("document_id") != document_id:
            return None
        from backend.reguaz.services.generation.evidence_v2 import EvidenceSelector
        from backend.reguaz.services.generation.v2_models import (
            EvidencePath,
            V2GenerationSettings,
        )

        selected, diagnostics = EvidenceSelector(
            self.artifacts, V2GenerationSettings()
        ).select(
            [
                {
                    "artifact_type": "child",
                    "record": child,
                    "role": "seed",
                    "relation_type": None,
                    "priority": 1,
                    "seed_rank": 1,
                    "reranker_score": None,
                    "selection_reason": "viewer highlight",
                    "provenance": [
                        EvidencePath(
                            seed_chunk_id=chunk_id,
                            selection_reason="viewer highlight",
                        )
                    ],
                }
            ]
        )
        if not selected:
            return DocumentHighlightResponse(
                document_id=document_id,
                page=child.get("page_start") or 1,
                article=_label((child.get("hierarchy") or {}).get("article")),
                chunk_id=chunk_id,
                chunk_start=None,
                chunk_end=None,
                highlighted_text="",
                offset_status="unresolved",
                warning=(
                    diagnostics[0]["reason"] if diagnostics else "highlight unresolved"
                ),
            )
        evidence = selected[0]
        first = evidence.selectors[0]
        return DocumentHighlightResponse(
            document_id=document_id,
            page=first.page or child.get("page_start") or 1,
            article=_label((child.get("hierarchy") or {}).get("article")),
            chunk_id=chunk_id,
            chunk_start=first.position.start,
            chunk_end=first.position.end,
            highlighted_text=first.quote.exact,
            offset_status="supported",
            selectors=[item.model_dump(mode="json") for item in evidence.selectors],
        )


def _label(value: object) -> Optional[str]:
    if isinstance(value, dict):
        parts = [str(value[key]) for key in ("number", "title") if value.get(key)]
        return " — ".join(parts) or None
    return str(value) if value else None
