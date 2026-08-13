from __future__ import annotations

import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .classifier import classify_document, infer_document_type
from .config import ChunkingConfig
from .identifiers import (
    chunk_id,
    document_id,
    document_version_id,
    logical_chunk_id,
    parent_id,
    prefixed,
    sha256_text,
)
from .models import (
    ChildChunk,
    DocumentMetadata,
    Manifest,
    ParentChunk,
    Quality,
    QualityReport,
    Relations,
)
from .normalizer import canonical_content, normalize_for_parsing
from .packer import AdaptivePacker, ChunkDraft, atomic_models, sentence_count
from .parser import StructureParser
from .relations import wire_relations
from .storage import AtomicBuildDirectory, write_json, write_jsonl
from .tokenizer import TokenCounter


@dataclass(slots=True)
class ProcessedDocument:
    metadata: DocumentMetadata
    parents: list[ParentChunk]
    chunks: list[ChildChunk]
    report: QualityReport
    profile: str
    profiles: list[str]
    canonical_content_hash: str


@dataclass(slots=True)
class PipelineResult:
    manifest: Manifest
    output_root: Path | None

    @property
    def succeeded(self) -> bool:
        return self.manifest.failed_document_count == 0


@dataclass(slots=True)
class Discovery:
    path: Path
    relative_path: str
    category: str
    raw_text: str
    canonical_text: str
    title: str
    short_title: str
    profile: str
    profiles: list[str]
    document_type: str
    base_document_id: str
    identity_fallback: bool


class ChunkingPipeline:
    def __init__(self, config: ChunkingConfig, tokenizer: TokenCounter) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.parser = StructureParser()
        self.packer = AdaptivePacker(config, tokenizer)

    def discover(self, document: str | None = None, limit: int | None = None) -> list[Path]:
        root = self.config.input_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"input root does not exist: {root}")
        # Actual directory entries are retained; normalized names are never reconstructed.
        paths = sorted(
            (path for path in root.rglob("*.md") if path.is_file()),
            key=lambda path: unicodedata.normalize("NFC", path.relative_to(root).as_posix()),
        )
        if document:
            needle = unicodedata.normalize("NFC", document)
            paths = [
                path
                for path in paths
                if needle
                in {
                    unicodedata.normalize("NFC", path.name),
                    unicodedata.normalize("NFC", path.stem),
                    unicodedata.normalize("NFC", path.relative_to(root).as_posix()),
                }
            ]
            if not paths:
                raise FileNotFoundError(f"requested document was not discovered: {document}")
        return paths[:limit] if limit is not None else paths

    def dry_run(self, *, document: str | None = None, limit: int | None = None) -> dict:
        paths = self.discover(document, limit)
        detected: Counter[str] = Counter()
        problems: list[dict[str, str]] = []
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8")
                title, _ = self._titles(path, text)
                profile, profiles = classify_document(text, path.parent.name, title)
                detected.update(profiles)
                if not text.strip():
                    problems.append({"source": str(path), "problem": "empty document"})
            except Exception as exc:
                problems.append({"source": str(path), "problem": str(exc)})
        return {
            "discovered_documents": len(paths),
            "detected_profiles": dict(sorted(detected.items())),
            "expected_output_root": str(self.config.output_root.resolve()),
            "expected_files_per_successful_document": 4,
            "tokenizer_id": self.tokenizer.tokenizer_id,
            "tokenizer_available": True,
            "approximate_tokenizer": self.tokenizer.approximate,
            "potential_processing_problems": problems,
        }

    def run(
        self,
        *,
        document: str | None = None,
        limit: int | None = None,
        force: bool = False,
        fail_on_quality_errors: bool = False,
    ) -> PipelineResult:
        paths = self.discover(document, limit)
        discoveries, discovery_failures = self._prepare_discoveries(paths)
        assigned_ids = self._assign_unique_document_ids(discoveries)
        processed: list[ProcessedDocument] = []
        failures = list(discovery_failures)
        failed_reports: list[QualityReport] = [
            QualityReport(
                document_id=prefixed("doc", failure["source"]),
                status="failed",
                profile="UNKNOWN",
                profiles_detected=[],
                parent_count=0,
                child_count=0,
                atomic_unit_count=0,
                table_count=0,
                unclassified_node_count=0,
                unresolved_reference_count=0,
                duplicate_chunk_count=0,
                min_tokens=0,
                median_tokens=0,
                p95_tokens=0,
                max_tokens=0,
                quality_flags=["low_text_quality"],
                errors=[failure["error"]],
            )
            for failure in discovery_failures
        ]
        for discovery in discoveries:
            try:
                item = self._process(discovery, assigned_ids[discovery.relative_path])
                if fail_on_quality_errors and self._quality_errors(item.report):
                    raise ValueError(
                        "quality errors: " + ", ".join(sorted(self._quality_errors(item.report)))
                    )
                processed.append(item)
            except Exception as exc:
                failures.append({"source": discovery.relative_path, "error": f"{type(exc).__name__}: {exc}"})
                failed_reports.append(self._failed_report(discovery, assigned_ids[discovery.relative_path], str(exc)))

        self._mark_duplicate_documents(processed)
        manifest = self._manifest(paths, processed, failures)
        self._validate_corpus(processed, manifest)
        with AtomicBuildDirectory(self.config.output_root, force=force) as build:
            documents_dir = build / "documents"
            documents_dir.mkdir()
            for item in processed:
                target = documents_dir / item.metadata.document_id
                target.mkdir()
                write_json(target / "document.json", item.metadata)
                write_jsonl(target / "parents.jsonl", item.parents)
                write_jsonl(target / "chunks.jsonl", item.chunks)
                write_json(target / "quality_report.json", item.report)
            for report in failed_reports:
                target = documents_dir / report.document_id
                target.mkdir(exist_ok=True)
                write_json(target / "quality_report.json", report)
            write_json(build / "manifest.json", manifest)
        return PipelineResult(manifest=manifest, output_root=self.config.output_root.resolve())

    def _prepare_discoveries(self, paths: list[Path]) -> tuple[list[Discovery], list[dict[str, str]]]:
        root = self.config.input_root.resolve()
        result: list[Discovery] = []
        failures: list[dict[str, str]] = []
        for path in paths:
            relative = path.relative_to(root).as_posix()
            try:
                raw = path.read_text(encoding="utf-8")
                canonical = canonical_content(raw)
                title, short_title = self._titles(path, raw)
                category = path.parent.name
                profile, profiles = classify_document(raw, category, title)
                doc_type = infer_document_type(profile, title)
                base_id, fallback = document_id(
                    title=title,
                    document_type=doc_type,
                    category=category,
                    authority=None,
                    act_number=self._act_number(raw),
                    relative_path=relative,
                )
                result.append(
                    Discovery(
                        path=path,
                        relative_path=relative,
                        category=category,
                        raw_text=raw,
                        canonical_text=canonical,
                        title=title,
                        short_title=short_title,
                        profile=profile,
                        profiles=profiles,
                        document_type=doc_type,
                        base_document_id=base_id,
                        identity_fallback=fallback,
                    )
                )
            except Exception as exc:
                failures.append({"source": relative, "error": f"{type(exc).__name__}: {exc}"})
        return result, failures

    @staticmethod
    def _assign_unique_document_ids(discoveries: list[Discovery]) -> dict[str, str]:
        grouped: dict[str, list[Discovery]] = defaultdict(list)
        for discovery in discoveries:
            grouped[discovery.base_document_id].append(discovery)
        assigned: dict[str, str] = {}
        for base_id, group in grouped.items():
            for discovery in sorted(group, key=lambda value: value.relative_path):
                assigned[discovery.relative_path] = (
                    base_id
                    if len(group) == 1
                    else prefixed("doc", base_id, discovery.relative_path)
                )
        return assigned

    def _process(self, discovery: Discovery, doc_id: str) -> ProcessedDocument:
        if not discovery.raw_text.strip():
            return self._process_empty(discovery, doc_id)
        normalized = normalize_for_parsing(discovery.raw_text)
        doc_version = document_version_id(doc_id, discovery.canonical_text)
        parsed = self.parser.parse(normalized, discovery.profile)
        drafts = self.packer.pack(parsed, document_id=doc_id, title=discovery.title)
        if not drafts:
            raise ValueError("parser produced no retrieval chunks")
        metadata = DocumentMetadata(
            document_id=doc_id,
            document_version_id=doc_version,
            title=discovery.title,
            short_title=discovery.short_title,
            document_type=discovery.document_type,
            act_number=self._act_number(discovery.raw_text),
            category=discovery.category,
            source_file=discovery.path.name,
            source_relative_path=discovery.relative_path,
            source_sha256=hashlib.sha256(discovery.raw_text.encode("utf-8")).hexdigest(),
        )
        parents, chunks = self._materialize(metadata, drafts, normalized.has_page_markers)
        unresolved = wire_relations(chunks)
        self._add_special_relations(chunks, discovery.profiles)
        duplicates = self._mark_duplicate_chunks(chunks)
        flags = sorted({flag for chunk in chunks for flag in chunk.quality.quality_flags})
        if discovery.identity_fallback:
            flags.append("document_identity_fallback")
            for chunk in chunks:
                if "document_identity_fallback" not in chunk.quality.quality_flags:
                    chunk.quality.quality_flags.append("document_identity_fallback")
        tokens = [chunk.embedding_token_count for chunk in chunks]
        report = QualityReport(
            document_id=doc_id,
            status="success",
            profile=discovery.profile,
            profiles_detected=discovery.profiles,
            parent_count=len(parents),
            child_count=len(chunks),
            atomic_unit_count=sum(len(chunk.atomic_units) for chunk in chunks),
            table_count=parsed.table_count,
            unclassified_node_count=parsed.unclassified_count,
            unresolved_reference_count=unresolved,
            duplicate_chunk_count=duplicates,
            min_tokens=min(tokens),
            median_tokens=int(statistics.median(tokens)),
            p95_tokens=self._percentile(tokens, 0.95),
            max_tokens=max(tokens),
            quality_flags=sorted(set(flags)),
            errors=[],
        )
        self._validate_document(parents, chunks)
        return ProcessedDocument(
            metadata,
            parents,
            chunks,
            report,
            discovery.profile,
            discovery.profiles,
            sha256_text(discovery.canonical_text),
        )

    def _process_empty(self, discovery: Discovery, doc_id: str) -> ProcessedDocument:
        """Account for an empty source without inventing retrieval content."""
        metadata = DocumentMetadata(
            document_id=doc_id,
            document_version_id=document_version_id(doc_id, discovery.canonical_text),
            title=discovery.title,
            short_title=discovery.short_title,
            document_type=discovery.document_type,
            act_number=None,
            category=discovery.category,
            source_file=discovery.path.name,
            source_relative_path=discovery.relative_path,
            source_sha256=hashlib.sha256(discovery.raw_text.encode("utf-8")).hexdigest(),
        )
        report = QualityReport(
            document_id=doc_id,
            status="success",
            profile=discovery.profile,
            profiles_detected=discovery.profiles,
            parent_count=0,
            child_count=0,
            atomic_unit_count=0,
            table_count=0,
            unclassified_node_count=0,
            unresolved_reference_count=0,
            duplicate_chunk_count=0,
            min_tokens=0,
            median_tokens=0,
            p95_tokens=0,
            max_tokens=0,
            quality_flags=["low_text_quality"],
            errors=["input Markdown is empty; no legal content was invented"],
        )
        return ProcessedDocument(
            metadata,
            [],
            [],
            report,
            discovery.profile,
            discovery.profiles,
            sha256_text(discovery.canonical_text),
        )

    def _materialize(
        self, metadata: DocumentMetadata, drafts: list[ChunkDraft], has_pages: bool
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        grouped: list[list[ChunkDraft]] = []
        current: list[ChunkDraft] = []
        boundary = None
        for draft in drafts:
            if current and (draft.boundary is not boundary or self._draft_tokens([*current, draft]) > self.config.preferred_parent_max):
                grouped.append(current)
                current = []
            current.append(draft)
            boundary = draft.boundary
        if current:
            grouped.append(current)

        parents: list[ParentChunk] = []
        chunks: list[ChildChunk] = []
        continuation_roots: dict[str, str] = {}
        for segment_index, group in enumerate(grouped, start=1):
            root = group[0].boundary or group[0].node
            root_locator = root.locator()
            article_root_id = prefixed("article", metadata.document_id, root_locator)
            parent_chunk_id = parent_id(metadata.document_version_id, root_locator, segment_index)
            child_ids: list[str] = []
            staged: list[tuple[ChunkDraft, str, str]] = []
            for draft in group:
                unit_interval = "|".join(atom.logical_unit_id for atom in draft.atomic_units)
                logical_id = logical_chunk_id(metadata.document_id, unit_interval, draft.split_index)
                content_hash = sha256_text(draft.content)
                cid = chunk_id(metadata.document_version_id, logical_id, content_hash)
                child_ids.append(cid)
                staged.append((draft, logical_id, cid))
                first_unit = draft.atomic_units[0].logical_unit_id
                continuation_roots.setdefault(first_unit, cid)
            for draft, logical_id, cid in staged:
                pages = [span.page for span in draft.source_spans if span.page is not None]
                flags = list(draft.quality_flags)
                if has_pages and not pages:
                    flags.append("missing_page_locator")
                if len(set(pages)) > 1:
                    flags.append("cross_page_split")
                first_unit = draft.atomic_units[0].logical_unit_id
                continuation = draft.split_index > 1
                chunks.append(
                    ChildChunk(
                        chunk_id=cid,
                        logical_chunk_id=logical_id,
                        document_id=metadata.document_id,
                        document_version_id=metadata.document_version_id,
                        parent_chunk_id=parent_chunk_id,
                        article_root_id=article_root_id,
                        chunk_type="table_row_group" if draft.table else ("clause_group" if len(draft.atomic_units) > 1 else "atomic_unit"),
                        node_type="table" if draft.table else draft.node.node_type,
                        document_title=metadata.title,
                        short_title=metadata.short_title,
                        source_url=metadata.source_url,
                        source_file=metadata.source_file,
                        hierarchy=draft.hierarchy,
                        canonical_locator=draft.canonical_locator,
                        atomic_unit_ids=[atom.logical_unit_id for atom in draft.atomic_units],
                        atomic_units=atomic_models(draft.atomic_units),
                        content=draft.content,
                        embedding_text=draft.embedding_text,
                        parent_lead_in=draft.parent_lead_in,
                        page_start=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                        source_spans=draft.source_spans,
                        relations=Relations(
                            parent=parent_chunk_id,
                            annex_parent=(
                                parent_chunk_id
                                if root.node_type == "annex"
                                or any(node.node_type == "annex" for node in root.ancestors())
                                else None
                            ),
                            table_parent=parent_chunk_id if draft.table else None,
                        ),
                        split_index=draft.split_index,
                        split_count=draft.split_count,
                        is_continuation=continuation,
                        continuation_of=continuation_roots[first_unit] if continuation else None,
                        embedding_token_count=self.tokenizer.count(draft.embedding_text),
                        tokenizer_id=self.tokenizer.tokenizer_id,
                        character_count=len(draft.content),
                        sentence_count=sentence_count(draft.content),
                        packing_reason=draft.packing_reason,
                        content_sha256=sha256_text(draft.content),
                        quality=Quality(
                            parser_confidence=0.75 if flags else 0.98,
                            quality_flags=sorted(set(flags)),
                            normalizations=[],
                        ),
                        table=draft.table,
                    )
                )
            parent_content = "\n\n".join(draft.content for draft in group)
            spans = [span for draft in group for span in draft.source_spans]
            pages = [span.page for span in spans if span.page is not None]
            parent_tokens = self.tokenizer.count(parent_content)
            if parent_tokens > self.config.max_parent_tokens:
                raise ValueError(f"parent token limit exceeded: {parent_tokens}")
            parents.append(
                ParentChunk(
                    parent_chunk_id=parent_chunk_id,
                    article_root_id=article_root_id,
                    document_id=metadata.document_id,
                    document_version_id=metadata.document_version_id,
                    document_title=metadata.title,
                    parent_type=root.node_type,
                    hierarchy=root.hierarchy(),
                    canonical_locator=root_locator,
                    title=root.title,
                    content=parent_content,
                    parent_lead_in=root.own_content if root.children else None,
                    child_chunk_ids=child_ids,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    source_spans=spans,
                    token_count=parent_tokens,
                    content_sha256=sha256_text(parent_content),
                    quality=Quality(parser_confidence=0.98, quality_flags=[], normalizations=[]),
                )
            )
        return parents, chunks

    def _draft_tokens(self, drafts: list[ChunkDraft]) -> int:
        return self.tokenizer.count("\n\n".join(draft.content for draft in drafts))

    @staticmethod
    def _add_special_relations(chunks: list[ChildChunk], profiles: list[str]) -> None:
        if "DECISION_OR_AMENDMENT" not in profiles or "REGULATION" not in profiles:
            return
        approved = next((c for c in chunks if c.node_type in {"section", "clause"}), None)
        if approved:
            for chunk in chunks:
                if chunk.node_type == "decision_clause":
                    chunk.relations.approved_document = approved.parent_chunk_id

    @staticmethod
    def _mark_duplicate_chunks(chunks: list[ChildChunk]) -> int:
        by_hash: dict[str, list[ChildChunk]] = defaultdict(list)
        for chunk in chunks:
            by_hash[chunk.content_sha256].append(chunk)
        duplicates = 0
        for group in by_hash.values():
            if len(group) < 2:
                continue
            locators = {chunk.canonical_locator for chunk in group}
            if len(locators) == len(group):
                for chunk in group:
                    chunk.quality.quality_flags.append("duplicate_content")
                duplicates += len(group) - 1
        return duplicates

    @staticmethod
    def _mark_duplicate_documents(processed: list[ProcessedDocument]) -> None:
        groups: dict[str, list[ProcessedDocument]] = defaultdict(list)
        for item in processed:
            groups[item.canonical_content_hash].append(item)
        for group in groups.values():
            if len(group) < 2:
                continue
            # Canonical copy is the lexicographically smallest NFC relative source path.
            canonical = min(group, key=lambda item: unicodedata.normalize("NFC", item.metadata.source_relative_path))
            for item in group:
                if item is canonical:
                    continue
                item.metadata.duplicate_of_document_id = canonical.metadata.document_id
                item.metadata.is_canonical_document = False

    def _validate_document(self, parents: list[ParentChunk], chunks: list[ChildChunk]) -> None:
        parent_map = {parent.parent_chunk_id: parent for parent in parents}
        child_map = {chunk.chunk_id: chunk for chunk in chunks}
        if len(parent_map) != len(parents) or len(child_map) != len(chunks):
            raise ValueError("identifier collision within document")
        for chunk in chunks:
            if chunk.parent_chunk_id not in parent_map:
                raise ValueError(f"orphan child: {chunk.chunk_id}")
            if chunk.embedding_token_count > self.config.max_child_tokens:
                raise ValueError(
                    f"child token limit exceeded ({chunk.embedding_token_count}): {chunk.chunk_id}"
                )
            if "<!-- PAGE:" in chunk.content:
                raise ValueError(f"page marker leaked into content: {chunk.chunk_id}")
            if not chunk.canonical_locator or not chunk.embedding_text.strip():
                raise ValueError(f"chunk lacks locator or embedding text: {chunk.chunk_id}")
        for parent in parents:
            if not parent.child_chunk_ids or any(cid not in child_map for cid in parent.child_chunk_ids):
                raise ValueError(f"orphan parent: {parent.parent_chunk_id}")
            if parent.token_count > self.config.max_parent_tokens:
                raise ValueError(f"parent token limit exceeded: {parent.parent_chunk_id}")
        for chunk in chunks:
            previous = chunk.relations.previous_sibling
            following = chunk.relations.next_sibling
            for sibling_id in (previous, following):
                if sibling_id and child_map[sibling_id].parent_chunk_id != chunk.parent_chunk_id:
                    raise ValueError("sibling relation crossed a parent boundary")

    @staticmethod
    def _validate_corpus(processed: list[ProcessedDocument], manifest: Manifest) -> None:
        document_ids = [item.metadata.document_id for item in processed]
        chunk_ids = [chunk.chunk_id for item in processed for chunk in item.chunks]
        parent_ids = [parent.parent_chunk_id for item in processed for parent in item.parents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("corpus document identifier collision")
        if len(chunk_ids) != len(set(chunk_ids)) or len(parent_ids) != len(set(parent_ids)):
            raise ValueError("corpus chunk identifier collision")
        if manifest.discovered_document_count != manifest.successful_document_count + manifest.failed_document_count:
            raise ValueError("manifest does not account for every discovered document")

    def _manifest(
        self, paths: list[Path], processed: list[ProcessedDocument], failures: list[dict[str, str]]
    ) -> Manifest:
        chunks = [chunk for item in processed for chunk in item.chunks]
        parents = [parent for item in processed for parent in item.parents]
        tokens = [chunk.embedding_token_count for chunk in chunks]
        profile_counts = Counter(item.profile for item in processed)
        chunk_types = Counter(chunk.chunk_type for chunk in chunks)
        flags = Counter(flag for item in processed for flag in item.report.quality_flags)
        source_manifest = [
            {
                "source_relative_path": path.relative_to(self.config.input_root.resolve()).as_posix(),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ]
        corpus_hash = sha256_text(json.dumps(source_manifest, ensure_ascii=False, sort_keys=True))[:16]
        return Manifest(
            corpus_version=f"v2-{corpus_hash}",
            chunker_name="adaptive-structure-aware-hierarchical",
            chunker_version="2.0.0",
            parser_name="reguaz-structure-parser",
            parser_version="2.0.0",
            tokenizer_id=self.tokenizer.tokenizer_id,
            approximate_tokenizer=self.tokenizer.approximate,
            chunk_size_configuration={
                "minimum_target": self.config.min_child_tokens,
                "preferred_min": self.config.preferred_child_min,
                "preferred_max": self.config.preferred_child_max,
                "hard_maximum": self.config.max_child_tokens,
                "parent_preferred_min": self.config.preferred_parent_min,
                "parent_preferred_max": self.config.preferred_parent_max,
                "parent_maximum": self.config.max_parent_tokens,
            },
            input_root=str(self.config.input_root.resolve()),
            output_root=str(self.config.output_root.resolve()),
            discovered_document_count=len(paths),
            successful_document_count=len(processed),
            failed_document_count=len(failures),
            duplicate_document_count=sum(not item.metadata.is_canonical_document for item in processed),
            parent_count=len(parents),
            child_count=len(chunks),
            chunk_type_distribution=dict(sorted(chunk_types.items())),
            document_profile_distribution=dict(sorted(profile_counts.items())),
            token_distribution={
                "minimum": min(tokens, default=0),
                "median": int(statistics.median(tokens)) if tokens else 0,
                "p95": self._percentile(tokens, 0.95),
                "maximum": max(tokens, default=0),
            },
            unresolved_reference_count=sum(item.report.unresolved_reference_count for item in processed),
            unclassified_structure_count=sum(item.report.unclassified_node_count for item in processed),
            quality_flag_distribution=dict(sorted(flags.items())),
            source_manifest=source_manifest,
            failures=failures,
            build_timestamp=datetime.now(timezone.utc),
        )

    @staticmethod
    def _quality_errors(report: QualityReport) -> set[str]:
        return set(report.quality_flags) & {
            "low_text_quality",
            "broken_numbering",
            "oversized_atomic_unit",
            "missing_page_locator",
            "parser_low_confidence",
        }

    @staticmethod
    def _failed_report(discovery: Discovery, doc_id: str, error: str) -> QualityReport:
        return QualityReport(
            document_id=doc_id,
            status="failed",
            profile=discovery.profile,
            profiles_detected=discovery.profiles,
            parent_count=0,
            child_count=0,
            atomic_unit_count=0,
            table_count=0,
            unclassified_node_count=0,
            unresolved_reference_count=0,
            duplicate_chunk_count=0,
            min_tokens=0,
            median_tokens=0,
            p95_tokens=0,
            max_tokens=0,
            quality_flags=[],
            errors=[error],
        )

    @staticmethod
    def _percentile(values: list[int], quantile: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile + 0.999999)))
        return ordered[index]

    @staticmethod
    def _titles(path: Path, text: str) -> tuple[str, str]:
        stem = unicodedata.normalize("NFC", path.stem).strip("_ “”.")
        if not re.fullmatch(r"[0-9a-f-]{20,}", stem, re.I):
            title = stem
        else:
            lines = [
                re.sub(r"^#+\s*|\[\d+\]\s*$", "", line).strip()
                for line in text.splitlines()[:30]
                if line.strip() and "<!-- PAGE:" not in line and line.strip() != "---"
            ]
            candidates = [line for line in lines if 12 <= len(line) <= 240]
            title = max(candidates or lines or [stem], key=len)
        short = re.sub(r"^(?:“|\")|(?:”|\")$", "", title).strip()
        short = re.sub(r"\s+Azərbaycan Respublikasının.*$", "", short, flags=re.I) or title
        return title, short[:160]

    @staticmethod
    def _act_number(text: str) -> str | None:
        match = re.search(r"(?m)^\s*(?:№|Qərar\s*№)\s*([\w/-]+)\s*$", text[:5000], re.I)
        return match.group(1) if match else None
