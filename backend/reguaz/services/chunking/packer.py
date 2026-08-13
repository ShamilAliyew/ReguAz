from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .config import ChunkingConfig
from .identifiers import logical_unit_id, prefixed
from .models import AtomicUnit, Hierarchy, SourceSpan, TableMetadata
from .parser import ParseResult, StructureNode
from .tables import markdown_header, markdown_row, row_groups
from .tokenizer import TokenCounter


SENTENCE_RE = re.compile(r"(?<=[.!?…;:])\s+(?=[A-ZƏÖÜİŞÇĞ0-9])", re.UNICODE)


@dataclass(slots=True)
class AtomicDraft:
    node: StructureNode
    logical_unit_id: str
    locator: str
    content: str
    embedding_text: str
    parent_lead_in: str | None
    source_spans: list[SourceSpan]
    split_index: int = 1
    split_count: int = 1
    token_count: int = 0
    quality_flags: list[str] = field(default_factory=list)
    table: TableMetadata | None = None


@dataclass(slots=True)
class ChunkDraft:
    node: StructureNode
    atomic_units: list[AtomicDraft]
    content: str
    embedding_text: str
    hierarchy: Hierarchy
    canonical_locator: str
    parent_lead_in: str | None
    source_spans: list[SourceSpan]
    split_index: int
    split_count: int
    token_count: int
    packing_reason: str
    quality_flags: list[str]
    table: TableMetadata | None = None
    boundary: StructureNode | None = None


def sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?…]+(?:\s|$)", text)))


class AdaptivePacker:
    def __init__(self, config: ChunkingConfig, tokenizer: TokenCounter) -> None:
        self.config = config
        self.tokenizer = tokenizer

    def pack(self, parsed: ParseResult, *, document_id: str, title: str) -> list[ChunkDraft]:
        atoms: list[AtomicDraft] = []
        locator_occurrences: Counter[tuple[str, str]] = Counter()
        for node in parsed.nodes:
            occurrence_key = (node.node_type, node.locator())
            locator_occurrences[occurrence_key] += 1
            occurrence = locator_occurrences[occurrence_key]
            if node.node_type == "table":
                atoms.extend(self._table_atoms(node, document_id, title, occurrence))
            elif node.own_content and not (
                node.children
                and len(node.own_lines) == 1
                and node.node_type in {"chapter", "article", "section", "heading", "annex"}
            ):
                atoms.extend(self._node_atoms(node, document_id, title, occurrence))
        return self._pack_short_siblings(atoms, title)

    def _node_atoms(
        self, node: StructureNode, doc_id: str, title: str, occurrence: int
    ) -> list[AtomicDraft]:
        locator = node.locator()
        identity_locator = locator if occurrence == 1 else f"{locator} [occurrence {occurrence}]"
        unit_id = logical_unit_id(doc_id, node.node_type, identity_locator)
        lead_in = self._lead_in(node)
        prefix = self._embedding_prefix(title, node, self._embedding_lead_in(lead_in))
        parts, forced = self._split_to_limit(node.own_content, prefix)
        count = len(parts)
        result: list[AtomicDraft] = []
        for index, content in enumerate(parts, start=1):
            embedding = f"{prefix}\nText: {content}".strip()
            flags = list(node.quality_flags)
            if forced:
                flags.append("oversized_atomic_unit")
            result.append(
                AtomicDraft(
                    node=node,
                    logical_unit_id=unit_id,
                    locator=locator,
                    content=content,
                    embedding_text=embedding,
                    parent_lead_in=lead_in,
                    source_spans=node.source_spans(),
                    split_index=index,
                    split_count=count,
                    token_count=self.tokenizer.count(embedding),
                    quality_flags=sorted(set(flags)),
                )
            )
        return result

    def _table_atoms(
        self, node: StructureNode, doc_id: str, title: str, occurrence: int
    ) -> list[AtomicDraft]:
        locator = node.locator()
        table_id = prefixed("table", doc_id, locator, str(node.own_lines[0].line_number))
        table_title = self._table_title(node)
        prefix = self._embedding_prefix(title, node, None)
        if table_title:
            prefix += f"\nTable: {table_title}"
        table_prefix = f"{prefix}\nColumn headers: {' | '.join(node.table_headers)}"
        header = markdown_header(node.table_headers)
        budget = max(1, self.config.max_child_tokens - self.tokenizer.count(table_prefix) - 8)
        groups = row_groups(
            node.table_headers, node.table_rows, max_tokens=budget, count_tokens=self.tokenizer.count
        )
        if not groups:
            groups = [(1, 1, [row for row in node.table_rows if any(row)])]
        result: list[AtomicDraft] = []
        total_rows = max(1, len(node.table_rows))
        for group_index, (row_start, row_end, rows) in enumerate(groups, start=1):
            content = "\n".join(filter(None, [header, *(markdown_row(row) for row in rows)]))
            # A pathological single row is split deterministically; headers repeat.
            text_parts, forced = self._split_to_limit(content, table_prefix)
            for part_index, part in enumerate(text_parts, start=1):
                embedding = f"{table_prefix}\nText: {part}"
                split_count = len(text_parts)
                unit_locator = f"{locator}, sətirlər {row_start}-{row_end}"
                result.append(
                    AtomicDraft(
                        node=node,
                        logical_unit_id=logical_unit_id(
                            doc_id,
                            "table_row_group",
                            unit_locator if occurrence == 1 else f"{unit_locator} [occurrence {occurrence}]",
                        ),
                        locator=unit_locator,
                        content=part,
                        embedding_text=embedding,
                        parent_lead_in=None,
                        source_spans=node.source_spans(),
                        split_index=part_index,
                        split_count=split_count,
                        token_count=self.tokenizer.count(embedding),
                        quality_flags=sorted(set(node.quality_flags + (["oversized_atomic_unit"] if forced else []))),
                        table=TableMetadata(
                            table_id=table_id,
                            title=table_title,
                            column_headers=node.table_headers,
                            row_start=row_start,
                            row_end=row_end,
                            total_rows=total_rows,
                        ),
                    )
                )
        return result

    def _pack_short_siblings(self, atoms: list[AtomicDraft], title: str) -> list[ChunkDraft]:
        result: list[ChunkDraft] = []
        pending: list[AtomicDraft] = []

        def flush() -> None:
            nonlocal pending
            while pending:
                group = [pending.pop(0)]
                while pending and sum(a.token_count for a in group) < self.config.preferred_child_min:
                    candidate = group + [pending[0]]
                    embedding = self._packed_embedding(title, candidate)
                    if self.tokenizer.count(embedding) > self.config.max_child_tokens:
                        break
                    group.append(pending.pop(0))
                result.append(
                    self._to_chunk(
                        group,
                        "short_sibling_pack" if len(group) > 1 else "atomic_leaf",
                        title,
                    )
                )

        previous_key: tuple[int, int] | None = None
        for atom in atoms:
            boundary = self._boundary(atom.node)
            direct_parent = atom.node.parent or atom.node
            key = (id(boundary), id(direct_parent))
            eligible = atom.split_count == 1 and atom.token_count < self.config.min_child_tokens and atom.table is None
            if not eligible:
                flush()
                result.append(
                    self._to_chunk(
                        [atom],
                        "oversized_split" if atom.split_count > 1 else "atomic_leaf",
                        title,
                    )
                )
                previous_key = None
                continue
            if previous_key is not None and key != previous_key:
                flush()
            pending.append(atom)
            previous_key = key
        flush()
        return result

    def _to_chunk(self, atoms: list[AtomicDraft], reason: str, title: str) -> ChunkDraft:
        first = atoms[0]
        embedding = first.embedding_text if len(atoms) == 1 else self._packed_embedding(title, atoms)
        content = "\n\n".join(atom.content for atom in atoms)
        spans = [span for atom in atoms for span in atom.source_spans]
        return ChunkDraft(
            node=first.node,
            atomic_units=atoms,
            content=content,
            embedding_text=embedding,
            hierarchy=first.node.hierarchy(),
            canonical_locator=self._interval_locator(atoms),
            parent_lead_in=first.parent_lead_in,
            source_spans=spans,
            split_index=first.split_index,
            split_count=first.split_count,
            token_count=self.tokenizer.count(embedding),
            packing_reason=reason,
            quality_flags=sorted({flag for atom in atoms for flag in atom.quality_flags}),
            table=first.table,
            boundary=self._boundary(first.node),
        )

    def _packed_embedding(self, title: str, atoms: list[AtomicDraft]) -> str:
        hierarchy = "\n".join(atoms[0].node.hierarchy().heading_path)
        prefix = f"Document: {title}\nHierarchy: {hierarchy}".strip()
        return f"{prefix}\nText: " + "\n".join(atom.content for atom in atoms)

    def _split_to_limit(self, content: str, prefix: str) -> tuple[list[str], bool]:
        if self.tokenizer.count(f"{prefix}\nText: {content}") <= self.config.max_child_tokens:
            return [content], False
        budget = max(8, self.config.max_child_tokens - self.tokenizer.count(prefix) - 3)
        sentences = [x.strip() for x in SENTENCE_RE.split(content) if x.strip()]
        if len(sentences) == 1:
            sentences = [x.strip() for x in re.split(r"(?<=[,;:])\s+", content) if x.strip()]
        forced = False
        parts: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            if self.tokenizer.count(sentence) > budget:
                forced = True
                words = sentence.split()
                piece: list[str] = []
                for word in words:
                    if piece and self.tokenizer.count(" ".join([*piece, word])) > budget:
                        if current:
                            parts.append(" ".join(current))
                            current = []
                        parts.append(" ".join(piece))
                        piece = []
                    piece.append(word)
                sentence_pieces = [" ".join(piece)] if piece else []
                if current:
                    parts.append(" ".join(current))
                    current = []
                parts.extend(sentence_pieces)
            elif current and self.tokenizer.count(" ".join([*current, sentence])) > budget:
                parts.append(" ".join(current))
                current = [sentence]
            else:
                current.append(sentence)
        if current:
            parts.append(" ".join(current))
        return parts or [content], forced

    @staticmethod
    def _interval_locator(atoms: list[AtomicDraft]) -> str:
        if len(atoms) == 1:
            return atoms[0].locator
        return f"{atoms[0].locator} – {atoms[-1].locator}"

    @staticmethod
    def _boundary(node: StructureNode) -> StructureNode:
        candidates = [*node.ancestors(), node]
        for candidate in reversed(candidates):
            if candidate.node_type in {"article", "section", "heading", "annex", "table"}:
                return candidate
        return candidates[0] if candidates else node

    @staticmethod
    def _lead_in(node: StructureNode) -> str | None:
        parent = node.parent
        if parent and parent.node_type in {"clause", "subclause", "paragraph"} and parent.own_content:
            return parent.own_content
        return None

    def _embedding_lead_in(self, lead_in: str | None) -> str | None:
        """Keep the full lead-in in metadata, but bound contextual repetition."""
        if not lead_in:
            return None
        budget = min(120, self.config.max_child_tokens // 3)
        words = lead_in.split()
        kept: list[str] = []
        for word in words:
            if kept and self.tokenizer.count(" ".join([*kept, word])) > budget:
                break
            kept.append(word)
        return " ".join(kept)

    @staticmethod
    def _embedding_prefix(title: str, node: StructureNode, lead_in: str | None) -> str:
        hierarchy = node.hierarchy()
        lines = [f"Document: {title}"]
        labels = {
            "Chapter": hierarchy.chapter,
            "Article": hierarchy.article,
            "Section": hierarchy.section,
            "Annex": hierarchy.annex,
        }
        for label, value in labels.items():
            if value:
                rendered = " — ".join(x for x in (value.number, value.title) if x)
                lines.append(f"{label}: {rendered}")
        if hierarchy.clause:
            lines.append(f"Clause: {hierarchy.clause}")
        if hierarchy.subclause:
            lines.append(f"Subclause: {hierarchy.subclause}")
        if lead_in:
            lines.append(f"Parent lead-in: {lead_in}")
        return "\n".join(lines)

    @staticmethod
    def _table_title(node: StructureNode) -> str | None:
        parent = node.parent
        return parent.title if parent and parent.title else None


def atomic_models(atoms: list[AtomicDraft]) -> list[AtomicUnit]:
    return [
        AtomicUnit(
            logical_unit_id=atom.logical_unit_id,
            canonical_locator=atom.locator,
            source_spans=atom.source_spans,
        )
        for atom in atoms
    ]
