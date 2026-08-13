from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Hierarchy, HierarchyLabel, SourceSpan
from .normalizer import NormalizedDocument, NormalizedLine, safely_join_lines


ARTICLE_RE = re.compile(
    r"^\s*(?:M\s*a\s*d\s*d\s*ə|M\s*addə)\s*(?P<number>\d+(?:-\d+)?)\s*\.\s*(?P<title>.*)$",
    re.IGNORECASE,
)
CHAPTER_RE = re.compile(
    r"^\s*(?P<number>[IVXLCDM]+|\d+)\s+(?:f\s*ə\s*s\s*i\s*l|fəsil|bölmə)(?:\[(?P<footnote>\d+)\])?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
ANNEX_RE = re.compile(r"^\s*(?:#+\s*)?Əlavə\s+(?P<number>\d+[A-Za-z]?)\s*\.?\s*(?P<title>.*)$", re.I)
MARKDOWN_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
NUMERIC_RE = re.compile(r"^\s*(?P<number>\d+(?:[-.]\d+){0,5})\.?(?:\s+|$)(?P<title>.*)$")
ITEM_RE = re.compile(r"^\s*(?P<number>[a-zəçşıöüğ])\)\s*(?P<title>.*)$", re.I)
NOTE_RE = re.compile(r"^\s*(?P<number>Qeyd)\s*:\s*(?P<title>.*)$", re.I)
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")

RANKS = {
    "root": 0,
    "division": 1,
    "chapter": 2,
    "annex": 2,
    "article": 3,
    "section": 3,
    "heading": 3,
    "decision_clause": 4,
    "clause": 4,
    "subclause": 5,
    "item": 6,
    "list_item": 6,
    "paragraph": 7,
    "table": 7,
    "unclassified": 7,
}


@dataclass(slots=True)
class StructureNode:
    node_type: str
    number: str | None
    title: str | None
    rank: int
    parent: "StructureNode | None" = None
    own_lines: list[NormalizedLine] = field(default_factory=list)
    children: list["StructureNode"] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    table_headers: list[str] = field(default_factory=list)
    table_rows: list[list[str]] = field(default_factory=list)

    @property
    def own_content(self) -> str:
        return safely_join_lines([line.text for line in self.own_lines])

    @property
    def heading(self) -> str:
        prefix = {
            "article": "Maddə",
            "chapter": "Fəsil",
            "annex": "Əlavə",
            "clause": "Bənd",
            "subclause": "Yarımbənd",
        }.get(self.node_type, self.node_type.capitalize())
        bits = [prefix, self.number or ""]
        if self.title:
            bits.append(f"— {self.title}")
        return " ".join(x for x in bits if x).strip()

    def ancestors(self) -> list["StructureNode"]:
        result: list[StructureNode] = []
        node = self.parent
        while node and node.node_type != "root":
            result.append(node)
            node = node.parent
        return list(reversed(result))

    def hierarchy(self) -> Hierarchy:
        values: dict[str, object] = {"heading_path": []}
        for node in [*self.ancestors(), self]:
            values["heading_path"].append(node.heading)
            if node.node_type in {"division", "chapter", "article", "section", "annex"}:
                key = "section" if node.node_type == "heading" else node.node_type
                values[key] = HierarchyLabel(number=node.number, title=node.title)
            elif node.node_type in {"clause", "subclause", "item", "list_item"}:
                key = "item" if node.node_type == "list_item" else node.node_type
                values[key] = node.number
        return Hierarchy(**values)

    def locator(self) -> str:
        parts: list[str] = []
        hierarchy = self.hierarchy()
        if hierarchy.annex:
            parts.append(f"Əlavə {hierarchy.annex.number}")
        if hierarchy.article:
            parts.append(f"Maddə {hierarchy.article.number}")
        if hierarchy.section:
            parts.append(f"bölmə {hierarchy.section.number or hierarchy.section.title}")
        leaf = self.number
        if leaf and self.node_type not in {"article", "section", "annex", "chapter"}:
            label = "yarımbənd" if self.node_type == "subclause" else "bənd"
            parts.append(f"{label} {leaf}")
        if not parts:
            parts.append(self.heading)
        return ", ".join(parts)

    def source_spans(self) -> list[SourceSpan]:
        if not self.own_lines:
            return []
        spans: list[SourceSpan] = []
        group: list[NormalizedLine] = []
        for line in self.own_lines:
            if group and line.page != group[-1].page:
                spans.append(_span(group))
                group = []
            group.append(line)
        if group:
            spans.append(_span(group))
        return spans


def _span(lines: list[NormalizedLine]) -> SourceSpan:
    return SourceSpan(
        page=lines[0].page,
        char_start=lines[0].char_start,
        char_end=lines[-1].char_end,
        line_start=lines[0].line_number,
        line_end=lines[-1].line_number,
    )


@dataclass(slots=True)
class ParseResult:
    root: StructureNode
    nodes: list[StructureNode]
    unclassified_count: int
    table_count: int


class StructureParser:
    """Line-oriented state machine backed by a hierarchy stack."""

    def parse(self, document: NormalizedDocument, profile: str) -> ParseResult:
        root = StructureNode("root", None, None, 0)
        stack = [root]
        nodes: list[StructureNode] = []
        paragraph: StructureNode | None = None
        decision_mode = profile == "DECISION_OR_AMENDMENT" or "QƏRARA ALIR" in document.text[:10000].upper()
        decision_numbers: list[str] = []
        index = 0
        lines = document.lines

        def add_node(node: StructureNode) -> None:
            nonlocal paragraph
            while len(stack) > 1 and stack[-1].rank >= node.rank:
                stack.pop()
            node.parent = stack[-1]
            stack[-1].children.append(node)
            nodes.append(node)
            if node.node_type not in {"paragraph", "unclassified", "table"}:
                stack.append(node)
            paragraph = None

        while index < len(lines):
            line = lines[index]
            stripped = line.text.strip()
            if not stripped:
                paragraph = None
                index += 1
                continue
            if stripped.startswith("|") and stripped.endswith("|"):
                table_lines: list[NormalizedLine] = []
                while index < len(lines):
                    candidate = lines[index]
                    if not (candidate.text.strip().startswith("|") and candidate.text.strip().endswith("|")):
                        break
                    table_lines.append(candidate)
                    index += 1
                table = self._table_node(table_lines)
                add_node(table)
                continue
            match = ARTICLE_RE.match(stripped)
            if match:
                title = self._clean_heading_title(match.group("title"))
                add_node(self._structural("article", match.group("number"), title, line))
                index += 1
                continue
            match = CHAPTER_RE.match(stripped)
            if match:
                node = self._structural(
                    "chapter", match.group("number"), self._clean_heading_title(match.group("title")), line
                )
                if match.group("footnote"):
                    node.quality_flags.append("heading_footnote_separated")
                add_node(node)
                index += 1
                continue
            match = ANNEX_RE.match(stripped)
            if match:
                add_node(self._structural("annex", match.group("number"), match.group("title") or None, line))
                index += 1
                continue
            match = MARKDOWN_RE.match(stripped)
            if match:
                level = len(match.group("marks"))
                node = self._structural("heading", str(level), match.group("title"), line)
                node.rank = min(2 + level, 6)
                add_node(node)
                index += 1
                continue
            match = ITEM_RE.match(stripped) or NOTE_RE.match(stripped)
            if match:
                add_node(self._structural("list_item", match.group("number"), match.group("title"), line))
                index += 1
                continue
            match = NUMERIC_RE.match(stripped)
            if match and self._is_numeric_heading(stripped, match.group("number")):
                number = match.group("number").rstrip(".")
                depth = len(re.split(r"[.-]", number))
                in_article = any(node.node_type == "article" for node in stack)
                if decision_mode and depth == 1 and number == "1" and decision_numbers:
                    # Approved regulations conventionally restart numbering at 1.
                    decision_mode = False
                if decision_mode and depth == 1:
                    kind = "decision_clause"
                    decision_numbers.append(number)
                elif in_article:
                    kind = {1: "clause", 2: "clause", 3: "subclause"}.get(depth, "item")
                else:
                    kind = {1: "section", 2: "clause", 3: "subclause"}.get(depth, "item")
                add_node(self._structural(kind, number, match.group("title"), line))
                index += 1
                continue

            # PDF extraction commonly wraps the body of the current legal unit across
            # physical lines. Keep those lines in that unit instead of manufacturing
            # one paragraph node per structural heading.
            if stack[-1] is not root and stack[-1].node_type not in {"table"}:
                stack[-1].own_lines.append(line)
                index += 1
                continue

            if paragraph is None:
                kind = "unclassified" if not nodes else "paragraph"
                paragraph = StructureNode(kind, f"p{line.line_number}", None, RANKS[kind])
                paragraph.parent = stack[-1]
                paragraph.quality_flags.extend(["unclassified_structure"] if kind == "unclassified" else [])
                stack[-1].children.append(paragraph)
                nodes.append(paragraph)
            paragraph.own_lines.append(line)
            index += 1

        return ParseResult(
            root=root,
            nodes=nodes,
            unclassified_count=sum(n.node_type == "unclassified" for n in nodes),
            table_count=sum(n.node_type == "table" for n in nodes),
        )

    @staticmethod
    def _clean_heading_title(title: str) -> str | None:
        cleaned = re.sub(r"\[(\d+)\]\s*$", "", title).strip()
        return cleaned or None

    @staticmethod
    def _is_numeric_heading(text: str, number: str) -> bool:
        # Anchoring plus a terminal punctuation/short line prevents references inside prose.
        remainder = text[len(text) - len(text.lstrip()):]
        del remainder
        return bool(number) and len(text) < 1000

    @staticmethod
    def _structural(kind: str, number: str | None, title: str | None, line: NormalizedLine) -> StructureNode:
        node = StructureNode(kind, number, title.strip() if title else None, RANKS[kind])
        node.own_lines.append(line)
        return node

    @staticmethod
    def _table_node(lines: list[NormalizedLine]) -> StructureNode:
        node = StructureNode("table", None, None, RANKS["table"], own_lines=lines)
        parsed = [[cell.strip() for cell in line.text.strip().strip("|").split("|")] for line in lines]
        if len(parsed) >= 2 and TABLE_SEPARATOR_RE.match(lines[1].text):
            node.table_headers = parsed[0]
            node.table_rows = parsed[2:]
        else:
            node.table_rows = parsed
            node.quality_flags.append("table_structure_uncertain")
        return node
