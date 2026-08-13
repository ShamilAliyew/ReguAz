from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


PAGE_RE = re.compile(r"<!--\s*PAGE:\s*(\d+)\s*-->", re.IGNORECASE)


@dataclass(slots=True)
class NormalizedLine:
    text: str
    page: int | None
    char_start: int
    char_end: int
    line_number: int
    normalizations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NormalizedDocument:
    text: str
    lines: list[NormalizedLine]
    normalizations: list[str]
    has_page_markers: bool


def canonical_content(text: str) -> str:
    """Canonical full text used for exact duplicates and version identifiers."""
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def normalize_for_parsing(text: str) -> NormalizedDocument:
    original = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFC", original)
    global_norms: list[str] = []
    if normalized != original:
        global_norms.append("unicode_nfc")
    if text != original:
        global_norms.append("line_endings")

    lines: list[NormalizedLine] = []
    current_page: int | None = None
    offset = 0
    pending_page_separator = False
    has_markers = False
    raw_lines = normalized.splitlines(keepends=True)
    for index, raw in enumerate(raw_lines, start=1):
        without_newline = raw.rstrip("\r\n")
        markers = PAGE_RE.findall(without_newline)
        line_norms: list[str] = []
        if markers:
            current_page = int(markers[-1])
            has_markers = True
            without_newline = PAGE_RE.sub("", without_newline).strip()
            line_norms.append("page_marker_removed")
            pending_page_separator = not without_newline
        if pending_page_separator and without_newline.strip() == "---":
            offset += len(raw)
            pending_page_separator = False
            global_norms.append("generated_page_separator_removed")
            continue
        if without_newline.strip():
            pending_page_separator = False
        lines.append(
            NormalizedLine(
                text=without_newline,
                page=current_page,
                char_start=offset,
                char_end=offset + len(without_newline),
                line_number=index,
                normalizations=line_norms,
            )
        )
        offset += len(raw)
    return NormalizedDocument(
        text=normalized,
        lines=lines,
        normalizations=sorted(set(global_norms)),
        has_page_markers=has_markers,
    )


def safely_join_lines(lines: list[str]) -> str:
    """Join PDF-wrapped prose while retaining Markdown/list/table structure."""
    result: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            result.append(" ".join(part.strip() for part in paragraph).strip())
            paragraph.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        structural = (
            stripped.startswith(("#", "|", "- ", "* ", ">"))
            or bool(re.match(r"^(?:\d+(?:[.-]\d+)*\.?|[a-zəçşıöüğ]\))\s+", stripped, re.I))
        )
        if structural:
            flush()
            result.append(stripped)
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(result).strip()
