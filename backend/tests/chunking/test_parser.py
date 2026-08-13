from __future__ import annotations

import unicodedata

import pytest

from backend.reguaz.services.chunking.normalizer import normalize_for_parsing
from backend.reguaz.services.chunking.parser import StructureParser


def parse(text: str, profile: str = "LEGAL_ACT"):
    return StructureParser().parse(normalize_for_parsing(text), profile)


@pytest.mark.parametrize(
    ("heading", "node_type", "number"),
    [
        ("Maddə 1. Başlıq", "article", "1"),
        ("M a d d ə 1 . Başlıq", "article", "1"),
        ("Maddə 7-1. Başlıq", "article", "7-1"),
        ("I fəsil", "chapter", "I"),
        ("VII fəsil[213]", "chapter", "VII"),
        ("1.1. Mətn", "clause", "1.1"),
        ("1.1.1. Mətn", "subclause", "1.1.1"),
        ("1-1.2. Mətn", "subclause", "1-1.2"),
        ("a) Mətn", "list_item", "a"),
        ("b) Mətn", "list_item", "b"),
    ],
)
def test_structural_variants(heading: str, node_type: str, number: str) -> None:
    result = parse(heading, "REGULATION")
    node = result.nodes[0]
    assert (node.node_type, node.number) == (node_type, number)


def test_article_reference_inside_sentence_is_not_heading() -> None:
    result = parse("Bu Qanunun Maddə 12 müddəası tətbiq olunur.")
    assert all(node.node_type != "article" for node in result.nodes)


def test_parent_lead_in_hierarchy() -> None:
    result = parse("1. Bölmə\n3.5. Sistem azı aşağıdakıları əhatə edir:\n3.5.1. siyasət;\n3.5.2. nəzarət.", "REGULATION")
    subclauses = [node for node in result.nodes if node.node_type == "subclause"]
    assert [node.number for node in subclauses] == ["3.5.1", "3.5.2"]
    assert all(node.parent and node.parent.number == "3.5" for node in subclauses)


def test_page_markers_removed_and_spans_preserved() -> None:
    result = parse("<!-- PAGE: 3 -->\nMaddə 1. Başlıq\nBirinci sətir\n<!-- PAGE: 4 -->\n---\nikinci sətir")
    article = next(node for node in result.nodes if node.node_type == "article")
    assert "PAGE" not in article.own_content
    assert "Birinci sətir ikinci sətir" in article.own_content
    assert [span.page for span in article.source_spans()] == [3, 4]
    assert article.source_spans()[0].line_start == 2


def test_markdown_heading_table_and_annex() -> None:
    result = parse("# Rəhbərlik\nƏlavə 2\n## Cədvəl\n| A | B |\n| --- | --- |\n| 1 | 2 |", "GUIDANCE")
    assert any(node.node_type == "heading" for node in result.nodes)
    table = next(node for node in result.nodes if node.node_type == "table")
    assert table.table_headers == ["A", "B"]
    assert table.table_rows == [["1", "2"]]
    assert any(ancestor.node_type == "annex" for ancestor in table.ancestors())


def test_unicode_nfc_normalization() -> None:
    nfd = unicodedata.normalize("NFD", "Maddə 1. Ümumi")
    normalized = normalize_for_parsing(nfd)
    assert normalized.lines[0].text == unicodedata.normalize("NFC", nfd)
    assert "unicode_nfc" in normalized.normalizations
