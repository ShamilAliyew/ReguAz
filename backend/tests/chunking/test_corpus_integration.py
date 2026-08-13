from __future__ import annotations

from pathlib import Path

from backend.reguaz.services.chunking import ApproximateTokenizer, ChunkingConfig
from backend.reguaz.services.chunking.normalizer import normalize_for_parsing
from backend.reguaz.services.chunking.packer import AdaptivePacker
from backend.reguaz.services.chunking.parser import StructureParser


ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "data" / "processed" / "cleaned_documents"


def parse_file(relative: str, profile: str):
    path = CORPUS / relative
    text = path.read_text(encoding="utf-8")
    parsed = StructureParser().parse(normalize_for_parsing(text), profile)
    chunks = AdaptivePacker(ChunkingConfig(), ApproximateTokenizer()).pack(
        parsed, document_id="doc_integration", title=path.stem
    )
    return parsed, chunks


def test_representative_banking_law_spaced_articles_and_pages() -> None:
    parsed, chunks = parse_file("laws/Banklar Haqqında Qanun.md", "LEGAL_ACT")
    assert any(node.node_type == "article" and node.number == "1" for node in parsed.nodes)
    assert all("<!-- PAGE:" not in chunk.content for chunk in chunks)
    assert any(span.page for chunk in chunks for span in chunk.source_spans)


def test_representative_operational_risk_lead_in_and_annex_tables() -> None:
    parsed, chunks = parse_file(
        "risk_management/Əməliyyat Risklərinin İdarə Edilməsi Qaydası.md", "DECISION_OR_AMENDMENT"
    )
    node = next(node for node in parsed.nodes if node.number == "3.5.1")
    assert node.parent and node.parent.number == "3.5"
    child = next(chunk for chunk in chunks if any(atom.node.number == "3.5.1" for atom in chunk.atomic_units))
    assert child.parent_lead_in and "aşağıdakı" in child.parent_lead_in
    assert any(chunk.table for chunk in chunks)


def test_representative_methodology_numeric_headings_and_tables() -> None:
    parsed, chunks = parse_file(
        "guidance_and_methodology/feeb62eb12b5c144caa99818e.md", "GUIDANCE"
    )
    assert any(node.node_type in {"section", "clause"} for node in parsed.nodes)
    assert any(node.node_type == "table" for node in parsed.nodes)
    assert all(chunk.canonical_locator and chunk.embedding_text for chunk in chunks)
    assert all(chunk.token_count <= 450 for chunk in chunks)
