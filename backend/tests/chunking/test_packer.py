from __future__ import annotations

from backend.reguaz.services.chunking import ApproximateTokenizer, ChunkingConfig
from backend.reguaz.services.chunking.normalizer import normalize_for_parsing
from backend.reguaz.services.chunking.packer import AdaptivePacker
from backend.reguaz.services.chunking.parser import StructureParser


def packed(text: str, *, config: ChunkingConfig | None = None):
    tokenizer = ApproximateTokenizer()
    config = config or ChunkingConfig()
    parsed = StructureParser().parse(normalize_for_parsing(text), "REGULATION")
    return AdaptivePacker(config, tokenizer).pack(parsed, document_id="doc_test", title="Test")


def test_short_direct_siblings_pack_but_different_parents_do_not() -> None:
    text = "1. Bir\n1.1. qısa mətn.\n1.2. digər qısa mətn.\n2. İki\n2.1. üçüncü qısa mətn."
    chunks = packed(text)
    combined = [chunk for chunk in chunks if len(chunk.atomic_units) > 1]
    assert any({a.node.number for a in chunk.atomic_units} == {"1.1", "1.2"} for chunk in combined)
    assert not any({a.node.number for a in chunk.atomic_units} == {"1.2", "2.1"} for chunk in combined)


def test_oversized_leaf_sentence_split_without_blanket_overlap() -> None:
    sentence = " ".join(["söz"] * 35) + "."
    text = "1. Bölmə\n1.1. " + " ".join([sentence] * 8)
    config = ChunkingConfig(
        min_child_tokens=10,
        preferred_child_min=20,
        preferred_child_max=40,
        max_child_tokens=55,
        preferred_parent_min=80,
        preferred_parent_max=120,
        max_parent_tokens=180,
    )
    chunks = packed(text, config=config)
    splits = [chunk for chunk in chunks if chunk.split_count > 1]
    assert len(splits) > 1
    assert all(chunk.token_count <= 55 for chunk in splits)
    contents = [chunk.content for chunk in splits]
    assert all(contents[index] != contents[index + 1] for index in range(len(contents) - 1))
    assert [chunk.split_index for chunk in splits] == list(range(1, len(splits) + 1))


def test_table_row_grouping_repeats_headers_in_embedding() -> None:
    rows = "\n".join(f"| {i} | {'word ' * 20} |" for i in range(1, 15))
    config = ChunkingConfig(
        min_child_tokens=10,
        preferred_child_min=30,
        preferred_child_max=70,
        max_child_tokens=100,
        preferred_parent_min=120,
        preferred_parent_max=180,
        max_parent_tokens=240,
    )
    chunks = packed(f"# Form\n| Kod | İzah |\n| --- | --- |\n{rows}", config=config)
    tables = [chunk for chunk in chunks if chunk.table]
    assert len(tables) > 1
    assert all("Column headers: Kod | İzah" in chunk.embedding_text for chunk in tables)
    assert all(chunk.table.row_start <= chunk.table.row_end for chunk in tables)


def test_parent_lead_in_is_preserved() -> None:
    chunks = packed("1. Bölmə\n3.5. Sistem aşağıdakılardan ibarətdir:\n3.5.1. siyasət.\n3.5.2. nəzarət.")
    children = [chunk for chunk in chunks if any(atom.node.number == "3.5.1" for atom in chunk.atomic_units)]
    assert children
    assert children[0].parent_lead_in
    assert "Sistem aşağıdakılardan" in children[0].parent_lead_in
