# ReguAz V2 chunking and metadata pipeline

V2 implements **Adaptive Structure-Aware Hierarchical Chunking with Child-Level Retrieval and Relation-Aware Context Expansion**. It reads cleaned Markdown only and writes an isolated, citation-ready corpus. It does not extract PDFs, create embeddings, or access a vector database.

## Flow

1. Discover actual Unicode filesystem entries under `data/processed/cleaned_documents`.
2. Normalize Unicode NFC and line endings at parse time; consume page markers without rewriting input.
3. Classify deterministic `LEGAL_ACT`, `REGULATION`, `GUIDANCE`, `TABLE_OR_FORM`, and `DECISION_OR_AMENDMENT` profiles.
4. Parse with a state machine and hierarchy stack, including mixed profiles, tables, and annexes.
5. form atomic legal units, pack only short siblings under one structural parent, and sentence-split oversized units.
6. Create non-embedded parents and enriched child `embedding_text` with deterministic IDs and relations.
7. Strictly validate Pydantic schemas and parent/child, token, identifier, locator, and page-marker invariants.
8. Atomically publish `manifest.json` plus one document directory per ASCII-safe `document_id`.

## Commands

From the repository root, with the real BGE-M3 tokenizer:

```bash
HF_HUB_OFFLINE=1 poetry run python scripts/run_chunking_v2.py --dry-run --local-files-only
HF_HUB_OFFLINE=1 poetry run python scripts/run_chunking_v2.py --local-files-only
```

The real tokenizer is the default. It raises a clear error when unavailable. The deterministic approximate tokenizer is intended for offline validation and must be explicitly enabled; its use is recorded in the manifest:

```bash
poetry run python scripts/run_chunking_v2.py --dry-run --tokenizer approximate --allow-approx-tokenizer
poetry run python scripts/run_chunking_v2.py --tokenizer approximate --allow-approx-tokenizer
```

Options include `--input`, `--output`, `--document`, `--limit`, `--dry-run`, `--force`, `--fail-on-quality-errors`, `--tokenizer`, and `--allow-approx-tokenizer`. Existing V2 output is never overwritten without `--force`; the storage guard permits replacement only at an exact `*/data/processed/v2`-style target.

## Outputs and schemas

Each `data/processed/v2/documents/<document_id>/` contains `document.json`, `parents.jsonl`, `chunks.jsonl`, and `quality_report.json`. The global manifest records tokenizer mode, source hashes, corpus/chunk/profile distributions, failures, duplicates, relations, and quality totals. All schemas use Pydantic V2 with unknown fields forbidden.

Identifiers are prefixed SHA-256 values. Legal identity produces `document_id`; canonical content produces `document_version_id`; legal locators produce `logical_unit_id` and `logical_chunk_id`; version, logical ID, and content hash produce `chunk_id`. Timestamps and processing order never affect IDs. When multiple input copies resolve to the same legal identity, a deterministic relative-path discriminator gives each stored copy a distinct ID. Exact-duplicate canonical selection uses the lexicographically smallest NFC-normalized relative source path.

Children target 180–320 tokens and never exceed 450. Leaves below 60 tokens may be packed only with direct siblings under the same article/section parent. Oversized units split at sentence or subordinate punctuation boundaries without blanket overlap. Parent segments target 600–1200 and never exceed 1500 tokens. Table row groups preserve whole rows and repeat column headers in `embedding_text`.

Relations include parent/children, previous/next sibling, resolved or retained-unresolved legal references, annex parent, table parent, and approved-document links. Quality flags include text/structure/numbering/page/table/reference/duplicate/identity issues. Page and character/line source spans, logical units, hashes, document title/version, and source URL are available for later deterministic citations.

V1 directories are never write targets. The later embedding stage should enumerate only V2 `chunks.jsonl`, validate `schema_version == "2.0"`, embed `embedding_text`, retain `content` for display, and use `chunk_id` as the versioned retrieval key. Parent rows must not be embedded.
