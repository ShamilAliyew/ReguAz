# ReguAz V2 dense+sparse embeddings and Qdrant ingestion

This stage consumes only validated child records from `data/processed/v2/documents/*/chunks.jsonl`. It embeds `embedding_text`; parent chunks, display `content`, document metadata, and quality reports are never embedded.

## Representation

`BAAI/bge-m3` is pinned to an immutable Hugging Face revision and executed through `FlagEmbedding`. The adapter resolves the real Hugging Face snapshot directory and requires both auxiliary head files before inference; this prevents `FlagEmbedding` from silently random-initializing the sparse head when only a Hub model ID is supplied. Each child produces one normalized 1024-dimensional float32 named vector (`dense`) and one learned lexical-weight sparse vector (`sparse`). ColBERT vectors are intentionally disabled. Queries must later use the same model revision and no query instruction.

The category placed in Qdrant payload is joined from the corresponding `document.json` and must be one of the eight cleaned-document category directory names. It is never inferred from text or a classifier.

## Artifacts

Embedding outputs remain inside the existing V2 data tree:

```text
data/processed/v2/embeddings/bge_m3/<corpus_version>/
├── manifest.json
└── shards/part-*.jsonl
```

Shards use deterministic chunk ordering, atomic file replacement, SHA-256 checksums, and resumable `.inprogress` checkpoints. A forced rebuild keeps the previous completed artifact until the new staging build passes validation.

```bash
poetry run python scripts/run_embedding_v2.py --device cpu
```

After the pinned model snapshot (including `sparse_linear.pt` and `colbert_linear.pt`) is cached, an offline run can use `HF_HUB_OFFLINE=1` together with `--local-files-only`.

Useful options include `--batch-size`, `--max-length`, `--shard-size`, `--dry-run`, `--force`, and `--no-resume`.

## Qdrant

Local V2 Qdrant storage also remains in the V2 tree at `data/processed/v2/qdrant`. V1 `data/qdrant` is never opened or modified. Each child is one point containing named `dense` and `sparse` vectors plus citation-ready payload. Physical collections are content/model-versioned; `reguaz_v2_current` is a Qdrant alias, not a directory and not duplicate data. It is switched only after point-count, collection-metadata, dense-search, and sparse-search validation succeeds.

```bash
poetry run python scripts/run_qdrant_ingestion_v2.py
```

The local embedded Qdrant mode supports the vectors, payload, queries, and aliases used here, but its payload indexes have no performance effect. A server Qdrant deployment should reuse the same collection schema to obtain filter-index acceleration, operational snapshots, and concurrent service access.

## Required invariants

- chunk, embedding, and Qdrant counts are identical;
- every `chunk_id` is unique and deterministically mapped to a Qdrant UUID5;
- every dense vector is finite, normalized, and exactly 1024 dimensions;
- sparse indices/weights are aligned, unique, finite, and positive;
- embedding input hashes still match current `embedding_text` values;
- Qdrant collection metadata matches corpus and model revisions;
- only the eight allowed categories reach payloads;
- no parent is embedded and no V1 path is written.
