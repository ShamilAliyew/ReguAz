# ReguAZ V2 hybrid retrieval, reranking and evaluation

This read-only stage consumes the existing V2 manifests, chunks, embeddings and `reguaz_v2_current` alias. It does not mutate V1 or protected V2 artifacts.

## Optimized architecture

```text
query → one BGE-M3 dense+sparse encoding
      ├─ one Qdrant batch: dense top 30 + learned sparse top 30 (CPU)
      └─ BM25 child-content top 30 (CPU), overlapped with Qdrant
                         ↓
             lean equal-weight RRF (k=60)
                         ↓
       hydrate reranker fields for fused top 15 only
                         ↓
        BAAI/bge-reranker-v2-m3 → final top 5
                         ↓
       one full-payload fetch for final 5 only
                         ↓
            resolve unique direct parents
```

`HybridV2Retriever` was updated in place. V1 retrievers and collections are not redirected. The shared `V2ArtifactStore` loads authoritative child records once per process and lazily reads only parent files needed by final results.

## Startup and query contracts

The chunk manifest determines corpus version and child count. The embedding manifest resolves the immutable BGE-M3 revision. Startup validates the alias and physical collection, corpus/model metadata, named `dense` and `sparse` vectors, 1024-dimensional cosine dense vectors, and exact point count. The local client is explicitly closed.

The existing adapter keeps separate document and query paths. A query produces normalized finite dense output and aligned, unique, positive learned-sparse weights in one model call, without an instruction. The pinned snapshot must contain `sparse_linear.pt` and `colbert_linear.pt`; blank queries and invalid outputs fail fast.

The query encoder and reranker prefer Apple MPS, then CUDA, with a safe CPU fallback. Runtime metadata records the actual devices. Qdrant, BM25, RRF, artifact lookup, parent lookup and metrics are CPU work.

## Lean retrieval, fusion and hydration

Dense and sparse source searches request only `chunk_id`; BM25 source results contain identity, rank and score. BM25 indexes only NFC-normalized/case-folded tokens from child `content`. It never reads parents or indexes `embedding_text`. All sources support `category`, `document_id`, and `is_current` filters; current records are the default.

RRF uses:

```text
rrf_score(chunk) += source_weight / (rrf_k + source_rank)
```

Weights remain `1.0`, `rrf_k=60`, and ties break by `chunk_id`. Lean candidates preserve point ID, source ranks/scores, retriever names and exact per-source RRF contributions. Missing sources remain `null`.

Only the fused top 15 are hydrated from the shared child lookup. Reranker text remains:

```text
Document: <document_title>
Locator: <canonical_locator>
Hierarchy: <compact hierarchy>
Content: <content>
```

The reranker is loaded once and runs in batches. After final top 5 selection, all five deterministic point IDs are retrieved in one Qdrant request and their payload `chunk_id` values are validated. Direct parents are then resolved once per unique ID. Missing payloads fail clearly; missing parents produce structured warnings. Parent text is not used for reranking.

## Concurrency strategy

Three schedules were measured with shared loaded models:

1. Sequential dense → sparse → BM25.
2. One dense+sparse Qdrant batch overlapped with BM25.
3. Three concurrent dense, sparse and BM25 calls.

`qdrant_batch_bm25_overlap` was the fastest stable focused benchmark and avoids simultaneous independent operations against embedded Qdrant. Exact ordered parity was required across all ranking stages. `concurrent_all` remains a diagnostic option.

The recorded CPU benchmark used one warm-up and three measured repetitions of the same long legal query:

| Strategy | Retrieval-block mean | End-to-end mean |
|---|---:|---:|
| Sequential | 554.84 ms | 5,195.65 ms |
| Qdrant batch overlapped with BM25 | 515.55 ms | 5,112.80 ms |
| Three independent concurrent calls | 505.00 ms | 5,147.90 ms |

Although three independent calls had the shortest isolated retrieval block, the batch-overlap strategy had the shortest end-to-end mean, avoided local-Qdrant contention and preserved exact ordered results. On the separate frozen 50-question run, batch overlap reduced retrieval-block mean from 522.53 ms to 498.16 ms (4.66%). CPU reranker variance made that run's end-to-end mean 1.39% slower, so the report does not claim a statistically established total-latency win. The reranker remains the dominant cost.

For a Qdrant batch, `dense_search_ms` and `sparse_search_ms` both denote the shared request duration and must not be added. `qdrant_batch_ms` makes this explicit.

## CLI and timings

```bash
HF_HUB_OFFLINE=1 poetry run python scripts/run_hybrid_retrieval_v2.py \
  --query "Bankın minimum məcmu kapital normativi nə qədərdir?" \
  --retrieval-strategy qdrant_batch_bm25_overlap \
  --local-files-only --output /tmp/reguaz-v2-query.json
```

The CLI also accepts JSON/JSONL, `--query-split dev|test|all`, filters, all retrieval cutoffs, warmups, repeat counts, and `--compare-rerank-depths 15 20`. Comparison reuses one fused candidate list. V2 default is now `rerank_top_k=15` and final top 5.

Every run records query encoding, dense, sparse, BM25, parallel retrieval wall time, fusion, reranker hydration, reranking, final payload fetch, parent lookup and total latency. The legacy timing key remains `top20_hydration_ms` for report compatibility even though 15 candidates are now hydrated. Overlapped durations are not summed to calculate total. Counts, overlap, duplicate/empty conditions, warnings, devices and strategy are included. Cold start is separate from warm-query latency.

## Top-15 reranker comparison

Both models were evaluated on the same frozen 50 questions with exact dense, sparse, BM25 and RRF parity:

| Model | Full Hit@5 | Test Recall@5 | Test MRR@5 | Test nDCG@5 | Mean reranker latency |
|---|---:|---:|---:|---:|---:|
| `BAAI/bge-reranker-v2-m3` | 0.74 | 0.718 | 0.663 | 0.656 | 2,561 ms |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 0.78 | 0.685 | 0.574 | 0.575 | 8,031 ms |

Test exact-child Hit Rate@5 tied at 0.775. BGE won test Recall@5, MRR@5 and nDCG@5 and was 68% faster in reranking on the measured CPU environment, so V2 continues with immutable BGE revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Alibaba required pinned custom-code revision `40ced75c3017eb27626c9d4ea981bde21a2662f4`; it remains an evaluation option, not the default.

## Frozen 50-question dataset

- `data/evaluation/reguaz_v2_evaluation_50.jsonl`
- `data/evaluation/reguaz_v2_evaluation_50_manifest.json`

The dataset contains exactly 50 Azerbaijani questions: 10 dev / 40 test and 20 easy / 20 medium / 10 hard across all eight categories. The process selected and inspected source chunks before binding questions; retrieval output was never declared gold evidence. Every evidence text, child/logical/document-version ID, locator and hash is validated against V2. It contains references and citation expectations, but no generated answers. No manual expert review is claimed.

Build and validate:

```bash
PYTHONPATH=. poetry run python scripts/build_v2_evaluation_dataset.py
```

Baseline, optimized and comparison reports:

```bash
PYTHONPATH=. poetry run python scripts/run_v2_retrieval_evaluation.py \
  --strategy sequential --warmup-queries 3 --local-files-only \
  --output results/v2_retrieval_evaluation/baseline_sequential.json

PYTHONPATH=. poetry run python scripts/run_v2_retrieval_evaluation.py \
  --strategy qdrant_batch_bm25_overlap --warmup-queries 3 --local-files-only \
  --output results/v2_retrieval_evaluation/optimized_batch_overlap.json

PYTHONPATH=. poetry run python scripts/compare_v2_retrieval_reports.py \
  --baseline results/v2_retrieval_evaluation/baseline_sequential.json \
  --optimized results/v2_retrieval_evaluation/optimized_batch_overlap.json \
  --output results/v2_retrieval_evaluation/baseline_vs_optimized.json
```

Dense, learned sparse, BM25, RRF and reranked metrics are reported for full/dev/test and by difficulty, category, type and evidence cardinality. Exact-child, logical-chunk and document-level hits are distinct. Multi-evidence recall uses every gold child ID.

## Limitations and next phase

- Retrieval weights and cutoffs remain unchanged. Future tuning may use dev only; test stays frozen.
- MPS was unavailable in the recorded environment, so reports use CPU fallback.
- The preserved sequential baseline was run through the in-place retriever's explicit `sequential` strategy because this workspace has no Git history; exact ordered parity was therefore checked for every source, RRF and reranked list.
- Exact-child Hit Rate@5 is not perfect; the frozen report exposes remaining quality risk.
- Relation expansion, context budgeting, generation, final citation rendering and production API wiring remain out of scope.

The next phase is relation-aware parent/sibling expansion under an explicit context budget, using the already-resolved final child/parent provenance.
