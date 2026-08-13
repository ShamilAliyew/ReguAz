# ReguAZ V2 generation and citation pipeline

## Scope

Current V2 path extends the existing hybrid retriever in place:

```text
query
retrieval 30/30/30 + RRF k=60 + rerank top 15 + final top 5
depth-one relation expansion
evidence selection and deduplication
Gemma chat-template-aware context budget
grammar-constrained local generation
backend citation resolution
cleaned-Markdown viewer highlighting
```

No chunk, embedding, Qdrant, alias, retrieval model, or reranker rebuild occurs. Recursive expansion, unresolved-reference expansion, PDF coordinates, full 50-question generation evaluation, RAGAS, streaming, TTFT, and model-based entailment checking remain deferred.

## Relation policy

Expansion starts only from final-five reranked child chunks. Maximum depth is one. Defaults:

- per-seed expanded item cap: `4`;
- global expanded item cap: `16`;
- reference confidence threshold: `0.8` when confidence exists;
- parent excerpt cap: `700` characters.

Priority: seed, continuation, compact parent excerpt, structurally required sibling, resolved reference/referenced-by, annex/table parent, approved-document parent. Ordinary siblings are excluded unless split/continuation metadata proves structural completion. Unresolved and missing targets generate structured diagnostics. Expanded artifacts remain additional context, never retrieval results.

Actual V2 schema target types are respected: `annex_parent`, `table_parent`, and `approved_document` target parent artifacts; resolved references target child artifacts.

## Evidence and selectors

Every selected item is a strict Pydantic V2 `EvidenceItem`. Request-local `E1`, `E2`, and later IDs are assigned only after budget selection. Stable chunk/parent IDs, document/version IDs, canonical locator, hierarchy, pages, content/source hashes, relation provenance, and selection reason remain authoritative.

Selectors target `cleaned_markdown_utf8_codepoint_v1` and contain:

- full-representation `start/end` offsets;
- page-relative offsets when valid;
- exact quote plus prefix/suffix fallback;
- representation ID, document version, and source SHA-256.

Backend verifies exact text and hashes. Frontend first uses page-relative position, then a unique exact quote. Zero or multiple quote matches cause a visible warning; unrelated text is never highlighted.

Viewer displays authoritative cleaned Markdown, not original PDF. V2 spans have `bbox=null`; original PDFs and bounding boxes are absent. PDF-coordinate highlighting is therefore not claimed or fabricated.

## Context budget

Configured model: `gemma-4-E4B-it-Q4_K_M.gguf`, llama.cpp, context `8192`, output reserve `512`, safety margin `256`.

Budgeting uses loaded Gemma tokenizer and counts final GGUF chat-template rendering, rules, question, evidence JSON, and output contract. Evidence is deduplicated before counting. Seed evidence has retention priority. Every candidate addition is reserialized and recounted. Silent truncation is forbidden. Missing required seed evidence produces `insufficient_evidence`.

## Gemma prompt and output

GGUF `tokenizer.chat_template` is mandatory. Only a `user` message is supplied; system-level rules live at start of that user turn. Evidence and question are JSON-serialized with control-tag characters escaped. Retrieved text is explicitly treated as untrusted data.

Gemma returns only:

```json
{
  "status": "answered",
  "answer_blocks": [
    {"text": "Atomik iddia", "evidence_ids": ["E1"]}
  ],
  "limitations": []
}
```

llama.cpp GBNF generated from simple JSON Schema constrains syntax. Strict Pydantic validation then enforces size and status rules. Unknown or duplicate evidence IDs and citation-free answered blocks fail closed. At most one measured retry is allowed.

## Citations

Gemma never supplies document names, locator, page, IDs, hashes, URLs, or numeric citation markers. Backend resolves all metadata from per-request evidence map and immutable V2 artifacts, deduplicates citations, assigns numbers in first-use order, and renders backward-compatible `[1]` Markdown.

`source_validated=true` means artifact, version, hashes, source representation, and selectors passed validation. It does not prove semantic entailment. `claim_support_status="not_evaluated"` remains explicit until next full generation/citation evaluation phase.

## API and UI

Set:

```bash
REGUAZ_PIPELINE_VERSION=v2
VITE_USE_MOCK_API=false
```

`v2` is backend default. Only selected pipeline and its models load. Invalid values fail at settings validation; V2 never silently falls back to V1.

Existing `answer`, `sources`, and `metrics` remain. V2 adds `status`, `answer_blocks`, `citations`, `warnings`, `pipeline_version`, model metadata, and detailed timings. UI shows a V2 badge. Citation hover shows document, locator, page, and seed/expanded role. Click opens cleaned-document page and exact selectors.

## Privacy and timings

V2 logs request ID, status, timings, counts, warning count, model, and device. It does not log raw question, prompt, evidence, answer, or model response. Raw prompts are not written to debug files.

Timings use `time.perf_counter()` and include retrieval, relation expansion, evidence selection, budgeting, prompt build, generation, parsing, evidence-ID validation, citation resolution, API mapping, and wall-clock total. TTFT is not reported because generation is not genuinely streamed.

## Verification

```bash
poetry run python -m pytest -q backend/tests
poetry run ruff check backend/reguaz/services/generation/*_v2.py backend/reguaz/services/generation/gemma_service.py backend/reguaz/retrieval/v2_artifacts.py backend/app/main.py backend/app/services backend/app/schemas backend/app/core backend/app/api backend/tests/generation scripts/run_generation_v2_smoke.py

cd frontend
node node_modules/vitest/vitest.mjs run
node node_modules/eslint/bin/eslint.js . --ext ts,tsx --max-warnings 0
node node_modules/typescript/bin/tsc
node node_modules/vite/bin/vite.js build

poetry run python scripts/run_generation_v2_smoke.py
```

Final command uses only frozen ten-record development split. 40-record test split is untouched. Full 50-question generation, citation-faithfulness, and RAGAS evaluation is next stage.

## Generation model comparison

Local Gemma and NVIDIA NIM `openai/gpt-oss-120b` share the same cached V2 retrieval traces, evidence contract, backend validation, and citation resolver:

```bash
poetry run python scripts/compare_generation_models_v2.py --limit 10
```

The NVIDIA provider reads `NVIDIA_API_KEY` only from process environment or `.env`; keys and reasoning content are never persisted. UI provider is selected with `LLM_TYPE=gemma` or `LLM_TYPE=nvidia_gpt_oss`. A model is eligible only with 100% request completion, structured/evidence validation, and source validation; eligible model with lowest mean projected end-to-end latency wins.

August 2026 dev-10 run selected `gemma`: Gemma completed 10/10, while GPT-OSS timed out on 10/10 real ReguAZ prompts at the configured 60-second comparison ceiling. A short synthetic GPT-OSS prompt succeeded in 3.532 seconds, showing endpoint availability but not production-prompt stability. Current UI default therefore remains `LLM_TYPE=gemma`.

## Groq model selection

V2 also supports `groq_gpt_oss_20b` and `groq_gpt_oss_120b` through Groq's OpenAI-compatible HTTPS endpoint. The provider:

- reads `GROQ_API_KEY` only from environment or ignored `.env`;
- whitelists the two production GPT-OSS model IDs;
- requests strict JSON Schema output and still runs backend Pydantic/evidence validation;
- uses low reasoning effort for latency-sensitive factual regulatory Q&A;
- suppresses reasoning output and never logs prompts, evidence, answers, or secrets;
- lazily loads each selected provider once, reuses its client/pipeline per process, and applies bounded SDK retries;
- exposes client wall time plus provider queue, prompt, completion, total, and cached-token metadata when returned by Groq.

Backend `GET /llm-models` reports configured/loaded models. UI selection is sent as a whitelisted `llm_model` value on each `/chat` request. Local Gemma and remote providers share retrieval, relation expansion, evidence selection, structured output validation, and backend citation resolution. Selecting a remote model sends the user question and selected evidence to that provider.

Portable configuration uses paths relative to repository root. No user-specific absolute path is stored in code or reports. Add local secrets only to ignored `.env`:

```bash
GROQ_API_KEY=replace_with_rotated_key
```

Run the consent-gated frozen dev comparison:

```bash
poetry run python scripts/compare_generation_models_v2.py \
  --models gemma groq_gpt_oss_20b groq_gpt_oss_120b \
  --limit 10 \
  --allow-remote-evidence
```

Retrieval traces are computed once and reused by every model. Eligibility requires 100% completion, structured/evidence validation, and source validation. Report names both lowest-latency winner and quality-proxy leader. Token F1 and citation overlap are diagnostic proxies, not proof of semantic correctness, so UI default is not silently changed from this small dev run. Full generation/citation evaluation remains deferred.

### Frozen dev-10 result (12 August 2026)

The authorized real Groq run is stored at `results/v2_generation/groq_comparison_dev10.json`. Both models completed all ten requests against identical cached retrieval/evidence traces.

| Model | Mean generation wall time | Mean projected end-to-end | Provider compute | Structured/evidence validation | Token F1 | Gold child citation recall |
|---|---:|---:|---:|---:|---:|---:|
| Groq GPT-OSS 20B | 22.413 s | 24.502 s | 0.397 s | 90% | 0.302 | 0.30 |
| Groq GPT-OSS 120B | 21.693 s | 23.778 s | 0.602 s | 100% | 0.245 | 0.30 |
| Gemma 3 4B local, earlier same dev-10 run | 38.054 s | 40.145 s | not available | 100% | 0.304 | 0.21 |

The 120B model is the policy winner because it is the only Groq candidate with 100% structural/evidence validation and has the lower measured client wall time. This is not proof that its answers are semantically better: 20B and local Gemma have higher token-F1 proxies. Groq provider compute stayed below 1.3 seconds, while sustained client wall time was much higher; network and SDK retry/backoff time dominated this small run. Rate-limit headers were not persisted, so rate limiting is a likely explanation, not a confirmed diagnosis.

The local Gemma row comes from the earlier saved run on the same frozen development records and retrieval contract. A fresh Gemma rerun inside the Codex sandbox was blocked because llama.cpp could not create a Metal command queue, including its attempted CPU fallback; that environment failure is not counted as model quality. The UI default therefore remains explicit and configurable rather than being silently rewritten from a ten-question proxy benchmark.
