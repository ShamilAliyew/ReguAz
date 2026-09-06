# ReguAZ deployment on Render

## Production topology

The root `render.yaml` creates three Render resources in Frankfurt:

- `reguaz`: public Nginx web service serving the React build;
- `reguaz-backend`: private Docker service running one FastAPI worker;
- `reguaz-postgres`: private managed PostgreSQL for users and sessions.

The frontend proxies `/api/*` to the backend over Render's private network. The
backend is the only service allowed to hold provider and database credentials.
It connects outbound to Qdrant Cloud, Groq, and optionally OpenRouter.

The backend image contains only immutable, hash-validated V2 text artifacts:

- the chunk manifest;
- child and parent JSONL files;
- cleaned Markdown sources;
- the small BGE-M3 embedding manifest.

Embedding shards, embedded Qdrant storage, and the local Gemma GGUF are not
included. BGE-M3 and the reranker are downloaded at their pinned revisions into
the attached `/cache` disk on first startup. Production generation defaults to
Groq GPT-OSS 20B. The production image also omits `llama-cpp-python`; compiling
that optional local-Gemma runtime exceeds Render's build-memory allowance.
Gemma remains available in local installations and is imported only when it is
actually selected.

## 1. Create and populate Qdrant Cloud

Create a Qdrant Cloud cluster near the Render Frankfurt region and create a
database API key. Keep both values out of Git.

Set them only in the current terminal:

```bash
export QDRANT_URL='https://YOUR-CLUSTER.cloud.qdrant.io:6333'
export QDRANT_API_KEY='YOUR-QDRANT-DATABASE-API-KEY'
```

Upload the existing V2 vectors from the ignored local embedding shards:

```bash
poetry run python scripts/run_qdrant_ingestion_v2.py
```

The command creates a versioned physical collection, validates exactly 17,729
points, validates dense and sparse search, and then atomically assigns the
`reguaz_v2_current` alias. It writes its local audit report to
`logs/qdrant_ingestion_v2_cloud.json`. It does not modify the local embedded
Qdrant collection.

Do not use `--force-collection` unless the named remote physical collection is
known to be disposable. That option deletes the matching remote collection
before rebuilding it.

Validate all packaged sources and the remote collection:

```bash
poetry run python scripts/check_render_readiness.py --check-qdrant-cloud
```

Expected invariants include `child_count: 17729`, `dense_dimension: 1024`,
`qdrant.mode: remote`, and `status: ready`.

## 2. Commit the deployment contract

The embedding shards and Qdrant files remain ignored. The embedding manifest is
intentionally tracked because it pins the exact BGE-M3 revision used by the
query encoder and validates the remote collection.

```bash
git status --short
git add render.yaml .dockerignore .gitignore backend frontend scripts docs \
  data/processed/v2/embeddings/bge_m3/*/manifest.json
git commit -m "feat: prepare ReguAZ for Render deployment"
git push
```

Review unrelated working-tree changes before staging; do not blindly commit
files that are outside this deployment.

## 3. Create the Render Blueprint

In Render:

1. Open **Blueprints** and choose **New Blueprint Instance**.
2. Connect the GitHub repository containing the root `render.yaml`.
3. Keep all three resources in the Frankfurt region.
4. Provide the prompted secret values:
   - `QDRANT_URL`;
   - `QDRANT_API_KEY`;
   - `GROQ_API_KEY`;
   - `OPENROUTER_API_KEY` for TTS.
5. Apply the Blueprint and wait for the backend's first model download.

Do not add provider keys to frontend environment variables. `DATABASE_URL` and
the backend's private `API_UPSTREAM` address are injected automatically through
Blueprint references.

## 4. Verify the deployment

Open the public `reguaz` URL and check the proxied endpoint:

```text
https://YOUR-FRONTEND.onrender.com/api/health
```

The response must report:

- `status: healthy`;
- `pipeline_version: v2`;
- `chunk_lookup_size: 17729`;
- `auth_enabled: true`;
- `auth_database_ready: true`;
- `qdrant_mode: remote`.

Then register a user, sign out, sign in again, run one legal query, open a
citation, and request TTS once. Inspect backend logs if any stage fails.

## Operational limits

- The first backend startup downloads pinned BGE-M3 and reranker snapshots and
  is slower than subsequent starts.
- The 10 GB cache disk restricts the backend to one instance and removes
  zero-downtime deploys. It can later be replaced by a model-baked image if
  horizontal scaling becomes necessary.
- Render has no Apple MPS. BGE-M3 and reranking run on CPU; latency must be
  benchmarked after deployment.
- Groq receives the question and only the evidence selected for generation.
- Authentication is durable in PostgreSQL. Chat history is still stored in the
  user's browser and is not synchronized across devices.
- Database tables are currently created idempotently at startup. Introduce
  Alembic before the first schema-changing production release.
