# OneCredit company recommendations

CPU-only multilingual recommendations with RabbitMQ, FastAPI and PostgreSQL/pgvector. Assessment **2026.11**, based on upstream commit `d9bed24` / tag `mle-v2026.11`. The original brief is preserved in [ASSESSMENT.md](ASSESSMENT.md).

The consumer handles malformed input, duplicate delivery and out-of-order updates. Message deduplication and company changes commit together before acknowledgement. Similarity search runs in Postgres; HTTP requests never load an embedding model.

## Run the complete pipeline

No paid API key, Hugging Face token, GPU or cloud account is required. Data comes from the unchanged official feed. The default model is **Microsoft Harrier OSS v1 0.6B**, pinned, MIT licensed, CPU-only, with its full 1,024-dimensional vectors. It was the highest mean-task score among open-weight sub-1B entries in the retrieved MTEB Multilingual v2 snapshot; see [selection evidence](evaluation/results/leaderboard-selection.json). This is a scoped leaderboard claim, not a claim that it is the largest overall winner.

```bash
uv sync --locked
docker compose up -d --wait
uv run python -m scripts.pipeline
```

Or run `just pipeline`. This starts the actual consumer, publishes the official feed, drains and audits it, builds and validates HNSW, starts a temporary live API, benchmarks every searchable company at concurrency 1/8/32, evaluates recommendation quality and CPU/memory, then verifies a complete duplicate replay. It stops its worker/API processes and writes `evaluation/results/pipeline-latest/pipeline.json` plus raw metrics. It uses a new database schema and refuses a nonempty queue or another active consumer; it never truncates existing company data or purges messages.

For a retained run, set `DATABASE_SCHEMA` to the schema printed in that report before starting the API. The pipeline leaves that schema available for inspection. Use the default `public` schema for the manual sequence below. `SEARCH_MODE=auto` uses only a model-specific HNSW index that passed verification, otherwise exact search. `SEARCH_MODE=exact` forces the reference path. HNSW optimizes limits up to four; larger limits use exact search.

Current measured pipeline and leaderboard-model results: [evaluation/SOTA_REPORT.md](evaluation/SOTA_REPORT.md). The earlier E5/BGE results are retained as baselines, not the current default.

## Manual start from a clean checkout

Requirements: Docker Engine/Compose, uv, and network access for the first model download. uv installs Python 3.14. `just` is optional.

```bash
uv sync --locked
docker compose up -d --wait
uv run python -m onecredit.cli init-db
uv run python -m producer.publish
uv run python -m onecredit.consumer --idle-seconds 5
uv run python -m onecredit.cli audit
uv run python -m scripts.index_benchmark
uv run uvicorn onecredit.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl --fail --silent 'http://localhost:8000/companies/SYN-001550/similar?limit=4'
curl --fail --silent 'http://localhost:8000/ready'
```

For a continuous worker, omit `--idle-seconds`. Ctrl-C closes its connection; unfinished messages remain eligible for redelivery. Idle exit is for a finite demo, not proof that upstream publishing is complete.

With `just`: `just setup`, `just up`, `just init-db`, `just publish`, `just drain`, `just audit`, `just serve`. `just demo` performs ingestion/audit, briefly starts an HTTP server on port 18080, prints recommendations and stops it.

**Reset is destructive:** `docker compose down -v` removes local assessment data. Then repeat setup. The provided Compose file has no named durable volumes and is not a production persistence configuration.

## API

- `/health`: process liveness.
- `/ready`: database/schema availability; database errors return 503.
- `/companies/{reg}`: latest snapshot, including inactive/incomplete records and `recommendationEligible`.
- `/companies/{reg}/similar?limit=4&method=dense`: ranked cards. Limit 1–20. Unknown/inactive source: 404; unavailable/incompatible embedding: 409; invalid parameters: 422; database errors: 503.
- `/metrics`: bounded-label API request counters and duration histograms. Worker metrics are process-local; its structured logs are the operational output.
- `/docs`: generated OpenAPI documentation.

The recommendation response contains `registrationNumber`, `method`, `scoreType`, `modelVersion` and `results`. Each result contains `registrationNumber`, `companyName`, nullable `industry`/`state`, and `score`. Cosine similarity is not a probability. Ties use registration number. Self, inactive/unnamed companies and incompatible vector spaces are excluded.

`method=hybrid` exposes experimental SQL lexical/vector Reciprocal Rank Fusion. Dense Harrier remains default because evaluation does not support lexical fusion as the general default.

## Configuration

See [.env.example](.env.example). Export variables explicitly; `.env` is not automatically loaded.

- `DATABASE_URL`, `RABBITMQ_URL`: override local connection defaults.
- `POSTGRES_PORT`, `RABBITMQ_PORT`, `RABBITMQ_UI_PORT`: match the scaffold.
- `EMBEDDING_MODEL`: `microsoft/harrier-oss-v1-0.6b` (default), `intfloat/multilingual-e5-small`, or `BAAI/bge-small-en-v1.5`.
- `DATABASE_SCHEMA`: default `public`; the pipeline reports its isolated schema.
- `SEARCH_MODE`: `auto` (verified index if available), `exact`, or `hnsw` (requires a verified index).
- `MODEL_CACHE`: default `.models` (gitignored).

Artifacts are revision-pinned in `onecredit/config.py`. Harrier uses PyTorch CPU, last-token pooling, the publisher's symmetric STS instruction and untruncated 1,024-d vectors. Business input is limited to 512 model tokens while full source text remains stored. E5/BGE use ONNX Runtime CPU as comparison baselines. Harrier's weight download is approximately 1.2 GB and its measured CPU process peak was about 3.8 GiB RAM on Linux, excluding infrastructure. Budget 6–8 GB free RAM for the full stack and several GB of free disk. Linux dependencies explicitly select CPU PyTorch wheels. No remote model code is executed. After populating the cache, use `HF_HUB_OFFLINE=1` for offline runs.

To change models, **stop API and worker**, set the new model, run `init-db` to upgrade the old 384-d column without dropping data, run `uv run python -m onecredit.cli reembed`, validate the new model's index, then restart both with that setting. Do not serve during migration: filtering prevents mixed spaces but not incomplete coverage. Replaying processed messages does not re-embed them.

## Tests

```bash
uv run ruff check onecredit tests evaluation scripts
uv run ruff format --check onecredit tests evaluation scripts
uv run pytest -q

# Explicit opt-in: disposable database, never production.
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/companies \
TEST_RABBITMQ_URL=amqp://guest:guest@localhost:5672/ \
uv run pytest -q
```

Integration tests create/remove unique schemas. The broker test owns a separate durable queue and never purges `company-events`. Missing service URLs cause explicit skips. CI supplies both URLs and uses the provided Compose versions. Tests cover shuffled full-feed replay, concurrency, tombstones, rollback, acknowledgement failure, retries/cancellation, model isolation, poison input and API contracts. Deterministic test vectors are not used to make quality claims.

## Evaluation and audit

```bash
uv run python -m onecredit.cli inspect-feed
uv run python -m onecredit.cli audit
uv run python -m evaluation.run --output evaluation/results/harrier-evaluation.json
EMBEDDING_MODEL=intfloat/multilingual-e5-small \
uv run python -m evaluation.run --output evaluation/results/e5-small.json
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
uv run python -m evaluation.run --output evaluation/results/bge-small.json
uv run python -m scripts.benchmark
uv run python -m scripts.model_benchmark --output evaluation/results/harrier-performance.json
```

Evaluation uses a temporary SQL table; it never changes company embeddings. Its 24 descriptions come from the official feed: 12 bilingual activity families, six development and six held-out query families. Query-description copies are excluded; cross-language queries restrict candidates to the other language. Labels are provisional and require independent bilingual review. This is a small diagnostic, not full-feed quality or a validated human gold standard.

See [evaluation/SOTA_REPORT.md](evaluation/SOTA_REPORT.md) for current results, [evaluation/REPORT.md](evaluation/REPORT.md) for historical baselines, [cases](evaluation/cases.json), and [raw results](evaluation/results). The full pipeline and all 52 tests passed on the unchanged native PostgreSQL 16/pgvector and RabbitMQ Compose stack in [GitHub Actions](https://github.com/longhornyash/onecredit-company-recommendations/actions/runs/44676008272). Native measurements: 5,913 successful HTTP requests, p95 19.01 ms at concurrency eight, and 100% top-four ANN/exact agreement across all 1,971 searchable companies. Earlier PostgreSQL/WASM results remain separately labeled.

The strict clean-feed audit expects **2,022** company records, **2,014** active and **1,971** searchable under the documented input policy. All **2,450** unique bodies must be accounted for, including rejected input; latest versions must match exactly.

`uv run python -m onecredit.cli ingest-file` is an explicit recovery utility with the same transaction logic and publisher-compatible hashes. Normal ingestion is through RabbitMQ. File ingestion is not evidence of broker transport testing.

## Submission

Read [DESIGN.md](DESIGN.md) for assumptions, research, measured tradeoffs, 10× scaling, limits. The [public repository](https://github.com/longhornyash/onecredit-company-recommendations) was created from the official template. [Correctness CI](https://github.com/longhornyash/onecredit-company-recommendations/actions/runs/44676002430) and the [full native pipeline](https://github.com/longhornyash/onecredit-company-recommendations/actions/runs/44676008272) passed. Submit the repository URL and your résumé using the form in ASSESSMENT.md. No résumé or application has been submitted.
