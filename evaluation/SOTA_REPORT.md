# Current model and end-to-end pipeline results

Measured November 19, 2026, America/Los_Angeles. The complete native run passed: [pipeline.json](results/native-ci/pipeline.json). [Full native CI](https://github.com/longhornyash/onecredit-company-recommendations/actions/runs/44676008272) and [52-test correctness CI](https://github.com/longhornyash/onecredit-company-recommendations/actions/runs/44676002430) both succeeded on the unchanged assessment Compose stack.

## Model selection

Default: **microsoft/harrier-oss-v1-0.6b**, MIT licensed, CPU PyTorch, unquantized full 1,024-dimensional outputs, revision pinned. Last-token pooling uses its published symmetric semantic-similarity instruction. No paid inference, API key, Hugging Face login or GPU is required. Input tokenization is capped at 512 tokens for the short-description workload; full source text remains stored.

The live official MTEB Multilingual v2 backend returned 492 model entries. Harrier 0.6B had the highest mean-task score among returned open-weight sub-1B entries: approximately **69.01**. The overall 27B leader was about **74.27**, with reported model memory about **51 GB**. This is a dated, benchmark- and size-scoped leaderboard selection, not an absolute claim about every model, benchmark, or inference configuration. Full MTEB was not rerun locally. The source URL, snapshot hash, comparison entries and selection scope are saved in [leaderboard-selection.json](results/leaderboard-selection.json).

## Pipeline actually executed

The single command initializes a fresh schema, starts the real worker, invokes the unchanged official publisher, drains RabbitMQ, audits exact company/message state, builds and validates HNSW, serves real HTTP, measures load/quality/CPU/memory, and verifies full-feed replay.

Observed results on a GitHub-hosted Ubuntu CPU runner with native PostgreSQL 16.16/pgvector 0.8.7 and RabbitMQ 3.13.7:

- Entire verification/benchmark pipeline: approximately **175 seconds**, including quality tests, load tests and duplicate replay; not just ingestion.
- Published **2,518** events through **real RabbitMQ 3.13.7**.
- Outcomes: **2,343 applied**, **66 stale**, **41 rejected**, **68 duplicates**.
- Stored **2,022** company records: **2,014 active**, **1,971 searchable**.
- **Zero missing message bodies**, source-version mismatches or unexpected company records.
- **177** embedding calls; persistent/in-memory caching avoids repeated description inference.
- Active ingestion span: **51.99 seconds**, approximately **48.4 messages/second**. Publish-to-drain wall time was **57.85 seconds**, including a deliberate five-second idle exit. Worker startup was measured separately at **15.25 seconds**, including first-use model setup.
- Full replay: **2,518 duplicate outcomes**, **zero embedding calls**; final queue empty, with no worker left consuming.

Raw evidence: [audit](results/native-ci/audit.json), [demo](results/native-ci/demo.json), [pipeline](results/native-ci/pipeline.json). Every run uses an isolated schema and refuses a nonempty queue or another consumer; it does not truncate existing company data or purge messages.

## Retrieval algorithm and performance

Use a partial, model-specific **HNSW cosine index**, `ef_search=200`, retrieve 64 candidates, then rank those candidates by exact distance. The index is activated only after validation; auto mode otherwise uses exact search. Requests above four results use exact search. The exact path also handles insufficient ANN results.

[HNSW verification](results/native-ci/hnsw-verification.json) compared **all 1,971** searchable source companies against exact pgvector search. Top-four ID recall was **100%** in this snapshot; tie-aware distance acceptance was also **100%**. EXPLAIN confirmed the index scan. Building took **0.50 seconds**; full validation took approximately **23 seconds**. Agreement with exact retrieval is not a human relevance judgment or a guarantee for future data; revalidate after substantial changes.

[Live HTTP measurements](results/native-ci/http.json) queried every searchable company once at each concurrency level: **5,913 measured requests**, 20 warm-up requests, **zero HTTP errors**. Every measured response had four sorted cards and excluded its source.

- **Concurrency 1:** p50 **3.37 ms**, p95 **4.21 ms**, approximately **281 requests/second**.
- **Concurrency 8:** p50 **13.06 ms**, p95 **19.01 ms**, approximately **579 requests/second**.
- **Concurrency 32:** p50 **57.56 ms**, p95 **75.37 ms**, approximately **533 requests/second**.

Historical local PostgreSQL/WASM [exact-search](results/pipeline-harrier/http.json) and [indexed](results/pipeline-sota/http.json) runs measured p95 98.35 ms and 13.58 ms respectively at concurrency eight. They used a different environment from the native results above; do not infer a native speedup from that cross-host comparison. These are warm single-host experiments, not a production capacity study.

## Quality versus the original baselines

The unchanged fixture has 24 actual feed descriptions in 12 bilingual activity families, with six development and six held-out query families. Its labels are provisional and have not been independently reviewed. It is a diagnostic, not a full-feed human gold standard.

On the 12 held-out mixed-language queries:

- **Harrier dense:** nDCG@4 **0.8622**, same-activity top result **11/12**.
- **E5-small dense:** nDCG@4 **0.5937**, same-activity top result **6/12**.
- **BGE-small dense:** nDCG@4 **0.3150**, same-activity top result **3/12**.

On the 12 held-out opposite-language queries, Harrier had nDCG@4 **0.9657** and same-activity top result **12/12**. The lexical/RRF combination was not a consistent improvement; dense remains the recommendation default. See [Harrier per-query results](results/native-ci/quality.json), [E5](results/e5-small.json), and [BGE](results/bge-small.json).

## CPU, memory and requirements

Harrier embeds all 177 normalized descriptions on CPU with two configured PyTorch threads. The native Linux run measured **41.13 seconds**, about **4.30 descriptions/second**, and **3.82 GiB peak process RSS**. Model initialization was measured separately at 2.63 seconds using cached artifacts. Earlier macOS runs observed 1.84–2.61 GiB RSS; memory varies by host/runtime. These values exclude database/broker memory and network download. [Native measurements](results/native-ci/model-performance.json).

The weight download is approximately 1.2 GB. Budget several GB of free disk and **6–8 GB free RAM** for the model and local infrastructure; actual needs vary with the host. E5/BGE are lower-cost supported alternatives. Model downloads need Internet only on first use; inference itself stays local. A populated cache works with `HF_HUB_OFFLINE=1`.

## Verification scope

The native report records `native_postgres16_verified=true`. The unchanged official publisher sent all events through RabbitMQ, the audit found no missing bodies or company-state mismatches, full replay produced 2,518 duplicate outcomes with zero inference, and the final queue had zero messages and consumers. All **52 tests passed with zero skips**, including native database and broker integration tests: [test summary](results/tests-native.json).

These checks do not prove host-crash durability, production availability or sustained capacity. Quality labels remain provisional and need independent bilingual review. Earlier PostgreSQL 18.7/PGlite results in `pipeline-sota` are retained as historical results, not native PostgreSQL 16 claims. No paid API key, Hugging Face login, cloud credentials or additional data is needed to run the solution.


## Reproduce the requested pipeline

```bash
uv sync --locked
docker compose up -d --wait
uv run python -m scripts.pipeline
```

The `full-native-pipeline` GitHub Actions workflow was manually dispatched and passed; its raw artifact is checked in under `results/native-ci`. See README.md for serving a retained pipeline schema and for exact-search rollback.
