# Verification and recommendation evaluation

**Historical E5/BGE baseline report.** The default has since been upgraded to Harrier plus verified HNSW. See [SOTA_REPORT.md](SOTA_REPORT.md) for the current pipeline and performance; the original artifacts below are retained for comparison.

Local validation: November 19, 2026 (America/Los_Angeles). These are observed results, not projected production performance.

## Environment and boundaries

- Application: Python 3.14.5, locked uv dependencies, macOS ARM64.
- Broker: **real RabbitMQ 3.13.7**, the assessment's exact broker version, running locally on Erlang 26.2.5.21. The official publisher was used unchanged.
- Database: **PostgreSQL 18.7 / PGlite 0.5.17 with pgvector**, accessed through the PostgreSQL wire protocol. Native Docker/PostgreSQL startup was unavailable in this execution environment. The WASM multiplexer is not proof of native PostgreSQL 16 concurrency, persistence or performance.
- Native PostgreSQL 16 plus the provided Compose services still needs its own CI run. The repository includes that workflow, but no remote CI result is claimed.

## Correctness

[tests.json](results/tests.json): **49 passed, zero failed, zero errors, zero skipped** in approximately 4.1 seconds. This includes a real-broker channel-close/redelivery test. Tests cover malformed JSON/Unicode, normalization expansion, timestamps, vector validity, bounded retry/cancellation, duplicate claims, concurrent source updates, tombstones/reactivation, timestamp ties, rollback, post-commit ack failure, model isolation, API validation/errors/metric labels and full-feed shuffled replay.

[reproducibility.json](results/reproducibility.json): extracted the source archive into a clean directory, created a new virtual environment with `uv sync --locked`, and reran the entire suite: **49 passed**, with no failures/errors/skips. Ruff lint and format checks also passed in that fresh extraction.

[feed-audit.json](results/feed-audit.json) verifies the clean end-to-end feed state:

- 2,518 published events, 2,450 distinct bodies.
- 2,022 company records, 2,014 active, 1,971 searchable.
- Zero missing message bodies and zero source-version mismatches.
- Zero unexpected company records or message bodies.
- 41 unique rejected bodies; duplicate invalid bodies explain the higher rejected-line count. Raw bytes and rejection reasons remain inspectable in `rejected_events`.

[rabbitmq-ingestion.json](results/rabbitmq-ingestion.json): the actual broker/consumer path produced 2,343 applied, 66 stale, 41 rejected and 68 duplicate outcomes. Only 177 model inferences were needed. Summed per-message processing duration was approximately 3.86 seconds; this is **not** cold-start or overall ingestion wall time.

The separate [ingestion.json](results/ingestion.json) records a file-recovery run, explicitly labeled as file transport. It is not substituted for the broker evidence.

## Process crash and recovery

[crash-recovery.json](results/crash-recovery.json) records a separate fault-injection run against task-owned disposable services:

1. Reset the disposable database/queue and publish the official feed.
2. Start the actual consumer and kill its process with SIGKILL after 102 unique events have committed.
3. Restart the consumer and drain the remaining/requeued deliveries.
4. Verify the complete source-version/message audit passes.
5. Publish the entire feed again and drain it.

The second full replay produced **2,518 duplicate outcomes and zero embedding calls**. The final queue had zero ready messages and zero consumers. This tests real consumer death and broker redelivery, while the database durability caveat above still applies. The automated rollback and ack-failure tests cover the transaction boundaries separately.

## Recommendation diagnostic

[cases.json](cases.json) contains 12 bilingual activity families / 24 descriptions copied from the actual feed. Family labels and related-activity judgments are **provisional** and have not had independent bilingual review. Six development query families and six held-out query families are fixed in the fixture; all documents remain in the retrieval corpus. No fine-tuning or weight optimization was performed.

Two tasks exclude the exact query description: a mixed-language candidate corpus, and an opposite-language-only corpus. Same activity receives grade 2, explicitly related activity grade 1, otherwise 0. Report nDCG@4, relevance@4 and same-activity top-1. With only one same-family counterpart per query, relevance@4 has an intentionally low ceiling; it should not be read as a four-card full-feed precision estimate.

Observed held-out results (12 queries per task):

- **Mixed-language, dense E5:** nDCG@4 **0.5937**, same-activity top-1 **6/12**.
- **Mixed-language, dense BGE:** nDCG@4 **0.3150**, same-activity top-1 **3/12**.
- **Cross-language, dense E5:** nDCG@4 **0.8981**, same-activity top-1 **10/12**.
- **Cross-language, dense BGE:** nDCG@4 **0.6232**, same-activity top-1 **5/12**.
- E5 hybrid improved the cross-language nDCG to **0.9235**, but reduced mixed-language nDCG to **0.1944**. It is therefore exposed only as an experiment, not the general default.

The development groups point in the same model-selection direction. Raw per-query rankings, grades, case hashes, model revisions and database versions are in [e5-small.json](results/e5-small.json) and [bge-small.json](results/bge-small.json).

Failure analysis: shared accounting words can confuse accounting services with accounting software; language-specific overlap can outweigh a cross-language activity equivalent; location words can distract from activity. The mixed-language weakness is visible rather than hidden by the many identical templates in the full feed. Larger independently judged data is needed before making a general quality claim.

Decision: **ship dense multilingual E5-small**. It outperformed the English baseline in this diagnostic, while unweighted lexical fusion was not a consistent improvement. No state-of-the-art claim or statistical significance claim is made.

## CPU and memory cost

Separate fresh Python processes embedded all **177** eligible normalized descriptions from the full feed history, using cached model artifacts and two ONNX CPU threads:

- [E5](results/e5-performance.json): approximately **190 descriptions/second**, **815 MiB peak process RSS**, 1.24-second initialization.
- [BGE](results/bge-performance.json): approximately **148 descriptions/second**, **295 MiB peak process RSS**, 0.26-second initialization.

These are one-run local measurements, including long repetitive descriptions and excluding network download, database and broker memory. They do not establish a stable comparative throughput ranking. E5's higher memory cost is a concrete tradeoff for the observed multilingual quality improvement. Both runs verified normalized 384-dimensional vectors and that a cache hit avoids another inference.

Reproduce with `EMBEDDING_MODEL=intfloat/multilingual-e5-small HF_HUB_OFFLINE=1 python -m scripts.model_benchmark --output evaluation/results/e5-performance.json`, then repeat in a separate process with `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` and a different output filename.

## Live HTTP smoke benchmark

[http-benchmark.json](results/http-benchmark.json): 20 warm-up requests, then 200 requests at concurrency one and 200 at concurrency eight, across 100 company IDs against the fully ingested database.

- Sequential: p50 approximately **3.1 ms**, p95 **3.6 ms**.
- Concurrency eight: p50 approximately **19.4 ms**, p95 **24.4 ms**.
- **Zero HTTP errors** in 400 measured requests. Every response had four sorted cards and excluded the source company.

These are local warm HTTP timings on the WASM database, including client/network overhead. They do not establish a production SLO, native Postgres latency, sustained throughput or cold-start performance. The benchmark process starts and stops its own API server.

[demo-response.json](results/demo-response.json) contains the successful live response for `SYN-001550`; all four returned cards are Food & Beverage companies in this observed demo.

## Reproduce

Use README.md's startup sequence on the provided native Compose services. Then run the full suite with explicit `TEST_DATABASE_URL` / `TEST_RABBITMQ_URL`, run the strict audit, evaluate both models, and execute `python -m scripts.benchmark`. Do not replace these measured artifacts with results from a different environment without updating the environment labels.
