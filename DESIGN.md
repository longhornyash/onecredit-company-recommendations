# Design — assessment 2026.11

## Approach

Separate CPU ingestion from a read-only HTTP path. Postgres owns company state, processing history and vector retrieval; no second search service or paid inference API is needed. The default is pinned Microsoft Harrier OSS v1 0.6B: CPU PyTorch, full 1,024-dimensional normalized vectors, last-token pooling and its published symmetric STS instruction. The retrieved MTEB Multilingual v2 snapshot placed it highest by mean-task score among open-weight sub-1B entries. This scoped selection honors the laptop-CPU constraint; the overall 27B leader's reported memory is about 51 GB. E5-small and BGE-small remain baselines. See evaluation/results/leaderboard-selection.json and evaluation/SOTA_REPORT.md. Artifact revision, preprocessing, pooling, context limit and instruction identity are part of the embedding key; spaces are never mixed.

## Event semantics

- Declare exactly `company-events`, durable, with no extra arguments. Prefetch eight, one inference lane, sequential processing. CPU inference runs off the event loop.
- Claim `message_id`, apply the conditional company update, cache the embedding and store the processing outcome in one transaction. Commit before ack. Pre-commit failure rolls back; post-commit redelivery is a no-op. This guarantees idempotent database effects, not end-to-end exactly-once delivery.
- A SHA-256 body fingerprint detects reuse of a message ID for different content and quarantines the conflict.
- Compare `(source_updated_at, event_priority, canonical_event_hash)`: newest timestamp wins; deletion wins equal timestamps; equal-type conflicts use maximum canonical hash. This converges under replay but cannot recover missing upstream ordering information. A newer upsert can reactivate a company.
- An unseen deletion creates a tombstone. Nullable fields are intentional; older upserts cannot resurrect it.
- Assume upserts are **full snapshots**, not patches. Missing/null/blank fields become null. A newer incomplete snapshot clears the old embedding. The company remains visible in lookup but is not eligible for unreliable semantic recommendations.
- Require a usable registration number and source timestamp. Accept aware ISO times and integer epoch milliseconds. Reject naive times, booleans, invalid types and dates outside 1900–2200. Do not invent arrival-time ordering.
- Validate JSON object/type, duplicate keys, non-finite constants, field types, embedded NULs and sizes. Messages over 1 MiB and strings over 100,000 characters are quarantined. NFKC/whitespace normalization is versioned. Retain description text while tokenizer context limits truncate model input.
- Store rejected raw bytes/reason before ack. Corrupt identity/time cannot safely be reconstructed. “No lost companies” applies to accepted identifiable/versioned snapshots, with durable evidence for rejected input rather than fabricated recovery.
- Operational/model errors retry five times with exponential backoff/jitter, then exit visibly with work unacknowledged. A supervisor can restart after remediation. Poison input does not infinitely requeue.

## Efficiency and serving

The 2,518-message feed has 2,450 distinct bodies and 179 distinct raw descriptions. Persist embeddings by `(embedding_key, normalized_description_hash)` and use a bounded 512-entry memory LRU. Cheap duplicate/stale checks avoid inference; the database repeats ordering at mutation time to close races. Downloads/inference never happen inside a database transaction or HTTP request. Harrier's observed isolated-process peak RSS ranged from about 1.8 to 3.8 GiB across macOS and Linux; E5/BGE use less memory. Account for that cost before multiplying worker processes. Linux locks use the CPU PyTorch index, not CUDA dependencies. Model code is built into Transformers; remote model code is not executed.

Use a model-specific partial HNSW index only after a full-population verification gate. Build an inactive candidate index, compare every source company's top four with exact pgvector search, inspect the actual query plan, then activate its name only if the gate passes. Use `ef_search=200`, retrieve 64 candidates, rerank their exact distances and return four. Auto mode falls back to exact search when no verified index exists; requests above four and insufficient ANN candidate sets use exact search. One snapshot matched exact top-four IDs for all 1,971 source companies; that is not a future-update guarantee. Revalidate after substantial data changes.

A repeatable-read transaction gives source and candidates one snapshot. Exclude self, inactive/unnamed records and other embedding versions. SQL statements time out after five seconds; pool acquisition also has a five-second timeout. Limit is 1–20. All input values are parameters. Generic vector columns retain full 384- and 1,024-d model outputs; revision-key checks and vector validation prevent mixing. Explicit init-db upgrades the old dimension-constrained columns without dropping data. The partial index casts only its own model's rows to the correct dimension.

Optional hybrid ranking combines top-30 dense and lexical lists with `sum(1/(60+rank))` in SQL. Lexical retrieval uses Postgres `simple` text search, **not BM25**. Hybrid remains experimental because the diagnostic shows lexical ranking can amplify language-specific distractors.

## Evaluation and research

See evaluation/SOTA_REPORT.md and raw JSON; evaluation/REPORT.md is the historical E5/BGE baseline. The 24 verbatim feed descriptions have provisional family labels: grade 2 for the same activity, grade 1 for explicitly related activities, otherwise 0. These are not independent human judgments. Query families are preassigned development/held-out; both groups' documents remain in the retrieval corpus. No model is trained or fine-tuned. This small diagnostic is not a statistically reliable estimate of full-feed relevance. Separately, ANN agreement is evaluated on all 1,971 searchable company IDs; agreement with exact retrieval is not a human relevance judgment.

Research informs the choices without a claim of reproducing its models or achieving SOTA:

- [M3-Embedding, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.137/): multilingual and multiple retrieval signals.
- [BEIR, NeurIPS Datasets and Benchmarks 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html): compare lexical/neural quality and cost.
- [Reciprocal Rank Fusion, SIGIR 2009](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf): combine ranks without incompatible score scales.
- [MTEB, EACL 2023](https://aclanthology.org/2023.eacl-main.148/): task-specific model selection.
- [Alignment and Uniformity, ICML 2020](https://proceedings.mlr.press/v119/wang20k.html): motivates checking positive pairs and confusing negatives; no contrastive training is implemented.

## Verification and guardrails

Unit tests cover parsing/timestamps, vector validity, retries/cancellation and evaluation arithmetic. Database tests cover actual pgvector SQL, concurrency, rollback, deterministic ties, deletion ordering, model isolation, shuffled full-feed processing and safe post-commit redelivery. A real-RabbitMQ test verifies requeue after channel closure. CI runs the provided native service versions.

Native verification passed in GitHub Actions using the unchanged Compose file: PostgreSQL 16.16/pgvector 0.8.7 and RabbitMQ 3.13.7. All 52 tests passed with zero skips. The full official publisher → worker → SQL → live HTTP pipeline, exact-state audit and complete replay passed; see evaluation/results/native-ci and evaluation/results/tests-native.json. All 1,971 sources matched exact top-four retrieval; 5,913 HTTP requests had zero errors. Earlier local tests used PostgreSQL 18.7/WASM (PGlite 0.5.17); those results remain separately labeled and are not native performance claims. Actual consumer SIGKILL recovery was exercised in that earlier environment.

Include liveness/readiness, bounded-label API metrics, structured worker outcomes and strict feed accounting. Worker metrics are process-local; exposing them to a monitoring system is future deployment work. Raw rejected bodies are not logged. An offline re-embedding command is included; serving during that migration is explicitly unsupported.

## Production and 10× volume

Measure ingestion CPU, queue age, API p95 and index maintenance at 10× volume. Recheck the included partial HNSW index against exact retrieval, tune `ef_search`/candidate count, and account for filtering. The present full-population agreement check can become a sampled continuous regression gate as the dataset grows. Exact mode remains the reference and rollback path.

Batch cache misses and inference, then scale bounded worker processes. Keep atomic message claims and conditional versions. Duplicate inference under cache races wastes CPU but cannot corrupt state. Retain tombstones and dedupe history at least through the maximum replay horizon; blind expiry breaks recovery.

Provision durable volumes/managed storage, tested backups/PITR, broker persistence and disk alarms. The unchanged provided Compose file is local infrastructure, not production storage. Deploy API and worker separately; use a read-only DB role for serving, TLS, managed secrets, network restrictions and gateway auth/rate limits. Monitor quarantine growth, oldest queued event, processing failures, model identity, eligible-record coverage and API latency/errors. Set SLOs from product requirements and native measurements.

Use migrations for evolving schemas. For online model upgrades, maintain a versioned embedding set, evaluate/backfill while tracking source changes, then atomically switch serving version with rollback. The take-home's `reembed` command is a simpler offline migration.

## Limits

Independent bilingual labels, full-feed relevance judgments, database/broker host crash durability, sustained multi-worker load, production deployment and backup/restore drills are not claimed. Consumer-process SIGKILL recovery was exercised; it is not a host durability test. The public template-based repository and native CI verification are complete. Submitting the repository URL and résumé through the assessment form remains the candidate's final step.

Relevance labels are provisional and have not been independently human-reviewed. Models are pretrained open-source models; no custom training or SOTA performance is claimed.
