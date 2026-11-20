# Implementation plan — assessment 2026.11

1. Preserve the official feed, publisher, assessment version and queue contract.
2. Validate events; normalize aware ISO timestamps and epoch milliseconds; retain incomplete company snapshots and deletion tombstones. Quarantine unusable events durably.
3. Commit message deduplication and version-conditional company mutation atomically. Acknowledge only after commit. Resolve equal timestamps deterministically (deletion first, then canonical payload hash).
4. Embed business descriptions on CPU with pinned leaderboard-selected Harrier 0.6B, compared against E5/BGE baselines. Cache by preprocessing/model identity and normalized text. Preserve full 1,024-d outputs and never mix vector spaces.
5. Serve typed FastAPI cards through verified model-specific HNSW, with exact reference/fallback and optional SQL lexical/RRF experiments. Activate ANN only after full-population agreement and query-plan checks.
6. Test real Postgres and RabbitMQ: replay, reorder, concurrency, tombstones, poison events, transaction rollback and post-commit redelivery. Separate deterministic-vector correctness tests from real-model quality evaluation.
7. Evaluate actual description families with bilingual relevance judgments, duplicate-text exclusion, held-out categories, nDCG@4, relevance@4, latency and ingestion accounting. Record evaluation limitations.
8. Provide locked dependencies, CI, run/reset/test/eval commands, health/readiness, structured logs and DESIGN.md. Verify from a clean database before delivery.

Time budget: 30m dependency/data inspection; 90m ingestion; 45m embeddings; 45m API; 45m tests; at most 60m quality deep dive; 45m documentation/demo. Stop development at six hours and disclose remaining limitations.

Execution result: the original E5 build passed 49 tests and process-crash recovery. The subsequent leaderboard/model/index upgrade passed 52 tests, then the one-command full Harrier/HNSW pipeline passed with real RabbitMQ 3.13.7 and PostgreSQL/WASM: official publisher → worker → database → index verification → live HTTP → quality/CPU metrics → full duplicate replay. All 1,971 searchable source companies were checked for ANN/exact top-four agreement and exercised at concurrency 1/8/32. See evaluation/SOTA_REPORT.md and raw evidence; the earlier E5/BGE artifacts remain as baselines.

Final verification: published the official-template-based repository as longhornyash/onecredit-company-recommendations. GitHub Actions ran the unchanged Compose stack (PostgreSQL 16.16/pgvector 0.8.7 and RabbitMQ 3.13.7): 52 tests passed, zero skips; the complete Harrier/HNSW pipeline passed. Native audit, replay, ANN agreement, quality, HTTP and CPU/RSS measurements are saved in evaluation/results/native-ci. Earlier local PostgreSQL/WASM results remain labeled as historical. No résumé/application has been submitted.
