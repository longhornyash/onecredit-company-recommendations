"""Build HNSW offline and activate it only after full-population distance validation."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from psycopg import sql

from onecredit.config import Settings
from onecredit.db import make_pool
from onecredit.embeddings import validate_vector
from onecredit.repository import Repository


async def build_and_verify(settings, output):
    started = time.perf_counter()
    candidate = settings.search_index_name + "_candidate"
    report = {
        "model": settings.embedding_key,
        "dimensions": settings.embedding_dimensions,
        "ef_search": 200,
        "candidate_pool": 64,
        "state": "running",
    }
    async with make_pool(settings, max_size=1) as pool:
        async with pool.connection() as connection:
            report["database_version"] = (await (await connection.execute("SELECT version() AS v")).fetchone())["v"]
            await connection.execute("SET LOCAL statement_timeout = '60s'")
            existing = await (
                await connection.execute(
                    "SELECT 1 FROM pg_indexes WHERE schemaname=%s AND indexname=%s",
                    (settings.database_schema, settings.search_index_name),
                )
            ).fetchone()
            if existing:
                raise RuntimeError("verified index already exists; use a fresh pipeline schema for a new comparison")
            await connection.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(candidate)))
            await connection.execute(
                sql.SQL("""CREATE INDEX {name} ON companies USING hnsw
                ((embedding::vector({dim})) vector_cosine_ops) WITH (m=16,ef_construction=64)
                WHERE is_active AND company_name IS NOT NULL AND embedding_key={key}""").format(
                    name=sql.Identifier(candidate),
                    dim=sql.SQL(str(settings.embedding_dimensions)),
                    key=sql.Literal(settings.embedding_key),
                )
            )
            sources = await (
                await connection.execute(
                    "SELECT registration_number,embedding FROM companies WHERE is_active AND company_name IS NOT NULL "
                    "AND embedding_key=%s ORDER BY registration_number",
                    (settings.embedding_key,),
                )
            ).fetchall()
        report["build_seconds"] = time.perf_counter() - started
        exact = Repository(pool, settings.embedding_key, settings.embedding_dimensions, "exact")
        ann = Repository(pool, settings.embedding_key, settings.embedding_dimensions, "hnsw")
        passed = 0
        id_recall = []
        failures = []
        for source in sources:
            reg = source["registration_number"]
            _, gold = await exact.similar(reg, 4, "dense")
            _, approximate = await ann.similar(reg, 4, "dense")
            cutoff = min((r["score"] for r in gold), default=-1)
            good = len(approximate) == len(gold) and all(r["score"] >= cutoff - 1e-6 for r in approximate)
            passed += good
            gold_ids = {r["registration_number"] for r in gold}
            id_recall.append(len(gold_ids & {r["registration_number"] for r in approximate}) / len(gold) if gold else 1)
            if not good:
                failures.append(reg)
        async with pool.connection() as connection:
            await connection.execute("SET LOCAL enable_seqscan = off")
            await connection.execute("SET LOCAL hnsw.ef_search = 200")
            vector = np.array(validate_vector(sources[0]["embedding"], settings.embedding_dimensions), dtype=np.float32)
            plan = await (
                await connection.execute(
                    sql.SQL("""EXPLAIN (FORMAT JSON)
                SELECT registration_number FROM companies WHERE is_active AND company_name IS NOT NULL
                AND embedding_key=%s AND registration_number<>%s
                ORDER BY embedding::vector({dim}) <=> %s LIMIT 64""").format(
                        dim=sql.SQL(str(settings.embedding_dimensions))
                    ),
                    (settings.embedding_key, sources[0]["registration_number"], vector),
                )
            ).fetchone()
            plan_text = json.dumps(plan, default=str)
            uses_index = candidate in plan_text and "Index Scan" in plan_text
            accepted = passed == len(sources) and bool(sources) and uses_index
            if accepted:
                await connection.execute(
                    sql.SQL("ALTER INDEX {} RENAME TO {}").format(
                        sql.Identifier(candidate), sql.Identifier(settings.search_index_name)
                    )
                )
            else:
                await connection.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(candidate)))
        report.update(
            {
                "state": "passed" if accepted else "rejected",
                "queries": len(sources),
                "tie_aware_top4_valid_fraction": passed / len(sources) if sources else 0,
                "strict_id_recall_at_4": sum(id_recall) / len(id_recall) if id_recall else 0,
                "queries_outside_exact_distance_cutoff": failures,
                "index_scan_verified": uses_index,
                "total_seconds": time.perf_counter() - started,
                "limitations": "Distance-cutoff agreement on this snapshot, not a guarantee for future updates. Exact ID recall penalizes equally good tied neighbours.",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not accepted:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/hnsw-verification.json"))
    args = parser.parse_args()
    asyncio.run(build_and_verify(Settings(), args.output))


if __name__ == "__main__":
    main()
