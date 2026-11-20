"""Closed-corpus bilingual retrieval diagnostic; all ranking executes in pgvector SQL."""

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from onecredit.cli import feed_reference
from onecredit.config import Settings
from onecredit.db import initialize, make_pool
from onecredit.embeddings import Embedder


def ndcg(grades, ideal, k=4):
    def dcg(values):
        return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(values[:k]))

    denominator = dcg(sorted(ideal, reverse=True))
    return dcg(grades) / denominator if denominator else 0.0


async def evaluate(settings: Settings, cases: Path, output: Path):
    spec = json.loads(cases.read_text())
    documents = [
        {
            "id": f"{family['id']}-{lang}",
            "family": family["id"],
            "language": lang,
            "split": family["split"],
            "text": family[lang],
        }
        for family in spec["families"]
        for lang in ("en", "ms")
    ]
    current, _, _ = feed_reference(Path("data/feed.jsonl"))
    observed = {event.nature_of_business for event in current.values()}
    assert all(doc["text"] in observed for doc in documents), "evaluation text must come from official feed"
    related = {frozenset(pair) for pair in spec["related_family_pairs"]}
    by_id = {doc["id"]: doc for doc in documents}

    def grade(query, candidate):
        if query["family"] == candidate["family"]:
            return 2
        return 1 if frozenset((query["family"], candidate["family"])) in related else 0

    started = time.perf_counter()
    embedder = await asyncio.to_thread(Embedder, settings)
    vectors = {doc["id"]: await asyncio.to_thread(embedder.embed, doc["text"]) for doc in documents}
    embedding_seconds = time.perf_counter() - started
    await initialize(settings)
    results = []
    async with make_pool(settings, max_size=1) as pool, pool.connection() as connection:
        database_version = (await (await connection.execute("SELECT version() AS version")).fetchone())["version"]
        await connection.execute("DROP TABLE IF EXISTS pg_temp.eval_documents")
        await connection.execute("""CREATE TEMP TABLE eval_documents(
            id TEXT PRIMARY KEY, family TEXT, language TEXT, description TEXT,
            embedding vector, tokens tsvector) ON COMMIT PRESERVE ROWS""")
        for doc in documents:
            await connection.execute(
                "INSERT INTO eval_documents VALUES (%s,%s,%s,%s,%s,to_tsvector('simple',%s))",
                (
                    doc["id"],
                    doc["family"],
                    doc["language"],
                    doc["text"],
                    np.array(vectors[doc["id"]], dtype=np.float32),
                    doc["text"],
                ),
            )
        for task in ("different_description", "cross_language"):
            for query in documents:
                candidates = [
                    doc
                    for doc in documents
                    if doc["text"] != query["text"]
                    and (task != "cross_language" or doc["language"] != query["language"])
                ]
                ids = [doc["id"] for doc in candidates]
                params = {"ids": ids, "vector": np.array(vectors[query["id"]], dtype=np.float32), "text": query["text"]}
                for method in ("dense", "lexical", "hybrid"):
                    started = time.perf_counter()
                    rows = await (
                        await connection.execute(
                            """
                      WITH candidates AS MATERIALIZED (SELECT * FROM eval_documents WHERE id=ANY(%(ids)s)),
                      dense AS (SELECT id,row_number() OVER (ORDER BY embedding <=> %(vector)s,id) AS rank
                                FROM candidates),
                      query AS (SELECT websearch_to_tsquery('simple',array_to_string(
                                tsvector_to_array(to_tsvector('simple',%(text)s)),' OR ')) AS q),
                      lexical AS (SELECT id,row_number() OVER (ORDER BY ts_rank_cd(tokens,q) DESC,id) AS rank
                                  FROM candidates,query WHERE tokens @@ q),
                      votes AS (
                        SELECT id,1.0/(60+rank) AS score,'dense' AS method FROM dense WHERE rank<=30
                        UNION ALL SELECT id,1.0/(60+rank),'lexical' FROM lexical WHERE rank<=30
                      )
                      SELECT id,sum(score)::float8 AS score FROM votes WHERE
                        (%(method)s='hybrid' OR method=%(method)s)
                      GROUP BY id ORDER BY score DESC,id LIMIT 4
                      """,
                            {**params, "method": method},
                        )
                    ).fetchall()
                    elapsed = (time.perf_counter() - started) * 1000
                    grades = [grade(query, by_id[row["id"]]) for row in rows]
                    results.append(
                        {
                            "query": query["id"],
                            "split": query["split"],
                            "task": task,
                            "method": method,
                            "ndcg_at_4": ndcg(grades, [grade(query, d) for d in candidates]),
                            "relevance_at_4": sum(g > 0 for g in grades) / 4,
                            "same_activity_at_1": int(bool(grades) and grades[0] == 2),
                            "sql_ms": elapsed,
                            "returned": [row["id"] for row in rows],
                            "grades": grades,
                        }
                    )
        await connection.execute("DROP TABLE eval_documents")
    summary = []
    for split in ("development", "held_out"):
        for task in ("different_description", "cross_language"):
            for method in ("dense", "lexical", "hybrid"):
                group = [r for r in results if (r["split"], r["task"], r["method"]) == (split, task, method)]
                summary.append(
                    {
                        "split": split,
                        "task": task,
                        "method": method,
                        "queries": len(group),
                        **{
                            metric: statistics.mean(r[metric] for r in group)
                            for metric in ("ndcg_at_4", "relevance_at_4", "same_activity_at_1")
                        },
                        "sql_p50_ms": statistics.median(r["sql_ms"] for r in group),
                        "sql_p95_ms": sorted(r["sql_ms"] for r in group)[math.ceil(0.95 * len(group)) - 1],
                    }
                )
    report = {
        "evaluation_version": spec["version"],
        "database_version": database_version,
        "cases_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
        "model": settings.embedding_key,
        "documents": len(documents),
        "embedding_load_and_inference_seconds": embedding_seconds,
        "provenance": spec["provenance"],
        "limitations": [
            "24-document closed corpus; not full-feed recommendation quality",
            "provisional labels; independent bilingual review pending",
            "SQL timing includes Python/client overhead; not HTTP or production latency",
            "held-out activity families; no fine-tuning or parameter tuning on these labels",
        ],
        "summary": summary,
        "queries": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"model": settings.model, "summary": summary, "output": str(output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/evaluation.json"))
    args = parser.parse_args()
    asyncio.run(evaluate(Settings(), args.cases, args.output))


if __name__ == "__main__":
    main()
