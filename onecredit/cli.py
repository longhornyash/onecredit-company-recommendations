"""Explicit database initialization, audit and model migration commands."""

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from onecredit.config import Settings
from onecredit.db import initialize, make_pool
from onecredit.embeddings import Embedder, validate_vector
from onecredit.events import InvalidEvent, parse_event
from onecredit.repository import Repository


def feed_reference(path: Path):
    latest = {}
    rejected = Counter()
    bodies = set()
    for body in path.read_bytes().splitlines():
        bodies.add(hashlib.sha256(body).hexdigest())
        try:
            event = parse_event(body)
        except InvalidEvent as exc:
            rejected[str(exc)] += 1
            continue
        old = latest.get(event.registration_number)
        if old is None or event.version > old.version:
            latest[event.registration_number] = event
    return latest, rejected, bodies


async def audit(settings: Settings, feed: Path):
    latest, rejected, bodies = feed_reference(feed)
    async with make_pool(settings) as pool, pool.connection() as connection:
        rows = await (
            await connection.execute("SELECT registration_number,event_hash,is_active,embedding_key FROM companies")
        ).fetchall()
        actual = {row["registration_number"]: row for row in rows}
        mismatches = [
            reg for reg, event in latest.items() if reg not in actual or actual[reg]["event_hash"] != event.event_hash
        ]
        processed = await (await connection.execute("SELECT body_hash,outcome FROM processed_messages")).fetchall()
        processed_hashes = {row["body_hash"] for row in processed}
        report = {
            "feed_messages": len(feed.read_bytes().splitlines()),
            "feed_unique_bodies": len(bodies),
            "expected_company_records": len(latest),
            "actual_company_records": len(actual),
            "expected_active": sum(e.active for e in latest.values()),
            "expected_eligible": sum(e.eligible for e in latest.values()),
            "actual_searchable": sum(r["is_active"] and r["embedding_key"] == settings.embedding_key for r in rows),
            "rejected_feed_lines_by_reason": dict(rejected),
            "processing_outcomes": dict(Counter(row["outcome"] for row in processed)),
            "missing_message_bodies": len(bodies - processed_hashes),
            "company_version_mismatches": mismatches,
            "unexpected_company_records": sorted(actual.keys() - latest.keys()),
            "unexpected_message_bodies": len(processed_hashes - bodies),
        }
        report["passed"] = (
            not mismatches
            and not (bodies - processed_hashes)
            and not report["unexpected_company_records"]
            and not report["unexpected_message_bodies"]
            and (report["actual_searchable"] == report["expected_eligible"])
        )
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)


async def reembed(settings: Settings):
    """Offline migration. Stop consumer/API; then restart both with the new model."""
    embedder = await asyncio.to_thread(Embedder, settings)
    async with make_pool(settings) as pool:
        async with pool.connection() as connection:
            rows = await (
                await connection.execute(
                    "SELECT registration_number,nature_of_business,event_hash FROM companies WHERE is_active "
                    "AND company_name IS NOT NULL AND nature_of_business IS NOT NULL ORDER BY registration_number"
                )
            ).fetchall()
        for row in rows:
            vector = np.array(
                validate_vector(
                    await asyncio.to_thread(embedder.embed, row["nature_of_business"]), settings.embedding_dimensions
                ),
                dtype=np.float32,
            )
            text_hash = hashlib.sha256(row["nature_of_business"].encode()).hexdigest()
            async with pool.connection() as connection:
                await connection.execute(
                    "UPDATE companies SET embedding=%s,embedding_key=%s WHERE registration_number=%s "
                    "AND event_hash=%s AND is_active",
                    (vector, settings.embedding_key, row["registration_number"], row["event_hash"]),
                )
                await connection.execute(
                    "INSERT INTO embedding_cache VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                    (settings.embedding_key, text_hash, vector),
                )
    print(json.dumps({"reembedded": len(rows), "embedding_key": settings.embedding_key}))


async def ingest_file(settings: Settings, feed: Path):
    """Recovery/demo utility. Normal ingestion uses the RabbitMQ consumer."""
    import time

    started = time.perf_counter()
    embedder = await asyncio.to_thread(Embedder, settings)
    counts = Counter()
    async with make_pool(settings) as pool:
        repository = Repository(pool, settings.embedding_key, settings.embedding_dimensions)
        for raw in feed.read_bytes().splitlines():
            counts[await repository.process(hashlib.sha256(raw).hexdigest(), raw, embedder)] += 1
    print(
        json.dumps(
            {
                "transport": "file-recovery",
                "counts": dict(counts),
                "model_inferences": embedder.calls,
                "seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init-db", "audit", "reembed", "inspect-feed", "ingest-file"])
    parser.add_argument("--feed", type=Path, default=Path("data/feed.jsonl"))
    args = parser.parse_args()
    settings = Settings()
    if args.command == "init-db":
        asyncio.run(initialize(settings))
    elif args.command == "audit":
        asyncio.run(audit(settings, args.feed))
    elif args.command == "reembed":
        asyncio.run(reembed(settings))
    elif args.command == "ingest-file":
        asyncio.run(ingest_file(settings, args.feed))
    else:
        latest, rejected, bodies = feed_reference(args.feed)
        print(
            json.dumps(
                {
                    "unique_bodies": len(bodies),
                    "company_records": len(latest),
                    "active": sum(e.active for e in latest.values()),
                    "eligible": sum(e.eligible for e in latest.values()),
                    "rejections": dict(rejected),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
