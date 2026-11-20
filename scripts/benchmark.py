"""Bounded live-HTTP smoke benchmark; report local measurements, not production claims."""

import argparse
import asyncio
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

from onecredit.config import Settings
from onecredit.db import make_pool


async def benchmark(output: Path, requests: int = 200, all_companies=False, concurrencies=(1, 8)):
    settings = Settings()
    async with make_pool(settings, max_size=1) as pool, pool.connection() as connection:
        rows = await (
            await connection.execute(
                "SELECT registration_number FROM companies WHERE is_active AND embedding_key=%s "
                "ORDER BY registration_number LIMIT %s",
                (settings.embedding_key, 1_000_000 if all_companies else 100),
            )
        ).fetchall()
        version = (await (await connection.execute("SELECT version() AS v")).fetchone())["v"]
        index = await (
            await connection.execute(
                "SELECT 1 FROM pg_indexes WHERE schemaname=%s AND indexname=%s",
                (settings.database_schema, settings.search_index_name),
            )
        ).fetchone()
        algorithm = "hnsw" if index and settings.search_mode != "exact" else "exact"
    if not rows:
        raise RuntimeError("ingest data before benchmarking")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "onecredit.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "18081",
            "--no-access-log",
        ]
    )
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:18081", timeout=10) as client:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("API failed to start")
                try:
                    if (await client.get("/ready")).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError("API readiness timed out")
            for _ in range(20):
                response = await client.get(f"/companies/{rows[0]['registration_number']}/similar")
                response.raise_for_status()
            measurements = []
            for concurrency in concurrencies:
                semaphore = asyncio.Semaphore(concurrency)

                async def request(i):
                    reg = rows[i % len(rows)]["registration_number"]
                    async with semaphore:
                        started = time.perf_counter()
                        response = await client.get(f"/companies/{reg}/similar?limit=4")
                        duration = (time.perf_counter() - started) * 1000
                        response.raise_for_status()
                        cards = response.json()["results"]
                        assert len(cards) == 4 and all(c["registrationNumber"] != reg for c in cards)
                        assert all(cards[j]["score"] >= cards[j + 1]["score"] for j in range(len(cards) - 1))
                        return duration

                started = time.perf_counter()
                durations = await asyncio.gather(*(request(i) for i in range(requests)))
                total = time.perf_counter() - started
                measurements.append(
                    {
                        "concurrency": concurrency,
                        "requests": requests,
                        "distinct_source_companies": min(requests, len(rows)),
                        "errors": 0,
                        "p50_ms": statistics.median(durations),
                        "p95_ms": sorted(durations)[math.ceil(0.95 * requests) - 1],
                        "max_ms": max(durations),
                        "requests_per_second": requests / total,
                    }
                )
            report = {
                "runtime": sys.version,
                "platform": platform.platform(),
                "database_version": version,
                "model": settings.embedding_key,
                "source_company_population": len(rows),
                "method": "dense",
                "search_algorithm": algorithm,
                "measurements": measurements,
                "limitations": "Local warm HTTP smoke test; no durability, availability, or production SLO claim.",
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
    finally:
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, 10)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/http-benchmark.json"))
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--all-companies", action="store_true")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8])
    args = parser.parse_args()
    if not 1 <= args.requests <= 10000 or any(c < 1 or c > 64 for c in args.concurrency):
        parser.error("requests must be 1–10000 and concurrency 1–64")
    asyncio.run(benchmark(args.output, args.requests, args.all_companies, args.concurrency))


if __name__ == "__main__":
    main()
