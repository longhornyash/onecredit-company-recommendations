"""One-command official publisher → RabbitMQ → worker → pgvector → HTTP verification.

Uses a new schema, never truncates existing company data or purges the shared queue.
The queue must be empty and have no other consumers; run on assessment services.
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aio_pika
from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row

from onecredit.config import Settings


async def broker_state(settings):
    connection = await aio_pika.connect_robust(settings.rabbitmq_url, timeout=5)
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue("company-events", durable=True)
        properties = connection.transport.connection.server_properties
        return {
            "ready_messages": queue.declaration_result.message_count,
            "consumers": queue.declaration_result.consumer_count,
            "product": str(properties.get("product")),
            "version": str(properties.get("version")),
        }


def consumer_summary(path):
    events = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and "event" in event:
            events.append(event)
    complete = next((e for e in events if e["event"] == "consumer_complete"), None)
    if complete is None:
        raise RuntimeError(f"consumer did not finish successfully; inspect {path}")
    messages = [e for e in events if e["event"] == "message_processed"]
    if messages:
        span = (
            datetime.fromisoformat(messages[-1]["timestamp"]) - datetime.fromisoformat(messages[0]["timestamp"])
        ).total_seconds() + messages[0]["duration_ms"] / 1000
        complete = {
            **complete,
            "processing_span_seconds": span,
            "messages_per_second": len(messages) / span if span else None,
            "sum_message_processing_seconds": sum(e["duration_ms"] for e in messages) / 1000,
        }
    return complete


async def run_pipeline(
    output_dir: Path, benchmark_requests=0, skip_quality=False, skip_model_benchmark=False, skip_index=False
):
    settings = Settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "state": "running",
        "model": settings.embedding_key,
        "embedding_dimensions": settings.embedding_dimensions,
        "feed_sha256": hashlib.sha256(Path("data/feed.jsonl").read_bytes()).hexdigest(),
        "stages": [],
    }
    report_path = output_dir / "pipeline.json"
    env = dict(os.environ)
    worker = None
    schema = f"pipeline_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    env["DATABASE_SCHEMA"] = schema
    env["SEARCH_MODE"] = "exact" if skip_index else "auto"
    report["database_schema"] = schema

    def save():
        report_path.write_text(json.dumps(report, indent=2))

    def command(stage, module, *args, timeout=300):
        print(json.dumps({"stage": stage, "state": "started"}), flush=True)
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-m", module, *map(str, args)], env=env, capture_output=True, text=True, timeout=timeout
        )
        (output_dir / f"{stage}.log").write_text(result.stdout + result.stderr)
        report["stages"].append(
            {"name": stage, "wall_seconds": time.perf_counter() - started, "exit_code": result.returncode}
        )
        save()
        if result.returncode:
            raise RuntimeError(f"{stage} failed; inspect {output_dir}/{stage}.log")
        print(
            json.dumps({"stage": stage, "state": "passed", "seconds": report["stages"][-1]["wall_seconds"]}), flush=True
        )
        return result.stdout

    try:
        before = await broker_state(settings)
        if before["ready_messages"] or before["consumers"]:
            raise RuntimeError(
                "assessment queue must be empty and have no other consumers; refusing to consume or purge existing work"
            )
        report["broker"] = before
        async with await AsyncConnection.connect(
            settings.database_url, autocommit=True, row_factory=dict_row, prepare_threshold=None
        ) as connection:
            version = (await (await connection.execute("SELECT version() AS v")).fetchone())["v"]
            report["database_version"] = version
            report["native_postgres16_verified"] = version.startswith("PostgreSQL 16.") and "PGlite" not in version
            await connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        command("initialize", "onecredit.cli", "init-db")

        worker_log = output_dir / "consumer.log"
        with worker_log.open("w") as log:
            started = time.perf_counter()
            worker = subprocess.Popen(
                [sys.executable, "-m", "onecredit.consumer", "--idle-seconds", "5"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 120
            while '"event": "consumer_ready"' not in worker_log.read_text():
                if worker.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("consumer failed to become ready; inspect consumer.log")
                await asyncio.sleep(0.1)
            report["consumer_startup_seconds"] = time.perf_counter() - started
            publish_start = time.perf_counter()
            command("publish", "producer.publish")
            await asyncio.to_thread(worker.wait, 300)
            if worker.returncode:
                raise RuntimeError("consumer failed; inspect consumer.log")
            report["publish_to_drain_wall_seconds"] = time.perf_counter() - publish_start
            report["drain_idle_shutdown_seconds"] = 5
        report["ingestion"] = consumer_summary(worker_log)
        audit = json.loads(command("audit", "onecredit.cli", "audit"))
        report["audit"] = audit
        (output_dir / "audit.json").write_text(json.dumps(audit, indent=2))
        if not skip_index:
            command("index_verification", "scripts.index_benchmark", "--output", output_dir / "hnsw-verification.json")
            report["search_algorithm"] = "verified HNSW with exact fallback"
        else:
            report["search_algorithm"] = "exact"
        demo = json.loads(command("http_demo", "scripts.demo"))
        (output_dir / "demo.json").write_text(json.dumps(demo, indent=2))
        requests = benchmark_requests or audit["expected_eligible"]
        command(
            "http_benchmark",
            "scripts.benchmark",
            "--output",
            output_dir / "http.json",
            "--requests",
            requests,
            "--all-companies",
            "--concurrency",
            1,
            8,
            32,
            timeout=300,
        )
        if not skip_quality:
            command("recommendation_evaluation", "evaluation.run", "--output", output_dir / "quality.json")
        if not skip_model_benchmark:
            command(
                "cpu_memory_benchmark", "scripts.model_benchmark", "--output", output_dir / "model-performance.json"
            )
        command("replay_publish", "producer.publish")
        command("replay_consume", "onecredit.consumer", "--idle-seconds", "2")
        replay = consumer_summary(output_dir / "replay_consume.log")
        expected = len(Path("data/feed.jsonl").read_bytes().splitlines())
        if replay["counts"] != {"duplicate": expected} or replay["model_inferences"] != 0:
            raise RuntimeError("full replay was not idempotent")
        report["replay"] = replay
        report["final_queue"] = await broker_state(settings)
        if report["final_queue"]["ready_messages"] or report["final_queue"]["consumers"]:
            raise RuntimeError("queue did not fully drain")
        report["state"] = "passed"
        report["limitations"] = [
            "Single-machine warm performance study, not a production SLO.",
            "Quality fixture has provisional judgments; independent review required.",
        ]
        if not report["native_postgres16_verified"]:
            report["limitations"].append("This run is not native PostgreSQL 16 verification; inspect database_version.")
    except BaseException as exc:
        report["state"] = "failed"
        report["failure_type"] = type(exc).__name__
        report["failure_reason"] = str(exc)
        raise
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                await asyncio.to_thread(worker.wait, 10)
            except subprocess.TimeoutExpired:
                worker.kill()
                await asyncio.to_thread(worker.wait)
        report["completed_at_utc"] = datetime.now(UTC).isoformat()
        save()
    print(
        json.dumps(
            {
                "state": report["state"],
                "report": str(report_path),
                "schema": schema,
                "native_postgres16_verified": report["native_postgres16_verified"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/pipeline-latest"))
    parser.add_argument("--benchmark-requests", type=int, default=0)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-model-benchmark", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.benchmark_requests <= 10000:
        parser.error("benchmark-requests must be 0–10000 (0 uses all eligible companies)")
    asyncio.run(
        run_pipeline(
            args.output_dir, args.benchmark_requests, args.skip_quality, args.skip_model_benchmark, args.skip_index
        )
    )


if __name__ == "__main__":
    main()
