"""Run continuously, or drain the queue with --idle-seconds for a reproducible demo."""

import argparse
import asyncio
import random
import time
from collections import Counter

import aio_pika

from onecredit.config import Settings
from onecredit.db import make_pool
from onecredit.embeddings import Embedder
from onecredit.observability import EVENTS, PROCESSING, configure_logging, log
from onecredit.repository import Repository

QUEUE = "company-events"


async def handle_message(message, repository, embedder):
    started = time.perf_counter()
    outcome = await repository.process(message.message_id, message.body, embedder)
    # Deliberately outside the DB transaction: redelivery after an ack failure is safe.
    await message.ack()
    EVENTS.labels(outcome=outcome).inc()
    PROCESSING.observe(time.perf_counter() - started)
    log(
        "message_processed",
        message_id=message.message_id,
        outcome=outcome,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return outcome


async def process_with_retry(message, repository, embedder, attempts=5, sleep=None):
    sleep = sleep or asyncio.sleep
    for attempt in range(attempts):
        try:
            return await handle_message(message, repository, embedder)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            EVENTS.labels(outcome="processing_error").inc()
            log("processing_error", message_id=message.message_id, error_type=type(exc).__name__, attempt=attempt + 1)
            if attempt == attempts - 1:
                # Fail visibly; connection cleanup requeues unfinished work.
                raise
            await sleep(min(2**attempt, 16) + random.random())


async def consume(settings: Settings, idle_seconds: float = 0):
    embedder = await asyncio.to_thread(Embedder, settings)
    counts = Counter()
    async with make_pool(settings, max_size=2) as pool:
        await pool.wait()
        repository = Repository(pool, settings.embedding_key, settings.embedding_dimensions)
        connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=8)
            queue = await channel.declare_queue(QUEUE, durable=True)
            log("consumer_ready", embedding_key=settings.embedding_key)
            async with queue.iterator(timeout=idle_seconds or None) as iterator:
                while True:
                    try:
                        message = await anext(iterator)
                    except StopAsyncIteration:
                        break
                    except TimeoutError:
                        if idle_seconds:
                            break
                        raise
                    counts[await process_with_retry(message, repository, embedder)] += 1
    log("consumer_complete", counts=dict(counts), model_inferences=embedder.calls)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.idle_seconds < 0:
        parser.error("idle-seconds cannot be negative")
    configure_logging()
    try:
        asyncio.run(consume(Settings(), args.idle_seconds))
    except KeyboardInterrupt:
        log("consumer_stopped")


if __name__ == "__main__":
    main()
