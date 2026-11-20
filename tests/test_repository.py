import asyncio
import hashlib
import random
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from psycopg.errors import CheckViolation

from onecredit.api import create_app
from onecredit.cli import feed_reference
from onecredit.consumer import handle_message
from tests.conftest import body

pytestmark = pytest.mark.integration


async def test_duplicate_stale_and_metadata_reuses_vector(database):
    settings, repo, model = database
    assert await repo.process("new", body(timestamp="2026-08-16T00:00:00Z"), model) == "applied"
    assert await repo.process("new", body(timestamp="2026-08-16T00:00:00Z"), model) == "duplicate"
    assert await repo.process("old", body(), model) == "stale"
    assert await repo.process("metadata", body(timestamp="2026-08-17T00:00:00Z"), model) == "applied"
    assert model.calls == 1
    assert (await repo.company("TEST-001"))["source_updated_at"].day == 17


async def test_tombstone_first_and_newer_reactivation(database):
    _, repo, model = database
    await repo.process("delete", body(timestamp="2026-08-16T00:00:00Z", type="company-struck-off"), model)
    assert await repo.process("old", body(), model) == "stale"
    assert (await repo.company("TEST-001"))["is_active"] is False
    await repo.process("reactivate", body(timestamp="2026-08-17T00:00:00Z"), model)
    assert (await repo.company("TEST-001"))["is_active"] is True


async def test_poison_and_id_collision_are_durable(database):
    _, repo, model = database
    assert await repo.process("bad", b"null", model) == "rejected"
    assert await repo.process("bad", b"null", model) == "duplicate"
    assert await repo.process(None, body(), model) == "rejected"
    await repo.process("good", body(), model)
    assert await repo.process("good", body(reg="TEST-002"), model) == "rejected"
    assert await repo.company("TEST-002") is None
    async with repo.pool.connection() as connection:
        row = await (await connection.execute("SELECT count(*) AS n FROM rejected_events")).fetchone()
    assert row["n"] == 3


async def test_concurrent_claim_and_source_order(database):
    _, repo, model = database
    assert (
        sorted(await asyncio.gather(*(repo.process("same", body(), model) for _ in range(5))))
        == ["applied"] + ["duplicate"] * 4
    )
    events = [body(timestamp=f"2026-08-{day:02}T00:00:00Z") for day in range(15, 25)]
    await asyncio.gather(*(repo.process(str(i), payload, model) for i, payload in enumerate(reversed(events))))
    assert (await repo.company("TEST-001"))["source_updated_at"].day == 24


async def test_failed_transaction_rolls_back_dedup_claim(database):
    _, repo, model = database
    async with repo.pool.connection() as connection:
        await connection.execute(
            "ALTER TABLE companies ADD CONSTRAINT fail_test CHECK (registration_number <> 'TEST-001')"
        )
    with pytest.raises(CheckViolation):
        await repo.process("retry", body(), model)
    async with repo.pool.connection() as connection:
        assert (
            await (await connection.execute("SELECT * FROM processed_messages WHERE message_id='retry'")).fetchone()
            is None
        )
        await connection.execute("ALTER TABLE companies DROP CONSTRAINT fail_test")
    assert await repo.process("retry", body(), model) == "applied"


async def test_post_commit_ack_failure_redelivers_safely(database):
    _, repo, model = database

    class Message:
        message_id = "ack-test"
        body = body()

        async def ack(self):
            raise ConnectionError("simulated connection loss after commit")

    message = Message()
    with pytest.raises(ConnectionError):
        await handle_message(message, repo, model)
    assert await repo.process(message.message_id, message.body, model) == "duplicate"


async def test_api_contract_and_filters(database):
    settings, repo, model = database
    for i, text in enumerate(["bakery bread cakes", "bakery bread pastries", "cloud software hosting", "bakery bread"]):
        await repo.process(str(i), body(reg=f"TEST-{i}", description=text), model)
    await repo.process("delete", body(reg="TEST-3", timestamp="2026-09-01T00:00:00Z", type="company-struck-off"), model)
    await repo.process("incomplete", body(reg="TEST-empty", description=None), model)
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        for method in ("dense", "hybrid"):
            response = await client.get(f"/companies/TEST-0/similar?limit=2&method={method}")
            assert response.status_code == 200, response.text
            results = response.json()["results"]
            assert len(results) == 2
            assert results[0]["registrationNumber"] == "TEST-1"
            assert all(r["registrationNumber"] not in ("TEST-0", "TEST-3", "TEST-empty") for r in results)
            assert results[0]["score"] >= results[1]["score"]
        assert (await client.get("/companies/TEST-0/similar?limit=21")).status_code == 422
        assert (await client.get("/companies/TEST-0/similar?method=bad")).status_code == 422
        assert (await client.get("/companies/TEST-3/similar")).status_code == 404
        assert (await client.get("/companies/missing/similar")).status_code == 404
        assert (await client.get("/companies/TEST-empty/similar")).status_code == 409
        assert (await client.get("/companies/TEST-empty")).json()["recommendationEligible"] is False
        assert (await client.get("/ready")).status_code == 200
        assert (await client.get("/companies/" + "A" * 129)).status_code == 422
        assert (await client.get("/companies/%27%20OR%201%3D1--/similar")).status_code == 404


async def test_full_feed_random_order_replay(database):
    _, repo, model = database
    lines = Path("data/feed.jsonl").read_bytes().splitlines()
    random.Random(42).shuffle(lines)
    for raw in lines:
        await repo.process(hashlib.sha256(raw).hexdigest(), raw, model)
    expected, _, bodies = feed_reference(Path("data/feed.jsonl"))
    async with repo.pool.connection() as connection:
        actual = await (await connection.execute("SELECT registration_number,event_hash FROM companies")).fetchall()
        count = await (await connection.execute("SELECT count(*) AS n FROM processed_messages")).fetchone()
    assert {r["registration_number"]: r["event_hash"] for r in actual} == {r: e.event_hash for r, e in expected.items()}
    assert count["n"] == len(bodies)
    for raw in lines:
        assert await repo.process(hashlib.sha256(raw).hexdigest(), raw, model) == "duplicate"


async def test_equal_timestamp_conflicts_converge_in_any_order(database):
    _, repo, model = database
    from onecredit.events import parse_event

    for order in range(6):
        reg = f"TIE-{order}"
        payloads = [
            body(reg=reg, description="bread"),
            body(reg=reg, description="software"),
            body(reg=reg, type="company-struck-off"),
        ]
        random.Random(order).shuffle(payloads)
        for i, raw in enumerate(payloads):
            await repo.process(f"{reg}-{i}", raw, model)
        event = max(map(parse_event, payloads), key=lambda e: e.version)
        row = await repo.company(reg)
        assert row["is_active"] == event.active is False


async def test_incomplete_update_removes_old_embedding(database):
    _, repo, model = database
    await repo.process("valid", body(), model)
    await repo.process("empty", body(description=None, timestamp="2026-09-01T00:00:00Z"), model)
    row = await repo.company("TEST-001")
    assert row["nature_of_business"] is None
    assert row["has_embedding"] is False
    assert await repo.similar("TEST-001", 4, "dense") == ("unavailable", [])


async def test_model_spaces_are_never_mixed(database):
    _, repo, model = database
    await repo.process("source", body(reg="SOURCE"), model)
    await repo.process("candidate", body(reg="CANDIDATE"), model)
    async with repo.pool.connection() as connection:
        await connection.execute(
            "UPDATE companies SET embedding_key='another-model' WHERE registration_number='CANDIDATE'"
        )
    assert await repo.similar("SOURCE", 4, "dense") == ("ok", [])
    assert await repo.similar("CANDIDATE", 4, "dense") == ("unavailable", [])


async def test_database_errors_are_bounded_and_do_not_leak_details(database):
    from unittest.mock import AsyncMock

    from psycopg import OperationalError

    settings, _, _ = database
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        app.state.repository.similar = AsyncMock(side_effect=OperationalError("sensitive connection details"))
        response = await client.get("/companies/TEST-0/similar")
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "5"
        assert "sensitive" not in response.text
        metrics = (await client.get("/metrics")).text
        assert 'route="/companies/{registration_number}/similar"' in metrics
        assert 'route="/companies/TEST-0/similar"' not in metrics
