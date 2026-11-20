import hashlib
import json
import os
from dataclasses import replace
from uuid import uuid4

import pytest
import pytest_asyncio

from onecredit.config import Settings
from onecredit.db import initialize, make_pool
from onecredit.repository import Repository


class FakeEmbedder:
    """Deterministic, finite vectors for behavior tests, not quality claims."""

    def __init__(self, dimensions=384):
        self.calls = 0
        self.dimensions = dimensions

    def embed(self, text):
        self.calls += 1
        values = [0.0] * self.dimensions
        for word in text.lower().split():
            index = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dimensions
            values[index] += 1
        return values


def body(reg="TEST-001", timestamp="2026-08-14T00:00:00Z", description="bakery bread cakes", **extra):
    event = {
        "type": "company-upserted",
        "registrationNumber": reg,
        "updatedAt": timestamp,
        "data": {
            "companyName": reg,
            "natureOfBusiness": description,
            "industry": "Food & Beverage",
            "state": "Selangor",
        },
    }
    event.update(extra)
    return json.dumps(event).encode()


@pytest_asyncio.fixture
async def database():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("set TEST_DATABASE_URL to a disposable database; never uses the app database implicitly")
    from psycopg import AsyncConnection, sql

    # One schema per test avoids destructive truncation and permits concurrent runs.
    schema = "test_" + uuid4().hex
    base = Settings(database_url=url, model="intfloat/multilingual-e5-small", database_schema="public")
    await initialize(base)
    async with await AsyncConnection.connect(url, autocommit=True) as connection:
        await connection.execute(sql.SQL("CREATE SCHEMA {} ").format(sql.Identifier(schema)))
    settings = replace(base, database_schema=schema)
    await initialize(settings)
    try:
        async with make_pool(settings) as pool:
            await pool.wait()
            yield settings, Repository(pool, settings.embedding_key), FakeEmbedder()
    finally:
        async with await AsyncConnection.connect(url, autocommit=True) as connection:
            await connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
