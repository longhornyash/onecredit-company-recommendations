import pytest

from onecredit.db import initialize
from onecredit.embeddings import validate_vector
from onecredit.repository import Repository
from tests.conftest import FakeEmbedder, body


def test_full_1024_vector_is_preserved():
    assert len(validate_vector([1] * 1024, 1024)) == 1024
    with pytest.raises(ValueError):
        validate_vector([1] * 384, 1024)


@pytest.mark.integration
async def test_384_schema_upgrade_and_1024_model_isolation(database):
    settings, old, model = database
    await old.process("old", body(reg="OLD"), model)
    async with old.pool.connection() as connection:
        await connection.execute("ALTER TABLE companies ALTER COLUMN embedding TYPE vector(384)")
        await connection.execute("ALTER TABLE embedding_cache ALTER COLUMN embedding TYPE vector(384)")
    await initialize(settings)
    new = Repository(old.pool, "harrier-test", 1024)
    for i in range(2):
        await new.process(f"new-{i}", body(reg=f"NEW-{i}"), FakeEmbedder(1024))
    status, cards = await new.similar("NEW-0", 4, "dense")
    assert status == "ok" and [c["registration_number"] for c in cards] == ["NEW-1"]
    assert await new.similar("OLD", 4, "dense") == ("unavailable", [])
    assert await old.similar("OLD", 4, "dense") == ("ok", [])
