import json

import pytest

from onecredit.repository import Repository
from scripts.index_benchmark import build_and_verify
from tests.conftest import body


@pytest.mark.integration
async def test_only_verified_hnsw_is_activated(database, tmp_path):
    settings, repo, model = database
    for i, text in enumerate(
        [
            "bakery bread cakes",
            "bakery bread pastries",
            "software cloud hosting",
            "freight trucking delivery",
            "accounting tax finance",
            "bread pastries retail",
        ]
    ):
        await repo.process(str(i), body(reg=f"ANN-{i}", description=text), model)
    output = tmp_path / "index.json"
    await build_and_verify(settings, output)
    report = json.loads(output.read_text())
    assert report["state"] == "passed" and report["index_scan_verified"]
    assert report["strict_id_recall_at_4"] == 1
    ann = Repository(repo.pool, settings.embedding_key, 384, "hnsw")
    assert await ann.similar("ANN-0", 4, "dense") == await repo.similar("ANN-0", 4, "dense")
    assert await ann.similar("ANN-0", 5, "dense") == await repo.similar("ANN-0", 5, "dense")
