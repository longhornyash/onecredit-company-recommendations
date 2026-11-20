import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from onecredit.cli import feed_reference
from onecredit.embeddings import validate_vector
from onecredit.events import InvalidEvent, normalize_text, parse_event, parse_timestamp
from tests.conftest import body


@pytest.mark.parametrize(
    "payload", [b"null", b"[]", b"{", b'"hello"', b"{}", b'{"type":"company-upserted","type":"other"}', b"\xff"]
)
def test_rejects_poison(payload):
    with pytest.raises(InvalidEvent):
        parse_event(payload)


@pytest.mark.parametrize("value", [None, True, 1.5, "2026-08-14", "not-a-date", 10**40])
def test_ambiguous_timestamps_rejected(value):
    with pytest.raises(InvalidEvent):
        parse_timestamp(value)


def test_epoch_milliseconds_and_timezone_agree():
    expected = datetime(2026, 8, 14, tzinfo=UTC)
    assert parse_timestamp(int(expected.timestamp() * 1000)) == expected
    assert parse_timestamp("2026-08-14T08:00:00+08:00") == expected


def test_incomplete_company_is_preserved():
    event = parse_event(body(description=None))
    assert event.registration_number == "TEST-001"
    assert not event.eligible


def test_tombstone_wins_equal_timestamp():
    upsert = parse_event(body())
    deletion = parse_event(body(type="company-struck-off", data=None))
    assert deletion.version > upsert.version
    assert not deletion.active


def test_normalization_and_canonical_tie_break():
    first = body(description="  bread\n cakes  ")
    second = json.dumps(json.loads(first), sort_keys=True).encode()
    assert parse_event(first).event_hash == parse_event(second).event_hash
    assert normalize_text("Ｓｅｌａｎｇｏｒ", "state") == "Selangor"


@pytest.mark.parametrize("description", ["\ud800", "bread\x00cake", 123, True, "x" * 100001])
def test_bad_business_text_is_rejected_without_crashing(description):
    with pytest.raises(InvalidEvent):
        parse_event(body(description=description))


def test_normalization_cannot_expand_registration_past_api_limit():
    # A compatibility character expands into several Unicode scalar values.
    with pytest.raises(InvalidEvent):
        normalize_text("\ufdfa" * 10, "registrationNumber", maximum=128)


@pytest.mark.parametrize("vector", [[0] * 384, [1] * 383, [float("nan")] * 384, [float("inf")] * 384])
def test_invalid_vectors_fail_closed(vector):
    with pytest.raises(ValueError):
        validate_vector(vector)


def test_official_feed_accounting():
    latest, rejected, bodies = feed_reference(Path("data/feed.jsonl"))
    assert len(bodies) == 2450
    assert len(latest) >= 1900
    assert len(rejected) > 0
    assert any(not event.active for event in latest.values())
