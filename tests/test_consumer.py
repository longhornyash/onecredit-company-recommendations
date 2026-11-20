import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from onecredit.consumer import handle_message, process_with_retry


async def test_transient_error_retries_then_acknowledges_once():
    message = SimpleNamespace(message_id="test", body=b"{}", ack=AsyncMock())
    repository = SimpleNamespace(process=AsyncMock(side_effect=[TimeoutError(), "applied"]))
    sleep = AsyncMock()
    assert await process_with_retry(message, repository, None, sleep=sleep) == "applied"
    assert repository.process.await_count == 2
    message.ack.assert_awaited_once()
    sleep.assert_awaited_once()
    assert 1 <= sleep.call_args.args[0] < 2


async def test_exhausted_failure_never_acknowledges():
    message = SimpleNamespace(message_id="test", body=b"{}", ack=AsyncMock())
    repository = SimpleNamespace(process=AsyncMock(side_effect=RuntimeError("offline")))
    sleep = AsyncMock()
    with pytest.raises(RuntimeError, match="offline"):
        await process_with_retry(message, repository, None, sleep=sleep)
    assert repository.process.await_count == 5
    assert sleep.await_count == 4
    message.ack.assert_not_awaited()


async def test_cancellation_never_acknowledges_or_retries():
    message = SimpleNamespace(message_id="test", body=b"{}", ack=AsyncMock())
    repository = SimpleNamespace(process=AsyncMock(side_effect=asyncio.CancelledError()))
    sleep = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        await process_with_retry(message, repository, None, sleep=sleep)
    sleep.assert_not_awaited()
    message.ack.assert_not_awaited()


@pytest.mark.parametrize("outcome", ["applied", "stale", "duplicate", "rejected"])
async def test_every_committed_outcome_is_acknowledged(outcome):
    message = SimpleNamespace(message_id="test", body=b"{}", ack=AsyncMock())
    repository = SimpleNamespace(process=AsyncMock(return_value=outcome))
    assert await handle_message(message, repository, None) == outcome
    message.ack.assert_awaited_once()
