import os
from uuid import uuid4

import aio_pika
import pytest

from onecredit.consumer import handle_message
from tests.conftest import body

pytestmark = [pytest.mark.integration, pytest.mark.rabbitmq]


async def test_real_broker_post_commit_redelivery(database):
    url = os.getenv("TEST_RABBITMQ_URL")
    if not url:
        pytest.skip("TEST_RABBITMQ_URL required; no fake broker substitutes for this test")
    _, repo, model = database
    connection = await aio_pika.connect_robust(url)
    async with connection:
        channel = await connection.channel(publisher_confirms=True)
        name = f"onecredit-test-{uuid4().hex}"
        queue = await channel.declare_queue(name, durable=True)
        try:
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body(), message_id="broker-redelivery", delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=name,
            )
            message = await queue.get(timeout=5)
            assert await repo.process(message.message_id, message.body, model) == "applied"
            # Simulate connection loss after DB commit, before acknowledgement.
            await channel.close()
            channel = await connection.channel()
            queue = await channel.declare_queue(name, durable=True)
            redelivered = await queue.get(timeout=5)
            assert redelivered.redelivered
            assert await handle_message(redelivered, repo, model) == "duplicate"
            assert await queue.get(fail=False, timeout=1) is None
        finally:
            if channel.is_closed:
                channel = await connection.channel()
            await channel.queue_delete(name)
