import json
import logging
from datetime import UTC, datetime

from prometheus_client import Counter, Histogram

EVENTS = Counter("company_events_total", "Message processing outcomes", ["outcome"])
PROCESSING = Histogram("company_event_duration_seconds", "End-to-end processing duration")
REQUESTS = Counter("company_api_requests_total", "API requests", ["route", "status"])
LATENCY = Histogram("company_api_duration_seconds", "API request duration", ["route"])


def log(event: str, **fields):
    logging.getLogger("onecredit").info(
        json.dumps({"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields})
    )


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
