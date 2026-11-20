"""Strict envelope validation; nullable fields are retained as incomplete snapshots."""

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

MAX_BODY_BYTES = 1_048_576


class InvalidEvent(ValueError):
    """A permanent input error, suitable for durable quarantine."""


def reject_constant(value):
    raise InvalidEvent("non-finite JSON value")


def normalize_text(value: Any, field: str, maximum: int = 100_000) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidEvent(f"{field}: expected string or null")
    if "\x00" in value or len(value) > maximum:
        raise InvalidEvent(f"{field}: NUL character or oversized value")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidEvent(f"{field}: invalid Unicode scalar value") from exc
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if len(normalized) > maximum:
        raise InvalidEvent(f"{field}: normalized value exceeds limit")
    return normalized or None


def parse_timestamp(value: Any) -> datetime:
    try:
        if isinstance(value, int) and not isinstance(value, bool):
            result = datetime.fromtimestamp(value / 1000, UTC)
        elif isinstance(value, str):
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if result.tzinfo is None:
                raise ValueError("timezone required")
            result = result.astimezone(UTC)
        else:
            raise ValueError("expected aware ISO string or integer epoch milliseconds")
        if not 1900 <= result.year <= 2200:
            raise ValueError("outside supported source date range")
        return result
    except (ValueError, OverflowError, OSError) as exc:
        raise InvalidEvent("updatedAt: invalid or ambiguous timestamp") from exc


@dataclass(frozen=True)
class CompanyEvent:
    registration_number: str
    updated_at: datetime
    active: bool
    company_name: str | None
    nature_of_business: str | None
    state: str | None
    industry: str | None
    event_hash: str

    @property
    def priority(self) -> int:
        return 1 if self.active else 2

    @property
    def eligible(self) -> bool:
        return bool(self.active and self.company_name and self.nature_of_business)

    @property
    def version(self) -> tuple[datetime, int, str]:
        return self.updated_at, self.priority, self.event_hash


def parse_event(body: bytes) -> CompanyEvent:
    if len(body) > MAX_BODY_BYTES:
        raise InvalidEvent("message exceeds 1 MiB input limit")
    try:

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise InvalidEvent("duplicate JSON key")
                result[key] = value
            return result

        payload = json.loads(
            body,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise InvalidEvent("invalid JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidEvent("event must be a JSON object")
    event_type = payload.get("type")
    if event_type not in ("company-upserted", "company-struck-off"):
        raise InvalidEvent("unsupported event type")
    reg = normalize_text(payload.get("registrationNumber"), "registrationNumber", 128)
    if not reg:
        raise InvalidEvent("registrationNumber required")
    updated = parse_timestamp(payload.get("updatedAt"))
    active = event_type == "company-upserted"
    data = payload.get("data", {})
    if active and not isinstance(data, dict):
        raise InvalidEvent("data must be an object for upsert")
    fields = [
        normalize_text(data.get(k), k) if active else None
        for k in ("companyName", "natureOfBusiness", "state", "industry")
    ]
    canonical = json.dumps([reg, updated.isoformat(), active, *fields], ensure_ascii=False)
    return CompanyEvent(reg, updated, active, *fields, hashlib.sha256(canonical.encode()).hexdigest())
