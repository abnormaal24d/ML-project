"""Pure parsing of server-provided response rate-limit headers."""

from __future__ import annotations

import datetime
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime


def parse_retry_after_seconds(
    value: object,
    *,
    now: datetime.datetime,
) -> float | None:
    """Parse Retry-After seconds or an HTTP date into a finite delay."""

    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        numeric_delay = float(normalized)
    except ValueError:
        numeric_delay = None
    if numeric_delay is not None:
        if not math.isfinite(numeric_delay) or numeric_delay < 0:
            return None
        return numeric_delay
    try:
        retry_at = parsedate_to_datetime(normalized)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delay_seconds = max(0.0, (retry_at - now).total_seconds())
    return delay_seconds if math.isfinite(delay_seconds) else None


@dataclass(frozen=True, slots=True)
class ResponseRateLimitHints:
    """Server-provided host pacing hints extracted from response headers."""

    retry_after_seconds: float | None = None
    rate_limit_remaining: int | None = None
    rate_limit_reset_seconds: float | None = None

    @property
    def has_delay(self) -> bool:
        if (
            self.retry_after_seconds is not None
            and self.retry_after_seconds > 0
        ):
            return True
        return (
            self.rate_limit_remaining == 0
            and self.rate_limit_reset_seconds is not None
            and self.rate_limit_reset_seconds > 0
        )

    @classmethod
    def from_headers(
        cls,
        headers: Mapping[str, object],
        *,
        now: datetime.datetime,
    ) -> ResponseRateLimitHints:
        """Parse standard retry and rate-limit response headers."""

        normalized = {
            str(key).strip().lower(): str(value).strip()
            for key, value in headers.items()
            if str(key).strip()
        }
        return cls(
            retry_after_seconds=parse_retry_after_seconds(
                normalized.get("retry-after"),
                now=now,
            ),
            rate_limit_remaining=cls._parse_int_header(
                normalized.get("x-ratelimit-remaining")
            ),
            rate_limit_reset_seconds=cls._parse_rate_limit_reset(
                normalized.get("x-ratelimit-reset"),
                now=now,
            ),
        )

    @staticmethod
    def _parse_int_header(value: str | None) -> int | None:
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _parse_rate_limit_reset(
        value: str | None,
        *,
        now: datetime.datetime,
    ) -> float | None:
        if not value:
            return None
        try:
            numeric_value = float(value)
        except ValueError:
            return None
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            return None
        now_epoch = now.timestamp()
        if numeric_value > now_epoch:
            return max(0.0, numeric_value - now_epoch)
        return numeric_value
