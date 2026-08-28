from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING

from config.environment.default_values import (
    DEFAULT_LOG_RATE_LIMIT_MAX_ENTRIES,
)
from logger.serializers import record_event_name

if TYPE_CHECKING:
    from collections.abc import Iterable

    from config.settings.logging import EventRateLimitRulesSettings


class ConsoleNoiseFilter(logging.Filter):
    """Keep routine machine events in the JSON file log only."""

    def __init__(self, suppressed_events: tuple[str, ...]) -> None:
        super().__init__()
        self._suppressed_events = frozenset(suppressed_events)

    def filter(self, record: logging.LogRecord) -> bool:
        return record_event_name(record) not in self._suppressed_events


@dataclass(slots=True)
class EventRateLimitFilter(logging.Filter):
    """Suppress repeated structured events according to configured governance."""

    default_min_interval_sec: float
    governance: dict[str, EventRateLimitRulesSettings] = field(
        default_factory=dict
    )
    max_entries: int = DEFAULT_LOG_RATE_LIMIT_MAX_ENTRIES
    _last_seen: OrderedDict[tuple[object, ...], float] = field(
        default_factory=OrderedDict
    )
    _lock: RLock = field(default_factory=RLock)

    def filter(self, record: logging.LogRecord) -> bool:
        record_fields = record.__dict__

        cached_decision = record_fields.get("_project_rate_limit_accepted")
        if cached_decision is not None:
            return bool(cached_decision)

        event_name = record_event_name(record)
        rules = self.governance.get(event_name)

        min_interval = (
            rules.min_interval_sec
            if rules is not None
            else self.default_min_interval_sec
        )

        if min_interval <= 0:
            record_fields["_project_rate_limit_accepted"] = True
            return True

        key = self._build_key(record, event_name, rules)
        now = time.monotonic()

        with self._lock:
            previous = self._last_seen.get(key)

            if previous is not None and now - previous < min_interval:
                record_fields["_project_rate_limit_accepted"] = False
                return False

            self._remember(key=key, timestamp=now)

        record_fields["_project_rate_limit_accepted"] = True
        return True

    def _build_key(
        self,
        record: logging.LogRecord,
        event_name: str,
        rules: EventRateLimitRulesSettings | None,
    ) -> tuple[object, ...]:
        field_names: Iterable[str] = (
            rules.field_names if rules is not None else ()
        )

        return (
            event_name,
            *(record.__dict__.get(field_name) for field_name in field_names),
        )

    def _remember(
        self,
        *,
        key: tuple[object, ...],
        timestamp: float,
    ) -> None:
        self._last_seen[key] = timestamp
        self._last_seen.move_to_end(key)

        if len(self._last_seen) > self.max_entries:
            self._prune(timestamp)

    def _prune(self, now: float) -> None:
        max_interval = max(
            (
                self.default_min_interval_sec,
                *(
                    rules.min_interval_sec
                    for rules in self.governance.values()
                ),
                1.0,
            )
        )

        stale_before = now - (max_interval * 2)

        while self._last_seen:
            key, timestamp = next(iter(self._last_seen.items()))
            if timestamp >= stale_before:
                break
            self._last_seen.pop(key, None)

        while len(self._last_seen) > self.max_entries:
            self._last_seen.popitem(last=False)
