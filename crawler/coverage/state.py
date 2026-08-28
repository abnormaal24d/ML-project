"""Configured live coverage state."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from config.coverage.settings import CoverageSettings
from config.validation.coverage_settings import (
    nonnegative_int,
    normalize_kind,
)
from crawler.coverage.snapshot import CoverageSnapshot


@dataclass(slots=True)
class CoverageState:
    """Track target and collected counts for configured media kinds."""

    settings: CoverageSettings
    targets: dict[str, int] = field(default_factory=dict)
    collected: dict[str, int] = field(default_factory=dict)
    _lock: Any = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )
    _version: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_settings(cls, settings: CoverageSettings) -> CoverageState:
        return cls(
            settings=settings,
            targets=normalized_count_mapping(
                settings.targets.modality_targets
            ),
        )

    def snapshot(self) -> CoverageSnapshot:
        """Return one atomic immutable snapshot of the live coverage state."""

        with self._lock:
            return CoverageSnapshot.from_counts(
                version=self._version,
                captured_at_monotonic=monotonic(),
                media_kinds=self.settings.kinds.media_kinds,
                targets_by_kind=self.targets,
                collected_by_kind=self.collected,
                source=type(self).__name__,
            )

    def target(self, kind: str) -> int:
        with self._lock:
            return self._target_unlocked(kind)

    def count(self, kind: str) -> int:
        with self._lock:
            return self._count_unlocked(kind)

    def missing(self, kind: str) -> int:
        with self._lock:
            return self._missing_unlocked(kind)

    def target_met(self, *, kind: str) -> bool:
        with self._lock:
            target = self._target_unlocked(kind)
            return target <= 0 or self._count_unlocked(kind) >= target

    def active_missing_kinds(self) -> tuple[str, ...]:
        snapshot = self.snapshot()
        return tuple(
            kind
            for kind, missing in snapshot.missing_by_kind.items()
            if missing > 0
        )

    def dominant_missing_kind(self) -> str | None:
        missing = {
            kind: count
            for kind, count in self.snapshot().missing_by_kind.items()
            if count > 0
        }
        if not missing:
            return None
        return max(missing, key=missing.__getitem__)

    def record_collected(self, *, kind: str, count: int = 1) -> None:
        with self._lock:
            changed = self._record_collected_unlocked(
                kind=kind,
                count=count,
            )
            if changed:
                self._version += 1

    def record_collected_many(self, counts: dict[str, int]) -> None:
        with self._lock:
            changed = False
            for kind, count in counts.items():
                changed = (
                    self._record_collected_unlocked(
                        kind=kind,
                        count=count,
                    )
                    or changed
                )
            if changed:
                self._version += 1

    def adjust_for_kind_change(
        self,
        *,
        previous_kind: str | None,
        current_kind: str | None,
    ) -> None:
        """When a record for a URL changes kind, move the coverage credit."""

        prev = normalize_kind(previous_kind)
        curr = normalize_kind(current_kind)
        if not prev or not curr or prev == curr:
            return

        with self._lock:
            before = self._values_for_kinds_unlocked((prev, curr))

            if prev in self.collected:
                self.collected[prev] = max(
                    0,
                    self._count_unlocked(prev) - 1,
                )
            if curr in self._known_kinds_unlocked():
                self._record_collected_unlocked(kind=curr, count=1)

            if before != self._values_for_kinds_unlocked((prev, curr)):
                self._version += 1

    def apply_record_transition(
        self,
        *,
        previous_kind: str | None,
        current_kind: str,
        previous_eligible: bool,
        current_eligible: bool,
    ) -> None:
        """Apply one atomic logical coverage delta for a canonical record."""

        previous = normalize_kind(previous_kind)
        current = normalize_kind(current_kind)

        with self._lock:
            affected = tuple(kind for kind in (previous, current) if kind)
            before = self._values_for_kinds_unlocked(affected)

            if previous_eligible and previous:
                self.collected[previous] = max(
                    0,
                    self._count_unlocked(previous) - 1,
                )
            if current_eligible and current in self._known_kinds_unlocked():
                self._record_collected_unlocked(kind=current, count=1)

            if before != self._values_for_kinds_unlocked(affected):
                self._version += 1

    def _target_unlocked(self, kind: str) -> int:
        return nonnegative_int(self.targets.get(normalize_kind(kind), 0))

    def _count_unlocked(self, kind: str) -> int:
        return nonnegative_int(self.collected.get(normalize_kind(kind), 0))

    def _missing_unlocked(self, kind: str) -> int:
        return max(
            0,
            self._target_unlocked(kind) - self._count_unlocked(kind),
        )

    def _record_collected_unlocked(
        self,
        *,
        kind: str,
        count: int,
    ) -> bool:
        normalized = normalize_kind(kind)
        if normalized not in self._known_kinds_unlocked():
            return False
        if count <= 0:
            return False

        previous = self._count_unlocked(normalized)
        current = previous + int(count)
        self.collected[normalized] = current
        return current != previous

    def _known_kinds_unlocked(self) -> set[str]:
        return {
            normalize_kind(kind)
            for kind in self.settings.kinds.media_kinds
            if normalize_kind(kind)
        }

    def _values_for_kinds_unlocked(
        self,
        kinds: tuple[str, ...],
    ) -> dict[str, int]:
        return {
            kind: self._count_unlocked(kind) for kind in dict.fromkeys(kinds)
        }


def normalized_count_mapping(raw: dict[str, int]) -> dict[str, int]:
    return {
        normalize_kind(kind): nonnegative_int(value)
        for kind, value in raw.items()
        if normalize_kind(kind)
    }
