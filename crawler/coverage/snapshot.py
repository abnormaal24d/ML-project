"""Immutable coverage snapshot schemas for discovery decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, TypeVar

_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


def freeze_count_mapping(
    values: Mapping[_KeyT, _ValueT],
) -> Mapping[str, int]:
    """Return a normalized, immutable non-negative integer mapping."""

    normalized: dict[str, int] = {}
    for raw_kind, raw_count in values.items():
        kind = str(raw_kind or "").strip().lower()
        if not kind:
            continue
        normalized[kind] = _nonnegative_int(raw_count)
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class CoverageSnapshot:
    """One immutable and internally consistent coverage-state read."""

    version: int
    captured_at_monotonic: float
    missing_by_kind: Mapping[str, int] = field(hash=False)
    collected_by_kind: Mapping[str, int] = field(hash=False)
    targets_by_kind: Mapping[str, int] = field(hash=False)
    source: str = "coverage_state"

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or int(self.version) < 0:
            raise ValueError("coverage snapshot version must be non-negative")

        captured_at = float(self.captured_at_monotonic)
        if captured_at < 0.0 or not math.isfinite(captured_at):
            raise ValueError(
                "coverage snapshot monotonic timestamp must be finite "
                "and non-negative"
            )

        source = str(self.source).strip()
        if not source:
            raise ValueError("coverage snapshot source is required")

        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "captured_at_monotonic", captured_at)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "missing_by_kind",
            freeze_count_mapping(self.missing_by_kind),
        )
        object.__setattr__(
            self,
            "collected_by_kind",
            freeze_count_mapping(self.collected_by_kind),
        )
        object.__setattr__(
            self,
            "targets_by_kind",
            freeze_count_mapping(self.targets_by_kind),
        )

        for kind in set(self.targets_by_kind) | set(self.collected_by_kind):
            if kind not in self.missing_by_kind:
                continue
            expected_missing = max(
                0,
                self.targets_by_kind.get(kind, 0)
                - self.collected_by_kind.get(kind, 0),
            )
            if self.missing_by_kind[kind] != expected_missing:
                raise ValueError(
                    "coverage snapshot missing count conflicts with "
                    f"targets and collected counts for kind {kind!r}"
                )

    @classmethod
    def from_counts(
        cls,
        *,
        version: int,
        captured_at_monotonic: float,
        media_kinds: tuple[str, ...],
        targets_by_kind: Mapping[str, int],
        collected_by_kind: Mapping[str, int],
        source: str = "coverage_state",
    ) -> CoverageSnapshot:
        """Build a consistent snapshot from target and collected counts."""

        targets = freeze_count_mapping(targets_by_kind)
        collected = freeze_count_mapping(collected_by_kind)
        normalized_kinds = tuple(
            dict.fromkeys(
                kind
                for raw_kind in media_kinds
                if (kind := str(raw_kind or "").strip().lower())
            )
        )
        missing = {
            kind: max(
                0,
                targets.get(kind, 0) - collected.get(kind, 0),
            )
            for kind in normalized_kinds
        }
        return cls(
            version=version,
            captured_at_monotonic=captured_at_monotonic,
            missing_by_kind=missing,
            collected_by_kind=collected,
            targets_by_kind=targets,
            source=source,
        )


def normalize_snapshot_missing(
    *,
    snapshot: CoverageSnapshot,
    media_kinds: tuple[str, ...],
) -> Mapping[str, int]:
    """Resolve one immutable gap mapping from a single snapshot.

    Missing entries are reconstructed from targets and collected counts, so a
    partial provider payload cannot silently turn a configured target into a
    zero gap. Unknown kinds are ignored.
    """

    normalized: dict[str, int] = {}
    for raw_kind in media_kinds:
        kind = str(raw_kind or "").strip().lower()
        if not kind or kind in normalized:
            continue

        if kind in snapshot.missing_by_kind:
            normalized[kind] = _nonnegative_int(snapshot.missing_by_kind[kind])
            continue

        normalized[kind] = max(
            0,
            _nonnegative_int(snapshot.targets_by_kind.get(kind, 0))
            - _nonnegative_int(snapshot.collected_by_kind.get(kind, 0)),
        )

    return freeze_count_mapping(normalized)


class CoverageSnapshotProvider(Protocol):
    """Provide one atomic coverage snapshot per caller request."""

    def snapshot(self) -> CoverageSnapshot:
        """Return one immutable point-in-time coverage snapshot."""
        ...


class CoverageUnavailableError(RuntimeError):
    """Raised when a required live coverage snapshot cannot be obtained."""

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        focus_enabled: bool,
        tracker_type: str,
        page_url: str | None = None,
        source_name: str | None = None,
    ) -> None:
        self.operation = str(operation).strip() or "coverage_snapshot"
        self.focus_enabled = bool(focus_enabled)
        self.tracker_type = str(tracker_type).strip() or "unknown"
        self.page_url = _clean_optional_text(page_url)
        self.source_name = _clean_optional_text(source_name)

        context = {
            "operation": self.operation,
            "focus_enabled": self.focus_enabled,
            "tracker_type": self.tracker_type,
        }
        if self.page_url is not None:
            context["page_url"] = self.page_url
        if self.source_name is not None:
            context["source_name"] = self.source_name

        rendered_context = ", ".join(
            f"{key}={value!r}" for key, value in context.items()
        )
        super().__init__(f"{message}; {rendered_context}")


def _clean_optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _nonnegative_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(0, int(str(value).strip()))
    except (TypeError, ValueError):
        return default
