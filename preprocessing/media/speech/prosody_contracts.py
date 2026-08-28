"""Canonical validated prosody contracts, independent from DSP extraction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

_PROSODY_FIELD_ORDER: Final[tuple[str, ...]] = (
    "pitch_hz",
    "energy",
    "tempo",
    "pause_ratio",
)
CANONICAL_PROSODY_FIELDS: Final[frozenset[str]] = frozenset(
    _PROSODY_FIELD_ORDER
)


class ProsodyValidationError(ValueError):
    """Raised when prosody data violates the canonical schema."""


class ProsodyStatus(StrEnum):
    """Canonical outcome of a prosody analysis."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


def _optional_finite_float(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float | None:
    """Validate and normalize an optional JSON-compatible number."""

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProsodyValidationError(
            f"{field_name} must be a JSON number or null"
        )

    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ProsodyValidationError(
            f"{field_name} cannot be represented as a float"
        ) from exc

    if not math.isfinite(parsed):
        raise ProsodyValidationError(f"{field_name} must be finite")

    if minimum is not None:
        below_minimum = (
            parsed < minimum if minimum_inclusive else parsed <= minimum
        )
        if below_minimum:
            operator = ">=" if minimum_inclusive else ">"
            raise ProsodyValidationError(
                f"{field_name} must be {operator} {minimum}"
            )

    if maximum is not None and parsed > maximum:
        raise ProsodyValidationError(f"{field_name} must be <= {maximum}")

    return parsed


@dataclass(frozen=True, slots=True, kw_only=True)
class ProsodyFeatures:
    """Validated canonical prosody measurements."""

    pitch_hz: float | None = None
    energy: float | None = None
    tempo: float | None = None
    pause_ratio: float | None = None

    def __post_init__(self) -> None:
        pitch_hz = _optional_finite_float(
            self.pitch_hz,
            field_name="prosody.pitch_hz",
            minimum=0.0,
            minimum_inclusive=False,
        )
        energy = _optional_finite_float(
            self.energy,
            field_name="prosody.energy",
            minimum=0.0,
            maximum=1.0,
        )
        tempo = _optional_finite_float(
            self.tempo,
            field_name="prosody.tempo",
            minimum=0.0,
        )
        pause_ratio = _optional_finite_float(
            self.pause_ratio,
            field_name="prosody.pause_ratio",
            minimum=0.0,
            maximum=1.0,
        )

        if all(
            measurement is None
            for measurement in (
                pitch_hz,
                energy,
                tempo,
                pause_ratio,
            )
        ):
            raise ProsodyValidationError(
                "prosody must contain at least one measurement"
            )

        object.__setattr__(self, "pitch_hz", pitch_hz)
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "tempo", tempo)
        object.__setattr__(self, "pause_ratio", pause_ratio)

    @property
    def is_complete(self) -> bool:
        """Return whether every canonical measurement is available."""

        return all(
            measurement is not None
            for measurement in (
                self.pitch_hz,
                self.energy,
                self.tempo,
                self.pause_ratio,
            )
        )

    @property
    def available_fields(self) -> frozenset[str]:
        """Return the names of measurements that are present."""

        return frozenset(
            field_name
            for field_name in _PROSODY_FIELD_ORDER
            if getattr(self, field_name) is not None
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ProsodyAnalysisResult:
    """Prosody features plus analysis metadata."""

    features: ProsodyFeatures | None = None
    pitch_std_hz: float | None = None
    duration_seconds: float | None = None
    status: ProsodyStatus = ProsodyStatus.UNAVAILABLE
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.features is not None and not isinstance(
            self.features,
            ProsodyFeatures,
        ):
            raise ProsodyValidationError(
                "features must be ProsodyFeatures or null"
            )

        status = self._normalize_status(self.status)
        reasons = self._normalize_reasons(self.reasons)
        pitch_std_hz = _optional_finite_float(
            self.pitch_std_hz,
            field_name="prosody.pitch_std_hz",
            minimum=0.0,
        )
        duration_seconds = _optional_finite_float(
            self.duration_seconds,
            field_name="prosody.duration_seconds",
            minimum=0.0,
            minimum_inclusive=False,
        )

        self._validate_state(
            status=status,
            reasons=reasons,
            pitch_std_hz=pitch_std_hz,
        )

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "pitch_std_hz", pitch_std_hz)
        object.__setattr__(
            self,
            "duration_seconds",
            duration_seconds,
        )

    @staticmethod
    def _normalize_status(value: object) -> ProsodyStatus:
        if isinstance(value, ProsodyStatus):
            return value

        if not isinstance(value, str):
            raise ProsodyValidationError("prosody status must be a string")

        try:
            return ProsodyStatus(value)
        except ValueError as exc:
            allowed = ", ".join(status.value for status in ProsodyStatus)
            raise ProsodyValidationError(
                f"unsupported prosody status {value!r}; "
                f"expected one of: {allowed}"
            ) from exc

    @staticmethod
    def _normalize_reasons(value: object) -> tuple[str, ...]:
        if not isinstance(value, tuple):
            raise ProsodyValidationError(
                "prosody reasons must be a tuple of strings"
            )

        normalized: list[str] = []
        for index, reason in enumerate(value):
            if not isinstance(reason, str):
                raise ProsodyValidationError(
                    f"prosody.reasons[{index}] must be a string"
                )

            cleaned = reason.strip()
            if not cleaned:
                raise ProsodyValidationError(
                    f"prosody.reasons[{index}] must not be empty"
                )
            normalized.append(cleaned)

        if len(normalized) != len(set(normalized)):
            raise ProsodyValidationError(
                "prosody reasons must not contain duplicates"
            )

        return tuple(normalized)

    def _validate_state(
        self,
        *,
        status: ProsodyStatus,
        reasons: tuple[str, ...],
        pitch_std_hz: float | None,
    ) -> None:
        features = self.features

        if status is ProsodyStatus.AVAILABLE:
            if features is None:
                raise ProsodyValidationError(
                    "available prosody must contain features"
                )
            if not features.is_complete:
                raise ProsodyValidationError(
                    "available prosody must contain every canonical "
                    "measurement"
                )
            if reasons:
                raise ProsodyValidationError(
                    "available prosody must not contain failure reasons"
                )

        elif status is ProsodyStatus.PARTIAL:
            if features is None:
                raise ProsodyValidationError(
                    "partial prosody must contain features"
                )
            if features.is_complete:
                raise ProsodyValidationError(
                    "complete features must use status 'available'"
                )
            if not reasons:
                raise ProsodyValidationError(
                    "partial prosody must explain which data is missing"
                )

        elif status in {
            ProsodyStatus.UNAVAILABLE,
            ProsodyStatus.FAILED,
        }:
            if features is not None:
                raise ProsodyValidationError(
                    f"{status.value} prosody must not contain features"
                )
            if status is ProsodyStatus.FAILED and not reasons:
                raise ProsodyValidationError(
                    "failed prosody must contain at least one reason"
                )

        if pitch_std_hz is not None:
            if features is None or features.pitch_hz is None:
                raise ProsodyValidationError("pitch_std_hz requires pitch_hz")


def parse_prosody(value: object) -> ProsodyFeatures | None:
    """Parse canonical prosody data from an untrusted mapping."""

    if value is None:
        return None

    if not isinstance(value, Mapping):
        raise ProsodyValidationError("prosody must be an object or null")

    raw_keys = tuple(value.keys())
    invalid_keys = tuple(key for key in raw_keys if not isinstance(key, str))
    if invalid_keys:
        rendered = ", ".join(repr(key) for key in invalid_keys)
        raise ProsodyValidationError(
            f"prosody field names must be strings; invalid keys: {rendered}"
        )

    unknown_fields = sorted(frozenset(raw_keys) - CANONICAL_PROSODY_FIELDS)
    if unknown_fields:
        joined = ", ".join(unknown_fields)
        raise ProsodyValidationError(
            f"prosody contains unsupported fields: {joined}"
        )

    return ProsodyFeatures(
        pitch_hz=_optional_finite_float(
            value.get("pitch_hz"),
            field_name="prosody.pitch_hz",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        energy=_optional_finite_float(
            value.get("energy"),
            field_name="prosody.energy",
            minimum=0.0,
            maximum=1.0,
        ),
        tempo=_optional_finite_float(
            value.get("tempo"),
            field_name="prosody.tempo",
            minimum=0.0,
        ),
        pause_ratio=_optional_finite_float(
            value.get("pause_ratio"),
            field_name="prosody.pause_ratio",
            minimum=0.0,
            maximum=1.0,
        ),
    )
