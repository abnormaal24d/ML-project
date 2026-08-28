"""Typed acceptance outcome for augmented media validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MediaValidationOutcome:
    """Validation status with diagnostic signals."""

    rejection_reason: str | None
    signals: dict[str, object]

    @property
    def accepted(self) -> bool:
        return self.rejection_reason is None
