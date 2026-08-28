"""Release status and decision contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from schemas.release import ReleaseStatus

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class ReleaseDecision:
    """Final status, unique reasons, and best-effort warnings."""

    status: ReleaseStatus
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reasons", tuple(str(reason) for reason in self.reasons)
        )
        object.__setattr__(
            self,
            "warnings",
            tuple(str(warning) for warning in self.warnings),
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status.value,
            "reasons": self.reasons,
        }
        if self.warnings:
            payload["warnings"] = self.warnings
        return payload


__all__ = ["ReleaseDecision", "ReleaseStatus"]
