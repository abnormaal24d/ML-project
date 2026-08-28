"""Application-wide contracts for semantic time and identity sources."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    """Provide semantic wall-clock time to application workflows."""

    def now(self) -> datetime: ...


class IdGenerator(Protocol):
    """Provide semantic identifiers to application workflows."""

    def generate(self) -> str: ...


__all__ = ["Clock", "IdGenerator"]
