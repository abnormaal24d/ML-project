"""Mutable runtime state for the URL scheduler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SchedulerRuntimeState:
    """Mutable runtime state shared by scheduler services."""

    closed: bool = False
    next_sequence_value: int = 0

    def is_closed(self) -> bool:
        return self.closed

    def allocate_sequence(self) -> int:
        sequence = self.next_sequence_value
        self.next_sequence_value += 1
        return sequence
