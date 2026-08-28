"""Canonical lifecycle states shared by workflow artifacts."""

from __future__ import annotations

from enum import StrEnum


class WorkflowLifecycleStatus(StrEnum):
    """Single lifecycle vocabulary for every workflow generation."""

    CREATED = "created"
    RUNNING = "running"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    RESUMABLE = "resumable"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        """Return whether no more work may run in this generation."""

        return self in {
            WorkflowLifecycleStatus.COMPLETED,
            WorkflowLifecycleStatus.FAILED,
            WorkflowLifecycleStatus.CANCELLED,
        }

    @property
    def recoverable(self) -> bool:
        """Return whether reconciliation may continue this generation."""

        return self in {
            WorkflowLifecycleStatus.RUNNING,
            WorkflowLifecycleStatus.RECOVERING,
            WorkflowLifecycleStatus.INCOMPLETE,
            WorkflowLifecycleStatus.RESUMABLE,
        }

    @classmethod
    def parse(
        cls, value: object, *, field: str = "status"
    ) -> "WorkflowLifecycleStatus":
        """Parse an exact lifecycle value and reject missing values."""

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty lifecycle string")
        try:
            return cls(value.strip())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"invalid {field} {value!r}; expected one of {allowed}"
            ) from exc
