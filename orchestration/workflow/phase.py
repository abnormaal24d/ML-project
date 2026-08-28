"""Generic phase contracts owned by the workflow layer.

These contracts describe what a workflow phase is and how its outcome flows
back into the coordinator. The bootstrap layer consumes them; the concrete
``run_blocking`` executor remains bootstrap-owned infrastructure.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from datachecker.workflow_decision import WorkflowExecutionPlan

RunBlocking = Callable[..., Awaitable[Any]]


class PhaseStatus(StrEnum):
    """Closed statuses emitted while executing one workflow phase."""

    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    RECRAWL_REQUESTED = "recrawl_requested"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PhaseOutcome:
    """Execution outcome for one workflow phase.

    ``status`` is the single source of truth: a phase only continues the
    workflow when it is SUCCEEDED or RECRAWL_REQUESTED. A recrawl request
    must carry the next plan, and no other status may carry one.
    """

    status: PhaseStatus
    next_plan: WorkflowExecutionPlan | None = None

    def __post_init__(self) -> None:
        if (self.status is PhaseStatus.RECRAWL_REQUESTED) != (
            self.next_plan is not None
        ):
            raise ValueError(
                "RECRAWL_REQUESTED requires a next_plan; any other status "
                "must not carry one."
            )


PhaseRunner = Callable[[WorkflowExecutionPlan], Awaitable[PhaseOutcome]]
