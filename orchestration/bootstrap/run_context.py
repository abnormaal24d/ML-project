"""Runtime identity context for crawler workflow executions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable identity for one workflow-stage execution."""

    workflow_id: str
    generation_id: str
    root_run_id: str
    run_id: str
    crawl_session_id: str
    stage: str
    parent_run_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the complete runtime identity."""

        _validate_identifier(
            "workflow_id",
            self.workflow_id,
            r"wf_[0-9a-f]{16,64}",
        )
        _validate_identifier(
            "generation_id",
            self.generation_id,
            r"gen_[0-9a-f]{16,64}",
        )
        _validate_identifier(
            "root_run_id",
            self.root_run_id,
            r"run_[a-z0-9_]{8,128}",
        )
        _validate_identifier(
            "run_id",
            self.run_id,
            r"run_[a-z0-9_]{8,128}",
        )
        _validate_identifier(
            "crawl_session_id",
            self.crawl_session_id,
            r"crawl_[0-9a-f]{16,64}",
        )
        _validate_identifier(
            "stage",
            self.stage,
            r"[a-z0-9_]{1,32}",
        )

        if self.parent_run_id is not None:
            _validate_identifier(
                "parent_run_id",
                self.parent_run_id,
                r"run_[a-z0-9_]{8,128}",
            )


def create_run_context(
    *,
    stage: str,
    parent: RunContext | None = None,
) -> RunContext:
    """Create a root identity or a child stage identity."""

    normalized_stage = _normalize_stage(stage)
    execution_token = uuid4().hex[:12]

    if parent is not None:
        return RunContext(
            workflow_id=parent.workflow_id,
            generation_id=parent.generation_id,
            root_run_id=parent.root_run_id,
            run_id=(
                f"run_{normalized_stage}_"
                f"{parent.workflow_id[3:]}_{execution_token}"
            ),
            crawl_session_id=parent.crawl_session_id,
            stage=normalized_stage,
            parent_run_id=parent.run_id,
        )

    workflow_token = uuid4().hex[:24]
    run_id = f"run_{normalized_stage}_{workflow_token}_{execution_token}"

    return RunContext(
        workflow_id=f"wf_{workflow_token}",
        generation_id=f"gen_{uuid4().hex}",
        root_run_id=run_id,
        run_id=run_id,
        crawl_session_id=f"crawl_{workflow_token}",
        stage=normalized_stage,
    )


def resume_run_context(
    *,
    stage: str,
    workflow_id: str,
    generation_id: str,
) -> RunContext:
    """Create a process identity in an existing workflow generation."""

    workflow_match = _validate_identifier(
        "workflow_id",
        workflow_id,
        r"wf_([0-9a-f]{16,64})",
    )
    _validate_identifier(
        "generation_id",
        generation_id,
        r"gen_[0-9a-f]{16,64}",
    )

    normalized_stage = _normalize_stage(stage)
    workflow_token = workflow_match.group(1)
    run_id = f"run_{normalized_stage}_{workflow_token}_{uuid4().hex[:12]}"

    return RunContext(
        workflow_id=workflow_id,
        generation_id=generation_id,
        root_run_id=run_id,
        run_id=run_id,
        crawl_session_id=f"crawl_{workflow_token}",
        stage=normalized_stage,
    )


def _normalize_stage(stage: str) -> str:
    """Normalize a stage name for runtime identifiers."""

    if not isinstance(stage, str):
        raise TypeError("stage must be a string")

    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        stage.strip().lower(),
    ).strip("_")

    if not normalized:
        raise ValueError("stage must not be empty")

    if len(normalized) > 32:
        raise ValueError("normalized stage must not exceed 32 characters")

    return normalized


def _validate_identifier(
    field_name: str,
    value: object,
    pattern: str,
) -> re.Match[str]:
    """Validate one runtime identifier."""

    if (
        not isinstance(value, str)
        or (match := re.fullmatch(pattern, value)) is None
    ):
        raise ValueError(f"{field_name} has invalid format: {value!r}")

    return match
