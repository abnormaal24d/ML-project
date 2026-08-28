"""Contracts between workflow plans and the augmentation phase runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from config.path_resolution.project_paths import ProjectPaths
from config.settings.datasets import (
    DatasetPathSettings,
    TrainingDatasetWriterSettings,
)
from datachecker.workflow_decision import (
    WorkflowAction,
    WorkflowDecisionReason,
    WorkflowExecutionPlan,
)
from orchestration.workflow.augmentation.phase_runner import AugmentPhaseRunner
from orchestration.workflow.phase import PhaseStatus


@pytest.mark.asyncio
async def test_augment_runner_passes_injected_output_root() -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    async def run_blocking(
        func: object,
        /,
        *args: object,
        **kwargs: object,
    ) -> None:
        del args
        kwargs.pop("timeout_seconds", None)
        calls.append((func, kwargs))

    training_root = Path("training/snapshot-42")
    output_root = Path("training/augmented-snapshot-42")

    writer_settings = TrainingDatasetWriterSettings(write_jsonl=True)
    dataset_paths = DatasetPathSettings()

    logger = SimpleNamespace(exception=lambda *a, **k: None)

    runner = AugmentPhaseRunner(
        logger=logger,  # type: ignore[arg-type]
        write_augmentation_manifest=lambda: None,
        run_blocking=run_blocking,
        io_timeout_seconds=1.0,
        augmentation_workflow=SimpleNamespace(),
        clock=SimpleNamespace(),
        dataset_paths=dataset_paths,
        writer_settings=writer_settings,
        project_paths=ProjectPaths(project_root=Path("/tmp")),
        output_root=output_root,
    )

    outcome = await runner.run(
        WorkflowExecutionPlan(
            action=WorkflowAction.AUGMENT,
            reason=WorkflowDecisionReason.AUGMENTATION_OUTPUT_MISSING,
            training_root=training_root,
        )
    )

    assert calls[0] == (
        runner._build_snapshot,
        {
            "training_root": training_root,
            "output_root": output_root,
        },
    )
    assert calls[1][0] is runner._write_augmentation_manifest
    assert outcome.status is PhaseStatus.SUCCEEDED
    assert outcome.next_plan is None


@pytest.mark.asyncio
async def test_augment_runner_requires_training_root() -> None:
    async def run_blocking(
        func: object,
        /,
        *args: object,
        **kwargs: object,
    ) -> None:
        raise AssertionError("no snapshot may be built without a root")

    runner = AugmentPhaseRunner(
        logger=SimpleNamespace(),  # type: ignore[arg-type]
        write_augmentation_manifest=lambda: None,
        run_blocking=run_blocking,
        io_timeout_seconds=1.0,
        augmentation_workflow=SimpleNamespace(),
        clock=SimpleNamespace(),
        dataset_paths=DatasetPathSettings(),
        writer_settings=TrainingDatasetWriterSettings(write_jsonl=True),
        project_paths=ProjectPaths(project_root=Path("/tmp")),
        output_root=Path("out"),
    )

    with pytest.raises(ValueError, match="missing training_root"):
        await runner.run(
            WorkflowExecutionPlan(
                action=WorkflowAction.AUGMENT,
                reason=WorkflowDecisionReason.AUGMENTATION_OUTPUT_MISSING,
            )
        )
