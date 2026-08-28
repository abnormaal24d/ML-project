"""Standalone curated-snapshot entrypoint (outer bootstrap boundary).

This is the only place where settings loading, logger construction and
``asyncio.run`` meet the curated snapshot runtime.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from logger.factory import ProjectLoggerFactory
from orchestration.composition.curated_snapshot import (
    build_curated_snapshot_runtime,
)
from orchestration.settings_loader import load
from shared.runtime_primitives import Clock, IdGenerator

if TYPE_CHECKING:
    from orchestration.workflow.curated_snapshot_runtime import (
        CuratedSnapshotRuntimeResult,
    )

_LOGGER_NAME = "orchestration.bootstrap.curated_snapshot_runner"


def build_curated_snapshot_from_config(
    *,
    project_root: str | Path | None,
    config_root: str | Path | None = None,
    environment: str | None,
    snapshot_id: str | None,
    context: object | None = None,
    raw_run_selection_mode: str | None = None,
    selected_run_ids: tuple[str, ...] | None = None,
    clock: Clock,
    id_generator: IdGenerator,
) -> CuratedSnapshotRuntimeResult:
    """CLI/sync entry: asyncio.run only at this outer boundary."""

    settings = load(
        profile=environment,
        project_root=project_root,
        config_root=config_root,
        environment=environment,
    )
    logger_factory = ProjectLoggerFactory(
        root_name=settings.logging.root_name,
        base_context={
            **settings.logging.base_log_fields,
            "stage": "curated_dataset",
            "environment": settings.application.environment,
            "external_context": context is not None,
        },
    )
    logger = logger_factory.get_logger(_LOGGER_NAME)

    runtime = build_curated_snapshot_runtime(
        settings=settings,
        logger=logger,
        clock=clock,
        id_generator=id_generator,
    )
    return asyncio.run(
        runtime.build(
            snapshot_id=snapshot_id,
            raw_run_selection_mode=raw_run_selection_mode,
            selected_run_ids=selected_run_ids,
        )
    )
