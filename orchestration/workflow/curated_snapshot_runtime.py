"""Workflow runtime for building curated dataset snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.curation.snapshots.dataset_assembly import (
        curated_dataset_assembler,
    )


class CuratedSnapshotAssemblerFactory(Protocol):
    """Builds one curated assembler for the requested dynamic selection."""

    def __call__(
        self,
        *,
        raw_run_selection_mode: str | None = None,
        selected_run_ids: tuple[str, ...] | None = None,
    ) -> "curated_dataset_assembler.CuratedDatasetAssembler": ...


@dataclass(frozen=True)
class CuratedSnapshotRuntimeResult:
    snapshot_id: str
    snapshot_directory: Path


class CuratedSnapshotRuntime:
    """Build curated snapshots through an injected assembler factory.

    Holds no settings and no construction policy; static dependencies are
    bound by composition, selection stays per call.
    """

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        assembler_factory: CuratedSnapshotAssemblerFactory,
        profile_log_fields: Mapping[str, object],
    ) -> None:
        self._logger = logger
        self._assembler_factory = assembler_factory
        self._profile_log_fields = dict(profile_log_fields)

    async def build(
        self,
        *,
        snapshot_id: str | None = None,
        raw_run_selection_mode: str | None = None,
        selected_run_ids: tuple[str, ...] | None = None,
    ) -> CuratedSnapshotRuntimeResult:
        """Async build: awaits multimodal preprocessing inside curation."""

        self._logger.info(
            "multimodal_profile_configured",
            **self._profile_log_fields,
        )
        assembler = self._assembler_factory(
            raw_run_selection_mode=raw_run_selection_mode,
            selected_run_ids=selected_run_ids,
        )
        result = await assembler.build(snapshot_id=snapshot_id)
        self._logger.info(
            "curated_snapshot_ready",
            snapshot_id=result.snapshot_id,
            directory=result.snapshot_directory.as_posix(),
        )
        return CuratedSnapshotRuntimeResult(
            snapshot_id=result.snapshot_id,
            snapshot_directory=result.snapshot_directory,
        )
