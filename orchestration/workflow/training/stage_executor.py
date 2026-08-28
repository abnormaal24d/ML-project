"""Training stage execution with unified lifecycle handling."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from logger.project_logger import ProjectLogger
from training.runtime.job_status.models import TrainingOperationStage
from training.runtime.job_status.payloads import TrainingResultPayload
from training.runtime.job_status.persistence import LifecycleIdentity
from training.runtime.job_status.store import TrainingJobStatusStore

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class StageResult(Generic[_T]):
    """Result of a stage execution."""

    result: _T | None = None
    error: BaseException | None = None


class TrainingStageExecutor:
    """Execute training stages with unified start/succeed/failed/cancelled handling.

    Eliminates boilerplate repetition across training workflow stages.
    Works for both attempt-level and campaign-level stages.
    """

    def __init__(
        self,
        *,
        status_store: TrainingJobStatusStore,
        logger: ProjectLogger,
    ) -> None:
        self._status_store = status_store
        self._logger = logger

    async def execute(
        self,
        *,
        identity: LifecycleIdentity,
        training_root: str,
        dataset_manifest_hash: str,
        stage: TrainingOperationStage,
        operation: Callable[[], Awaitable[_T]],
        status_result: Callable[[_T], "TrainingResultPayload | None"]
        | None = None,
    ) -> StageResult[_T]:
        """Execute a stage with full lifecycle tracking.

        Args:
            identity: Campaign or attempt identity
            training_root: Training root directory
            dataset_manifest_hash: Dataset manifest hash
            stage: Operation stage being executed
            operation: Async operation to execute
            status_result: Optional projector to extract persistable result from operation output

        Returns:
            StageResult with result or error
        """
        self._status_store.write_stage_started(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
            stage=stage,
        )

        try:
            operation_result = await operation()
        except (asyncio.CancelledError, KeyboardInterrupt) as exc:
            self._write_cancelled(
                identity=identity,
                training_root=training_root,
                dataset_manifest_hash=dataset_manifest_hash,
                error=exc,
            )
            return StageResult(error=exc)
        except Exception as exc:
            self._write_failed(
                identity=identity,
                training_root=training_root,
                dataset_manifest_hash=dataset_manifest_hash,
                error=exc,
            )
            return StageResult(error=exc)

        persisted_result = (
            status_result(operation_result)
            if status_result is not None
            else None
        )

        self._status_store.write_stage_succeeded(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
            result=persisted_result,
        )

        return StageResult(result=operation_result)

    def _write_failed(
        self,
        *,
        identity: LifecycleIdentity,
        training_root: str,
        dataset_manifest_hash: str,
        error: BaseException,
    ) -> None:
        try:
            self._status_store.write_stage_failed(
                identity=identity,
                training_root=training_root,
                dataset_manifest_hash=dataset_manifest_hash,
                result=None,
                error=error,
            )
        except Exception as write_error:
            self._logger.error(
                "train_status_write_failed_during_error",
                primary_error=str(error),
                write_error=str(write_error),
                snapshot_id=identity.snapshot_id,
                attempt_id=getattr(identity, "attempt_id", None),
            )

    def _write_cancelled(
        self,
        *,
        identity: LifecycleIdentity,
        training_root: str,
        dataset_manifest_hash: str,
        error: BaseException,
    ) -> None:
        try:
            self._status_store.write_stage_cancelled(
                identity=identity,
                training_root=training_root,
                dataset_manifest_hash=dataset_manifest_hash,
                result=None,
                error=error,
            )
        except Exception as write_error:
            self._logger.error(
                "train_status_write_failed_during_cancel",
                primary_error=str(error),
                write_error=str(write_error),
                snapshot_id=identity.snapshot_id,
                attempt_id=getattr(identity, "attempt_id", None),
            )
