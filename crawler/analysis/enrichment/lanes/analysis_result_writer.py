"""Serialized persistence for completed media analysis results."""

from __future__ import annotations

import asyncio

from crawler.analysis.enrichment.lanes.analysis_worker_lane import (
    AnalysisJobResult,
)
from crawler.processing.outcomes.processor_outcome import ProcessorOutcome
from logger.project_logger import ProjectLogger


class AnalysisResultWriter:
    """Persist analysis job results and clean owned fetch payload files.

    Payload cleanup runs only after persistence returns successfully so a
    failed write leaves the temporary source available for recovery.
    """

    def __init__(self, *, logger: ProjectLogger) -> None:
        self._logger = logger
        self._write_lock = asyncio.Lock()

    async def write_result(
        self, result: AnalysisJobResult
    ) -> ProcessorOutcome:
        job = result.job
        async with self._write_lock:
            if result.reason is not None:
                outcome = await job.processor.persist_analysis_failure(
                    task=job.task,
                    result=job.fetch_result,
                    reason=result.reason,
                    error_type=result.error_type,
                    error=result.error,
                )
            else:
                outcome = await job.processor.persist_analyzed_result(
                    task=job.task,
                    result=job.fetch_result,
                    analysis=result.analysis,
                )

            self._logger.debug(
                "analysis_result_written",
                task_id=job.task.task_id,
                url=job.task.url,
                result_kind=job.kind,
                final_url=job.fetch_result.final_url,
                outcome=outcome.status,
                reason=result.reason,
            )
            # Cleanup only after successful persistence (success or recorded
            # failure). A raised exception leaves the payload for recovery.
            self._cleanup_payload(result=result)
            return outcome

    async def drain(self) -> None:
        async with self._write_lock:
            return None

    def snapshot(self) -> dict[str, object]:
        return {
            "writer": type(self).__name__,
            "write_locked": self._write_lock.locked(),
        }

    def _cleanup_payload(self, *, result: AnalysisJobResult) -> None:
        fetch_result = result.job.fetch_result
        payload = fetch_result.payload

        if payload is None:
            return

        try:
            payload.cleanup()
        except OSError as exc:
            self._logger.warning(
                "analysis_payload_cleanup_failed",
                final_url=fetch_result.final_url,
                payload_path=str(getattr(payload, "temp_path", None)),
                error_type=type(exc).__name__,
                error=str(exc),
            )
