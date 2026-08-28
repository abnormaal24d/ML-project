"""Route fetched media and documents through analysis lanes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from crawler.analysis.enrichment.lanes.analysis_worker_lane import (
    AnalysisJob,
    AnalysisResult,
    AnalysisWorkerLane,
)
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.fetching.results.result import FetchResult
from crawler.processing.outcomes.processor_outcome import ProcessorOutcome
from crawler.processing.processors.persisting_processor import (
    PersistingProcessor,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.analysis.enrichment.lanes.analysis_result_writer import (
        AnalysisResultWriter,
    )


class AnalysisRouter:
    """Own and route all media/document analysis lanes."""

    def __init__(
        self,
        *,
        lanes: Mapping[MediaKind, AnalysisWorkerLane],
        record_writer: AnalysisResultWriter,
        logger: ProjectLogger,
    ) -> None:
        self._lanes = dict(lanes)
        self._record_writer = record_writer
        self._logger = logger
        self._force_stopped = False

    def owns(self, *, kind: MediaKind) -> bool:
        return kind in self._lanes

    async def start(self) -> None:
        for lane in self._lanes.values():
            await lane.start()

    async def submit(
        self,
        *,
        task: CrawlTask,
        fetch_result: FetchResult,
        kind: MediaKind,
        processor: PersistingProcessor[Any, AnalysisResult],
    ) -> ProcessorOutcome:
        lane = self._lanes.get(kind)
        queue_len = lane.queue_size() if lane is not None else 0
        self._logger.debug(
            "analysis_router_submit",
            task_id=task.task_id,
            url=fetch_result.final_url,
            kind=kind,
            queue_length=queue_len,
        )
        job = AnalysisJob(
            task=task,
            fetch_result=fetch_result,
            kind=kind,
            processor=processor,
        )

        lane = self._lanes.get(kind)
        if lane is None:
            return ProcessorOutcome.dropped(
                stage="analysis_handoff",
                reason="analysis_lane_missing",
            )

        await lane.submit_job(job)
        self._logger.debug(
            "media_analysis_handoff_submitted",
            task_id=task.task_id,
            url=task.url,
            result_kind=kind,
            final_url=fetch_result.final_url,
        )
        return ProcessorOutcome.success(
            stage="analysis_handoff",
            detail="analysis_handoff",
        )

    async def drain(self) -> None:
        if self._force_stopped:
            self._logger.warning(
                "media_analysis_drain_skipped_after_forced_stop",
                lanes=sorted(self._lanes),
            )
            return

        for lane in self._lanes.values():
            await lane.drain()
        await self._record_writer.drain()

    async def stop(self, *, force: bool = False) -> None:
        if force or self._force_stopped:
            self._force_stopped = True
            self._logger.warning(
                "media_analysis_stop_forced",
                lanes=sorted(self._lanes),
            )
        else:
            await self.drain()

        for lane in self._lanes.values():
            await lane.stop()

    def snapshot(self) -> dict[str, object]:
        """Return diagnostics for drain and stop observability."""
        return {
            "lanes": [lane.snapshot() for lane in self._lanes.values()],
            "writer": self._record_writer.snapshot(),
            "force_stopped": self._force_stopped,
        }
