"""Crawler runtime checkpoint and dead-letter reader."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from config.settings.crawler import CrawlStateStoreSettings
from crawler.discovery.task_identity import discovered_task_identity_from_parts
from crawler.runtime.state.crawl_checkpoint_store import CrawlerCheckpointStore
from crawler.runtime.state.crawl_dead_letter_reader import (
    CrawlerDeadLetterReader,
    DeadLetterTaskBatch,
)
from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecisionReason,
)
from crawler.scheduling.url_scheduler import UrlScheduler
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.runtime.state.crawl_state_writer import CrawlStateWriter


class CrawlStateReader:
    """Restore queued work from crawler checkpoint and dead-letter state."""

    def __init__(
        self,
        *,
        settings: CrawlStateStoreSettings,
        logger: ProjectLogger,
        scheduler: UrlScheduler,
        checkpoint_store: CrawlerCheckpointStore | None,
        current_seed_urls: tuple[str, ...],
        dead_letter_reader: CrawlerDeadLetterReader | None = None,
        checkpoint_writer: CrawlStateWriter | None = None,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._scheduler = scheduler
        self._checkpoint_store = checkpoint_store
        self._dead_letter_reader = dead_letter_reader
        self._checkpoint_writer = checkpoint_writer
        self._current_seed_urls = tuple(
            sorted(
                {
                    url.strip()
                    for url in current_seed_urls
                    if isinstance(url, str) and url.strip()
                }
            )
        )

    async def restore_checkpoint(self) -> int:
        if (
            self._checkpoint_store is None
            or not self._checkpoint_store.enabled
        ):
            return 0
        if not self._settings.resume_from_checkpoint:
            return 0

        checkpoint_payload = await asyncio.wait_for(
            asyncio.to_thread(
                self._checkpoint_store.load_checkpoint,
            ),
            timeout=30.0,
        )
        if not isinstance(checkpoint_payload, dict):
            return 0
        if not self._allows_restore(checkpoint_payload):
            return 0

        scheduler_payload = checkpoint_payload.get("scheduler")
        if not isinstance(scheduler_payload, dict):
            return 0

        restored = await self._scheduler.restore_state(
            payload=scheduler_payload,
            clear_existing=True,
        )
        if restored > 0:
            self._logger.info(
                "crawler_checkpoint_restored",
                tasks=restored,
            )
        return restored

    async def requeue_dead_letters(self) -> int:
        if (
            self._dead_letter_reader is None
            or not self._dead_letter_reader.enabled
        ):
            return 0
        if not self._settings.requeue_dead_letters_on_start:
            return 0

        batch = await asyncio.wait_for(
            asyncio.to_thread(
                self._dead_letter_reader.load_batch,
            ),
            timeout=30.0,
        )
        tasks = batch.tasks
        if not tasks:
            return 0

        decisions = await self._scheduler.enqueue_many(tasks)
        accepted_flags = tuple(decision.accepted for decision in decisions)
        accepted = sum(accepted_flags)
        checkpoint_candidates = tuple(
            decision.accepted
            or decision.reason == ScheduleDecisionReason.DUPLICATE_URL
            for decision in decisions
        )
        acknowledged = await self._acknowledge_durable_requeues(
            batch=batch,
            candidates=checkpoint_candidates,
        )
        self._logger.info(
            "crawler_dead_letters_requeued",
            requested=len(tasks),
            accepted=accepted,
            acknowledged=acknowledged,
        )
        return accepted

    async def _acknowledge_durable_requeues(
        self,
        *,
        batch: DeadLetterTaskBatch,
        candidates: tuple[bool, ...],
    ) -> int:
        if not self._settings.clear_dead_letters_on_requeue:
            return 0
        if not any(candidates):
            return 0

        checkpoint_writer = self._checkpoint_writer
        if checkpoint_writer is None or not checkpoint_writer.enabled:
            self._logger.warning(
                "crawler_dead_letter_acknowledge_skipped",
                reason="checkpoint_writer_unavailable",
                candidates=sum(candidates),
            )
            return 0

        checkpoint_written = await checkpoint_writer.write_checkpoint(
            final=False,
            max_queued_tasks=-1,
        )
        if not checkpoint_written:
            self._logger.warning(
                "crawler_dead_letter_acknowledge_skipped",
                reason="checkpoint_write_failed",
                candidates=sum(candidates),
            )
            return 0

        dead_letter_reader = self._dead_letter_reader
        checkpoint_store = self._checkpoint_store
        if dead_letter_reader is None or checkpoint_store is None:
            return 0
        scheduler_payload = await asyncio.wait_for(
            asyncio.to_thread(
                checkpoint_store.load_scheduler_checkpoint_payload,
            ),
            timeout=30.0,
        )
        checkpointed_identities = self._checkpoint_task_identities(
            scheduler_payload
        )
        durable = tuple(
            candidate and self._task_identity(task) in checkpointed_identities
            for task, candidate in zip(
                batch.tasks,
                candidates,
                strict=True,
            )
        )
        if not any(durable):
            return 0
        return await asyncio.wait_for(
            asyncio.to_thread(
                dead_letter_reader.acknowledge,
                batch=batch,
                accepted=durable,
            ),
            timeout=30.0,
        )

    @staticmethod
    def _checkpoint_task_identities(
        scheduler_payload: dict[str, object] | None,
    ) -> set[str]:
        if scheduler_payload is None:
            return set()

        identities: set[str] = set()
        for key in (
            "queued_tasks",
            "delayed_tasks",
            "requeued_inflight_tasks",
            "dispatching_tasks",
        ):
            entries = scheduler_payload.get(key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                url = entry.get("url")
                if not isinstance(url, str) or not url.strip():
                    continue
                identities.add(
                    discovered_task_identity_from_parts(
                        url=url,
                        kind=str(entry.get("kind") or ""),
                        source_type=str(entry.get("source_type") or ""),
                    )
                )
        return identities

    @staticmethod
    def _task_identity(task: CrawlTask) -> str:
        return discovered_task_identity_from_parts(
            url=task.url,
            kind=task.kind,
            source_type=task.source_type,
        )

    def _allows_restore(self, checkpoint_payload: dict[str, object]) -> bool:
        if not self._settings.resume_requires_seed_match:
            return True

        current_seed_urls = self._current_seed_urls
        if not current_seed_urls:
            return True

        run_context = checkpoint_payload.get("run_context")
        if not isinstance(run_context, dict):
            self._logger.warning(
                "crawler_checkpoint_restore_skipped",
                reason="missing_run_context",
                current_seed_count=len(current_seed_urls),
            )
            return False

        stored_seed_urls = run_context.get("seed_urls")
        if not isinstance(stored_seed_urls, list) or not all(
            isinstance(url, str) for url in stored_seed_urls
        ):
            self._logger.warning(
                "crawler_checkpoint_restore_skipped",
                reason="missing_seed_urls",
                current_seed_count=len(current_seed_urls),
            )
            return False

        normalized_stored_seed_urls = tuple(
            sorted(
                {
                    url.strip()
                    for url in stored_seed_urls
                    if isinstance(url, str) and url.strip()
                }
            )
        )
        if normalized_stored_seed_urls != current_seed_urls:
            self._logger.warning(
                "crawler_checkpoint_restore_skipped",
                reason="seed_mismatch",
                current_seed_count=len(current_seed_urls),
                checkpoint_seed_count=len(normalized_stored_seed_urls),
            )
            return False

        return True
