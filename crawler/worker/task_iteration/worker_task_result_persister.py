"""Persist worker task completion and emit callbacks."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.collection.discovery import WorkerPoolSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.processing.outcomes.processor_outcome import ProcessorOutcome
    from crawler.scheduling.url_scheduler import UrlScheduler


class WorkerTaskResultPersister:
    """Complete scheduler tasks and invoke optional result callbacks."""

    def __init__(
        self,
        *,
        settings: WorkerPoolSettings,
        scheduler: UrlScheduler,
        task_result_callback: Callable[..., Any] | None = None,
    ) -> None:
        self._settings = settings
        self._scheduler = scheduler
        self._task_result_callback = task_result_callback

    async def complete_and_emit(
        self,
        *,
        task: CrawlTask,
        completion_outcome: str,
        completion_fields: dict[str, object] | None,
        outcome: ProcessorOutcome | None,
        skip_callback: bool,
    ) -> tuple[BaseException | None, BaseException | None]:
        completion_error = await self._complete_task(
            task=task,
            completion_outcome=completion_outcome,
            completion_fields=completion_fields,
        )

        callback_error: BaseException | None = None
        if not skip_callback:
            callback_error = await self._emit_result_bounded(
                task=task,
                completion_outcome=completion_outcome,
                completion_fields=completion_fields,
                outcome=outcome,
            )

        return completion_error, callback_error

    async def _complete_task(
        self,
        *,
        task: CrawlTask,
        completion_outcome: str,
        completion_fields: dict[str, object] | None,
    ) -> BaseException | None:
        try:
            async with asyncio.timeout(
                float(self._settings.completion_timeout_seconds)
            ):
                await self._scheduler.complete(
                    task,
                    outcome=completion_outcome,
                    fields=completion_fields,
                )
        except TimeoutError as exc:
            return exc
        except (RuntimeError, OSError, ValueError) as exc:
            return exc
        return None

    async def _emit_result_bounded(
        self,
        *,
        task: CrawlTask,
        completion_outcome: str,
        completion_fields: dict[str, object] | None,
        outcome: ProcessorOutcome | None,
    ) -> BaseException | None:
        try:
            async with asyncio.timeout(
                float(self._settings.callback_timeout_seconds)
            ):
                await self._emit_task_result(
                    task=task,
                    outcome=completion_outcome,
                    fields=completion_fields,
                    result=outcome,
                )
        except TimeoutError as exc:
            return exc
        except (RuntimeError, OSError, ValueError) as exc:
            return exc
        return None

    async def _emit_task_result(
        self,
        *,
        task: CrawlTask,
        outcome: str,
        fields: dict[str, object] | None,
        result: ProcessorOutcome | None,
    ) -> None:
        if self._task_result_callback is None:
            return

        callback_result = self._task_result_callback(
            task=task,
            outcome=outcome,
            fields=fields,
            result=result,
        )
        if inspect.isawaitable(callback_result):
            await callback_result
