"""Apply queue-pressure and hostility rules to scheduler admission."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind
from crawler.discovery.media_page_classifier import (
    is_multimodal_media_page_url,
)
from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecisionReason,
)
from crawler.scheduling.scheduling_value_parser import coerce_float, coerce_int

if TYPE_CHECKING:
    from config.collection.discovery import SchedulingSettings
    from crawler.scheduling.host_control.host_advice_tracker import (
        HostAdviceTracker,
    )

from crawler.crawl_tasks.crawl_task import CrawlTask


class AdmissionPressureRules:
    """Evaluate scheduler queue pressure and host hostility."""

    def __init__(
        self,
        *,
        settings: SchedulingSettings,
        host_advice_tracker: HostAdviceTracker,
    ) -> None:
        self._settings = settings
        self._host_advice_tracker = host_advice_tracker

    def queue_pressure_state(self, *, queue_size: int) -> str:
        if self.is_critical_backpressure(queue_size=queue_size):
            return "critical"
        if self.is_high_backpressure(queue_size=queue_size):
            return "high"
        return "normal"

    def is_high_backpressure(self, *, queue_size: int) -> bool:
        threshold = coerce_int(self._settings.queue_high_watermark)
        if threshold is None:
            return False
        return queue_size >= threshold

    def is_critical_backpressure(self, *, queue_size: int) -> bool:
        threshold = coerce_int(self._settings.queue_critical_watermark)
        if threshold is None:
            return False
        return queue_size >= threshold

    def rejection_reason(
        self,
        *,
        task: CrawlTask,
        queue_size: int,
    ) -> ScheduleDecisionReason | None:
        if self.is_critical_backpressure(queue_size=queue_size):
            if task.source_type != "seed":
                return ScheduleDecisionReason.SCHEDULER_BACKPRESSURE
            return None

        if self.is_high_backpressure(queue_size=queue_size):
            if self._should_reject_for_high_backpressure(task=task):
                return ScheduleDecisionReason.SCHEDULER_BACKPRESSURE

        return None

    def hostility_rejection_reason(
        self,
        *,
        task: CrawlTask,
        host: str | None,
        use_host_advice: bool = True,
    ) -> ScheduleDecisionReason | None:
        if not use_host_advice or host is None or task.source_type == "seed":
            return None

        entry = self._host_advice_tracker.get(host)
        if entry is None:
            return None

        hostility_score = coerce_float(
            entry.advice.hostility_score,
            allow_bool=True,
        )
        if hostility_score is None or not isfinite(hostility_score):
            return None

        threshold = float(self._settings.hostility_reject_threshold)
        if hostility_score < threshold or task.kind is MediaKind.PAGE:
            return None

        return ScheduleDecisionReason.HOSTILITY_BACKPRESSURE

    @staticmethod
    def _should_reject_for_high_backpressure(*, task: CrawlTask) -> bool:
        if is_coverage_recovery_target_task(task):
            return False

        if (
            task.kind
            in {
                MediaKind.IMAGE,
                MediaKind.AUDIO,
                MediaKind.DOCUMENT,
                MediaKind.VIDEO,
            }
            and task.source_type == "embedded_asset"
        ):
            return False

        if task.kind is MediaKind.PAGE:
            return (
                task.source_type != "seed"
                and not is_multimodal_media_page_url(
                    task.url,
                    context=task.context,
                )
            )

        return task.source_type != "seed"


_TARGET_COVERAGE_KINDS = frozenset(
    {
        MediaKind.DOCUMENT,
        MediaKind.IMAGE,
        MediaKind.AUDIO,
        MediaKind.VIDEO,
    }
)


def coverage_recovery_target_kind(task: CrawlTask) -> MediaKind | None:
    """Return the target kind for selected direct coverage-recovery media tasks."""

    kind = task.kind
    if kind not in _TARGET_COVERAGE_KINDS:
        return None

    context = task.context
    selection_reason = (
        str(context.selection_reason or "").strip().lower()
        if context is not None
        else ""
    )

    if selection_reason != "coverage_recovery":
        return None

    return kind


def is_coverage_recovery_target_task(task: CrawlTask) -> bool:
    """Return true when this task directly fills missing raw modality coverage."""

    return coverage_recovery_target_kind(task) is not None
