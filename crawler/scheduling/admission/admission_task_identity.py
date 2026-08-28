"""Task identity and depth limits for scheduler admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind
from crawler.discovery.task_identity import discovered_task_identity

if TYPE_CHECKING:
    from config.collection.discovery import SchedulingSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask


def scheduler_task_identity_key(*, task: CrawlTask) -> str:
    """Return the duplicate identity from a canonical scheduler task."""

    return discovered_task_identity(task=task, normalized_url=task.url)


def scheduler_task_identity_key_for_url(*, task: CrawlTask, url: str) -> str:
    """Return an identity for an explicitly equivalent URL."""

    return discovered_task_identity(task=task, normalized_url=url)


def max_depth_for_task(
    *,
    task: CrawlTask,
    settings: SchedulingSettings,
) -> int:
    if task.kind is MediaKind.DOCUMENT:
        return settings.effective_max_document_candidate_depth()
    if task.kind is MediaKind.AUDIO:
        return settings.effective_max_audio_candidate_depth()
    if task.kind is MediaKind.VIDEO:
        return settings.effective_max_video_candidate_depth()

    if task.kind is not MediaKind.PAGE:
        return settings.max_depth

    selection_reason = ""
    if task.context is not None:
        selection_reason = (
            str(task.context.selection_reason or "").strip().lower()
        )

    if selection_reason.startswith("document_candidate_page"):
        return settings.effective_max_document_candidate_depth()
    if selection_reason.startswith("audio_candidate_page"):
        return settings.effective_max_audio_candidate_depth()
    if selection_reason.startswith("video_candidate_page"):
        return settings.effective_max_video_candidate_depth()

    return settings.max_depth
