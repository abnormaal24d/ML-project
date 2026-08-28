"""Rank page-discovered crawl tasks by discovery value."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from crawler.classification.media_kind import MediaKind
from crawler.discovery.media_page_classifier import (
    classify_candidate_page,
    is_multimodal_media_page_url,
    is_probably_decorative_asset_context,
    multimodal_context_signal_score,
)
from crawler.discovery.task_identity import discovered_task_identity
from crawler.numeric import coerce_finite_float

if TYPE_CHECKING:
    from config.collection.processors import PageDiscoveryRankingSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask

_MEDIA_ASSET_KINDS = frozenset(
    {
        MediaKind.IMAGE,
        MediaKind.AUDIO,
        MediaKind.VIDEO,
        MediaKind.DOCUMENT,
    }
)
_KIND_TIEBREAK = {
    MediaKind.DOCUMENT: 0,
    MediaKind.AUDIO: 1,
    MediaKind.VIDEO: 2,
    MediaKind.IMAGE: 3,
    MediaKind.FEED: 4,
    MediaKind.PAGE: 5,
}


def rank_page_discovery_tasks(
    *,
    tasks: Iterable[CrawlTask],
    ranking: PageDiscoveryRankingSettings,
    active_focus_kinds: tuple[str, ...],
    focus_asset_boost: float,
) -> list[CrawlTask]:
    """Return tasks ordered by descending discovery priority score."""

    scored_tasks = [
        (
            score_page_discovery_task(
                task=task,
                ranking=ranking,
                active_focus_kinds=active_focus_kinds,
                focus_asset_boost=focus_asset_boost,
            ),
            discovered_task_identity(task=task, normalized_url=task.url),
            index,
            task,
        )
        for index, task in enumerate(tasks)
    ]
    scored_tasks.sort(
        key=lambda item: (
            -item[0],
            _KIND_TIEBREAK.get(item[3].kind, 99),
            item[1],
            item[2],
        )
    )
    return [task for _, _, _, task in scored_tasks]


def score_page_discovery_task(
    *,
    task: CrawlTask,
    ranking: PageDiscoveryRankingSettings,
    active_focus_kinds: tuple[str, ...],
    focus_asset_boost: float,
) -> float:
    """Return the discovery priority score for one task."""

    score = ranking.kind_weights.get(
        task.kind.value,
        ranking.default_kind_weight,
    )
    context_signal_score = multimodal_context_signal_score(
        context=task.context
    )

    if task.source_type == "discovered_link":
        score += ranking.discovered_link_bonus
    elif task.source_type == "embedded_asset":
        if task.kind in {
            MediaKind.IMAGE,
            MediaKind.AUDIO,
            MediaKind.VIDEO,
        }:
            score -= ranking.embedded_media_asset_penalty
        else:
            score -= ranking.embedded_asset_penalty
    else:
        score -= ranking.non_link_source_penalty

    focus_kinds = _normalize_focus_kinds(active_focus_kinds)
    if task.kind in _MEDIA_ASSET_KINDS and task.kind in focus_kinds:
        score += focus_asset_boost

    path = urlparse(task.url).path.lower()
    if task.kind is MediaKind.PAGE:
        focus_classification = classify_candidate_page(
            task.url,
            context=task.context,
            candidate_kinds=tuple(kind.value for kind in focus_kinds),
        )
        is_media_page = is_multimodal_media_page_url(
            task.url,
            context=task.context,
        )
        if focus_classification.kind is not None:
            score += focus_asset_boost
            score += ranking.media_page_bonus
        elif is_media_page:
            score += ranking.media_page_bonus
        elif focus_kinds:
            score -= ranking.page_bonus
        if context_signal_score > 0:
            score += ranking.page_bonus * min(2, context_signal_score)
        if any(token in path for token in ranking.page_bonus_tokens):
            score += ranking.page_bonus
        if any(token in path for token in ranking.page_penalty_tokens):
            score -= ranking.page_penalty
    else:
        if context_signal_score > 0:
            score += ranking.page_bonus * min(2, context_signal_score)
        score += _asset_quality_bonus(context=task.context)
        score += _media_variant_bonus(task=task)
        if is_probably_decorative_asset_context(context=task.context):
            score -= ranking.asset_path_penalty
        if any(token in path for token in ranking.asset_penalty_tokens):
            score -= ranking.asset_path_penalty

    if "?" in task.url:
        score -= ranking.query_penalty
    return float(score)


def _normalize_focus_kinds(
    kinds: Iterable[str | MediaKind],
) -> tuple[MediaKind, ...]:
    normalized: list[MediaKind] = []
    supported = {
        MediaKind.DOCUMENT,
        MediaKind.AUDIO,
        MediaKind.VIDEO,
    }
    for kind in kinds:
        try:
            parsed_kind = MediaKind.parse(kind)
        except (TypeError, ValueError):
            continue
        if parsed_kind in supported and parsed_kind not in normalized:
            normalized.append(parsed_kind)
    return tuple(normalized)


def _asset_quality_bonus(*, context: Any) -> float:
    if context is None:
        return 0.0
    value = getattr(context, "asset_quality_score", None) or 0.0
    return (
        coerce_finite_float(
            value,
            default=0.0,
            minimum=-50.0,
            maximum=100.0,
        )
        / 50.0
    )


def _media_variant_bonus(*, task: CrawlTask) -> float:
    if task.kind is not MediaKind.VIDEO:
        return 0.0

    url = str(task.url or "").lower()
    bonus = 0.0
    if "public-output-media" in url or "/mp4/" in url:
        bonus += 0.6
    if "public-input-media" in url:
        bonus -= 0.6
    if "/captions/" in url:
        bonus -= 1.0
    return bonus
