"""Select page-discovered crawl tasks through one cohesive pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from crawler.classification.media_kind import MediaKind
from crawler.discovery.media_page_classifier import classify_candidate_page
from crawler.discovery.processing.discovered_url_normalization import (
    dedupe_url_key,
)
from crawler.discovery.processing.page_discovery_limits import (
    HostKindKey,
    limit_ranked_tasks,
)
from crawler.discovery.processing.page_discovery_ranking import (
    rank_page_discovery_tasks,
)
from crawler.discovery.task_identity import discovered_task_identity
from crawler.metrics.media_discovery_metrics import MediaDiscoveryMetrics

if TYPE_CHECKING:
    from config.collection.processors import PageDiscoveryRankingSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.url_filter.url_admission_filter import (
        UrlAdmissionFilter,
    )
    from crawler.scheduling.progress.scheduler_snapshot_reader import (
        DiscoveryCapacitySnapshot,
    )


@dataclass(frozen=True, slots=True)
class PageDiscoverySelectionResult:
    """Result of filtering, ranking, limiting, and annotating discovered tasks."""

    discovered_count: int
    tasks: tuple[CrawlTask, ...]
    duplicate_count: int
    filtered_count: int
    truncated_count: int
    metrics: dict[str, dict[str, int] | dict[str, float]] | None = None
    filtered_tasks: tuple[CrawlTask, ...] = ()
    capacity_skipped_count: int = 0
    scope_blocked_count: int = 0
    scope_blocked_tasks: tuple[CrawlTask, ...] = ()


@dataclass(frozen=True, slots=True)
class PageDiscoverySelectionRequest:
    """Data required by the page-discovery selection pipeline."""

    task_stream: Iterable[CrawlTask]
    max_total: int
    max_pages: int
    max_embedded_assets: int
    max_non_page_media: int
    ranking: PageDiscoveryRankingSettings
    active_focus_kinds: tuple[str, ...]
    focus_asset_boost: float
    host_normalizer: HostNormalizer
    kind_quotas: Mapping[str, int] | None = None
    coverage_missing_by_kind: Mapping[str, int] | None = None
    url_filter: UrlAdmissionFilter | None = None
    url_normalizer: UrlNormalizer | None = None
    already_seen: Callable[[str], bool] | None = None
    scheduler_capacity: DiscoveryCapacitySnapshot | None = None
    scope_eligibility: Mapping[str, bool] | None = None

    def validate(self) -> None:
        if self.task_stream is None:
            raise ValueError("task_stream must not be None")
        if self.focus_asset_boost < 0:
            raise ValueError("focus_asset_boost must be >= 0")
        for name, value in (
            ("max_total", self.max_total),
            ("max_pages", self.max_pages),
            ("max_embedded_assets", self.max_embedded_assets),
            ("max_non_page_media", self.max_non_page_media),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        for field_name, values in (
            ("kind_quotas", self.kind_quotas),
            ("coverage_missing_by_kind", self.coverage_missing_by_kind),
        ):
            for kind, value in (values or {}).items():
                if value < 0:
                    raise ValueError(f"{field_name}[{kind!r}] must be >= 0")


def select_page_discovery_tasks(
    *, request: PageDiscoverySelectionRequest
) -> PageDiscoverySelectionResult:
    """Filter, deduplicate, rank, limit, and annotate discovered tasks."""

    request.validate()
    eligible, metrics, filtered_tasks = _collect_eligible_tasks(
        task_stream=request.task_stream,
        url_filter=request.url_filter,
        url_normalizer=request.url_normalizer,
        already_seen=request.already_seen,
        host_normalizer=request.host_normalizer,
    )
    eligible, scope_blocked_tasks = _filter_scope_blocked(
        tasks=eligible,
        scope_eligibility=request.scope_eligibility,
        metrics=metrics,
    )
    ranked = rank_page_discovery_tasks(
        tasks=eligible,
        ranking=request.ranking,
        active_focus_kinds=request.active_focus_kinds,
        focus_asset_boost=request.focus_asset_boost,
    )
    selected, capacity_skipped, truncated_count = limit_ranked_tasks(
        ranked_tasks=ranked,
        max_total=request.max_total,
        max_pages=request.max_pages,
        max_embedded_assets=request.max_embedded_assets,
        max_non_page_media=request.max_non_page_media,
        kind_quotas=request.kind_quotas,
        coverage_missing_by_kind=request.coverage_missing_by_kind,
        focus_kinds=request.active_focus_kinds,
        remaining_capacity=_scheduler_capacity_map(request.scheduler_capacity),
        host_of=_candidate_host_of(request.host_normalizer),
    )
    selected_ids = {id(task) for task in selected}
    capacity_skipped_ids = {id(task) for task in capacity_skipped}
    annotated: list[CrawlTask] = []
    for task in selected:
        reason = selection_reason(
            task=task,
            active_focus_kinds=request.active_focus_kinds,
        )
        media_identity = discovery_dedupe_key(task=task)
        metrics.record_selected(task=task, reason=reason)
        annotated.append(
            annotate_selected_task(
                task=task,
                reason=reason,
                media_identity=media_identity,
            )
        )

    metrics.record_capacity_skipped_many(tasks=capacity_skipped)
    metrics.record_truncated_many(
        tasks=(
            task
            for task in ranked
            if id(task) not in selected_ids
            and id(task) not in capacity_skipped_ids
        )
    )
    metrics.mark_focus_targets(focus_kinds=request.active_focus_kinds)
    payload = metrics.as_payload()
    return PageDiscoverySelectionResult(
        tasks=tuple(annotated),
        discovered_count=_metric_total(payload, "discovered_by_kind"),
        duplicate_count=_metric_total(payload, "duplicate_by_kind"),
        filtered_count=_metric_total(payload, "filtered_by_kind"),
        truncated_count=truncated_count,
        metrics=payload,
        filtered_tasks=tuple(filtered_tasks),
        capacity_skipped_count=len(capacity_skipped),
        scope_blocked_count=len(scope_blocked_tasks),
        scope_blocked_tasks=tuple(scope_blocked_tasks),
    )


def _scheduler_capacity_map(
    capacity: DiscoveryCapacitySnapshot | None,
) -> Mapping[HostKindKey, int] | None:
    """Flatten the scheduler-owned capacity snapshot into a consumable map.

    Returns:
        None: scheduler capacity is unknown/unavailable (snapshot missing or malformed).
        Empty mapping: scheduler explicitly reports no capacity constraints.
        Non-empty mapping: per-(host, kind) remaining capacity; zero means
            scheduler explicitly reports no remaining capacity for that pair.

    The distinction between None and zero is essential: None means the scheduler
    did not provide a snapshot (treat as unlimited), while zero means the
    scheduler explicitly reports exhaustion for that (host, kind) pair.
    """

    if capacity is None:
        return None

    by_host_kind = getattr(capacity, "by_host_kind", None)
    if not isinstance(by_host_kind, Mapping):
        return None

    return {key: max(0, int(value)) for key, value in by_host_kind.items()}


def _candidate_host_of(
    host_normalizer: HostNormalizer,
) -> Callable[[CrawlTask], str | None]:
    """Return the canonical host for a candidate, mirroring the scheduler."""

    def host_of(task: CrawlTask) -> str | None:
        try:
            hostname = urlparse(task.url).hostname
        except ValueError:
            return None
        return host_normalizer.normalize(hostname)

    return host_of


def annotate_selected_task(
    *, task: CrawlTask, reason: str, media_identity: str
) -> CrawlTask:
    """Attach deterministic selection metadata to a selected task."""

    context_payload = (
        task.context.to_dict() if task.context is not None else {}
    )
    context_payload["selection_reason"] = reason
    context_payload["media_identity"] = media_identity
    if task.parent_url:
        context_payload.setdefault("source_page_url", task.parent_url)
    return task.clone(context=context_payload)


def selection_reason(
    *, task: CrawlTask, active_focus_kinds: tuple[str, ...]
) -> str:
    """Return the stable reason used for metrics and admission identity."""

    focus_kinds = _normalize_focus_kinds(active_focus_kinds)
    if task.kind in focus_kinds:
        return "coverage_recovery"
    if task.kind is MediaKind.PAGE:
        classification = classify_candidate_page(
            task.url,
            context=task.context,
            candidate_kinds=tuple(kind.value for kind in focus_kinds),
        )
        if classification.kind is not None:
            return (
                f"{classification.kind}_candidate_page:{classification.reason}"
            )
        if focus_kinds:
            return "fallback_page"
    return str(task.source_type or "selected").strip().lower() or "selected"


def _normalize_focus_kinds(
    kinds: Iterable[str | MediaKind],
) -> tuple[MediaKind, ...]:
    normalized: list[MediaKind] = []
    supported = {
        MediaKind.DOCUMENT,
        MediaKind.AUDIO,
        MediaKind.VIDEO,
        MediaKind.IMAGE,
    }
    for kind in kinds:
        try:
            parsed_kind = MediaKind.parse(kind)
        except (TypeError, ValueError):
            continue
        if parsed_kind in supported and parsed_kind not in normalized:
            normalized.append(parsed_kind)
    return tuple(normalized)


def discovery_dedupe_key(*, task: CrawlTask) -> str:
    """Return the canonical identity used for discovery deduplication.

    Tasks are canonicalized once at the start of selection, so identity is
    based on the canonical ``task.url`` plus kind/source semantics. Discovery
    owns no second URL-normalization strategy here.
    """

    return discovered_task_identity(task=task, normalized_url=task.url)


def _collect_eligible_tasks(
    *,
    task_stream: Iterable[CrawlTask],
    url_filter: UrlAdmissionFilter | None,
    url_normalizer: UrlNormalizer | None,
    already_seen: Callable[[str], bool] | None,
    host_normalizer: HostNormalizer,
) -> tuple[list[CrawlTask], MediaDiscoveryMetrics, list[CrawlTask]]:
    seen_identities: set[str] = set()
    eligible: list[CrawlTask] = []
    filtered: list[CrawlTask] = []
    metrics = MediaDiscoveryMetrics(host_normalizer=host_normalizer)

    for task in task_stream:
        metrics.record_discovered(task=task)
        canonical_task = _canonicalized_task(
            task=task,
            url_normalizer=url_normalizer,
        )
        if canonical_task is None:
            metrics.record_filtered(
                task=task,
                reason="url_normalizer:invalid_url",
            )
            filtered.append(task)
            continue
        task = canonical_task
        identity = discovery_dedupe_key(task=task)
        if identity in seen_identities:
            metrics.record_duplicate(
                task=task,
                reason="duplicate_before_selection:same_page_duplicate",
            )
            continue
        seen_identities.add(identity)
        if already_seen is not None and bool(already_seen(identity)):
            metrics.record_duplicate(
                task=task,
                reason="duplicate_before_selection:runtime_duplicate",
            )
            continue
        decision = (
            None if url_filter is None else url_filter.evaluate_task(task)
        )
        if decision is not None and not decision.allowed:
            metrics.record_filtered(
                task=task,
                reason=f"url_filter:{decision.reason or 'blocked'}",
            )
            filtered.append(task)
            continue
        eligible.append(task)
    return eligible, metrics, filtered


def _canonicalized_task(
    *,
    task: CrawlTask,
    url_normalizer: UrlNormalizer | None,
) -> CrawlTask | None:
    """Return the task with one canonical URL, or None when normalization fails.

    The canonical URL becomes the single URL contract for URL filtering,
    deduplication, ranking, scope preflight, and scheduler admission.
    """

    raw_url = str(task.url or "").strip()
    if not raw_url:
        return None
    normalized_url = (
        url_normalizer.normalize(raw_url)
        if url_normalizer is not None
        else dedupe_url_key(raw_url)
    )
    if not normalized_url:
        return None
    if normalized_url == task.url:
        return task
    return task.clone(url=normalized_url)


def _filter_scope_blocked(
    *,
    tasks: list[CrawlTask],
    scope_eligibility: Mapping[str, bool] | None,
    metrics: MediaDiscoveryMetrics,
) -> tuple[list[CrawlTask], list[CrawlTask]]:
    """Split candidates by the scheduler-owned crawl-scope preflight.

    Scope eligibility is keyed by full discovery identity (url|kind|source).
    When no preflight is available, every candidate passes; final scheduler
    admission remains the authority either way.
    """

    from crawler.discovery.task_identity import discovered_task_identity

    if not scope_eligibility:
        return tasks, []
    allowed: list[CrawlTask] = []
    scope_blocked: list[CrawlTask] = []
    for task in tasks:
        identity = discovered_task_identity(task=task, normalized_url=task.url)
        if scope_eligibility.get(identity, True):
            allowed.append(task)
            continue
        metrics.record_scope_blocked(
            task=task,
            reason="crawl_scope_blocked",
        )
        scope_blocked.append(task)
    return allowed, scope_blocked


def _metric_total(payload: Mapping[str, object], key: str) -> int:
    values = payload.get(key, {})
    if not isinstance(values, Mapping):
        return 0
    return sum(int(value) for value in values.values())
