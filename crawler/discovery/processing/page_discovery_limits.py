"""Apply caps, quotas, and fairness to ranked page-discovery tasks."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask

_MEDIA_KIND_PRIORITY = (
    MediaKind.DOCUMENT,
    MediaKind.AUDIO,
    MediaKind.VIDEO,
    MediaKind.IMAGE,
    MediaKind.FEED,
)
_MEDIA_KINDS = frozenset(_MEDIA_KIND_PRIORITY)

HostKindKey = tuple[str | None, MediaKind]


def limit_ranked_tasks(
    *,
    ranked_tasks: list[CrawlTask],
    max_total: int,
    max_pages: int,
    max_embedded_assets: int,
    max_non_page_media: int,
    kind_quotas: Mapping[str, int] | None = None,
    coverage_missing_by_kind: Mapping[str, int] | None = None,
    focus_kinds: Iterable[str] = (),
    remaining_capacity: Mapping[HostKindKey, int] | None = None,
    host_of: Callable[[CrawlTask], str | None] | None = None,
) -> tuple[list[CrawlTask], list[CrawlTask], int]:
    """Limit ranked tasks using host quotas, capacity, and fairness.

    ``remaining_capacity`` is a scheduler-owned view of frontier slots left
    per ``(host, kind)``. A candidate on a full host is capacity-skipped: it
    consumes no selection slot and the next ranked candidate is considered.
    Splits: ranking, grouping by host, round-robin selection in helpers.
    Returns ``(selected, capacity_skipped, truncated_count)`` where
    ``capacity_skipped`` is excluded from ``truncated_count``.
    """
    normalized_quotas = _normalize_kind_quotas(kind_quotas)
    normalized_missing = _normalize_kind_quotas(coverage_missing_by_kind)
    normalized_focus = _normalize_focus_kinds(focus_kinds)

    selected, capacity_skipped = _select_tasks_round_robin(
        ranked_tasks=ranked_tasks,
        max_total=max_total,
        max_pages=max_pages,
        max_embedded_assets=max_embedded_assets,
        max_non_page_media=max_non_page_media,
        quotas=normalized_quotas,
        coverage_missing_by_kind=normalized_missing,
        focus_kinds=normalized_focus,
        remaining_capacity=remaining_capacity,
        host_of=host_of,
    )
    truncated_count = max(
        0,
        len(ranked_tasks) - len(selected) - len(capacity_skipped),
    )
    return selected, capacity_skipped, truncated_count


def _select_tasks_round_robin(
    *,
    ranked_tasks: list[CrawlTask],
    max_total: int,
    max_pages: int,
    max_embedded_assets: int,
    max_non_page_media: int,
    quotas: Mapping[MediaKind, int],
    coverage_missing_by_kind: Mapping[MediaKind, int],
    focus_kinds: set[MediaKind],
    remaining_capacity: Mapping[HostKindKey, int] | None,
    host_of: Callable[[CrawlTask], str | None] | None,
) -> tuple[list[CrawlTask], list[CrawlTask]]:
    """Select fairly across kinds while respecting every configured cap."""

    total_cap = max(0, int(max_total))
    if total_cap == 0:
        return [], []

    buckets: dict[MediaKind, deque[CrawlTask]] = defaultdict(deque)
    for task in ranked_tasks:
        buckets[task.kind].append(task)

    reservation_kinds = _coverage_reservation_kinds(
        coverage_missing_by_kind=coverage_missing_by_kind,
        focus_kinds=focus_kinds,
    )
    ordered_kinds = tuple(
        dict.fromkeys(
            reservation_kinds + _MEDIA_KIND_PRIORITY + (MediaKind.PAGE,)
        )
    )
    ordered_kinds += tuple(sorted(set(buckets) - set(ordered_kinds)))

    selected: list[CrawlTask] = []
    capacity_skipped: list[CrawlTask] = []
    counts: dict[MediaKind, int] = defaultdict(int)
    page_count = 0
    embedded_asset_count = 0
    non_page_media_count = 0
    capacity_left: dict[HostKindKey, int] | None = (
        dict(remaining_capacity) if remaining_capacity is not None else None
    )

    while len(selected) < total_cap and buckets:
        progressed = False
        for kind in ordered_kinds:
            bucket = buckets.get(kind)
            if not bucket:
                buckets.pop(kind, None)
                continue

            candidate = bucket.popleft()
            if not bucket:
                buckets.pop(kind, None)

            is_page = candidate.kind is MediaKind.PAGE
            is_embedded = candidate.source_type == "embedded_asset"
            is_non_page_media = candidate.kind is not MediaKind.PAGE
            allowed = (
                (not is_page or page_count < max(0, int(max_pages)))
                and (
                    not is_embedded
                    or embedded_asset_count < max(0, int(max_embedded_assets))
                )
                and (
                    not is_non_page_media
                    or non_page_media_count < max(0, int(max_non_page_media))
                )
                and not _kind_quota_exhausted(
                    task=candidate,
                    quotas=quotas,
                    counts=counts,
                )
            )
            if not allowed:
                continue

            capacity_entry = _capacity_entry(
                task=candidate,
                capacity_left=capacity_left,
                host_of=host_of,
            )
            capacity_remaining = (
                None if capacity_entry is None else capacity_entry[1]
            )
            if capacity_remaining == 0:
                capacity_skipped.append(candidate)
                progressed = True
                continue
            if capacity_entry is not None:
                capacity_key, capacity_remaining = capacity_entry
                if capacity_left is None:
                    raise RuntimeError("capacity entry requires capacity map")
                capacity_left[capacity_key] = capacity_remaining - 1

            selected.append(candidate)
            counts[candidate.kind] += 1
            page_count += int(is_page)
            embedded_asset_count += int(is_embedded)
            non_page_media_count += int(is_non_page_media)
            progressed = True
            if len(selected) >= total_cap:
                break

        if not progressed:
            break

    return selected, capacity_skipped


def _capacity_entry(
    *,
    task: CrawlTask,
    capacity_left: Mapping[HostKindKey, int] | None,
    host_of: Callable[[CrawlTask], str | None] | None,
) -> tuple[HostKindKey, int] | None:
    """Return a frontier-capacity entry, or ``None`` when unknown.

    Unknown capacity (host absent from the scheduler view) is treated as
    available; the scheduler remains the final authority during admission.
    """

    if capacity_left is None or host_of is None:
        return None
    host = host_of(task)
    if host is None:
        return None
    key = (host, task.kind)
    remaining = capacity_left.get(key)
    if remaining is None:
        return None
    return key, remaining


def _normalize_kind_quotas(
    quotas: Mapping[str, int] | None,
) -> dict[MediaKind, int]:
    if quotas is None:
        return {}
    normalized: dict[MediaKind, int] = {}
    for kind, value in quotas.items():
        try:
            normalized_kind = MediaKind.parse(kind)
        except (TypeError, ValueError):
            continue
        if normalized_kind not in _MEDIA_KINDS:
            continue
        try:
            normalized[normalized_kind] = max(0, int(value))
        except (TypeError, ValueError):
            normalized[normalized_kind] = 0
    return normalized


def _normalize_focus_kinds(kinds: Iterable[str]) -> set[MediaKind]:
    normalized: set[MediaKind] = set()
    for kind in kinds:
        try:
            parsed_kind = MediaKind.parse(kind)
        except (TypeError, ValueError):
            continue
        if parsed_kind in _MEDIA_KINDS:
            normalized.add(parsed_kind)
    return normalized


def _kind_quota_exhausted(
    *,
    task: CrawlTask,
    quotas: Mapping[MediaKind, int],
    counts: Mapping[MediaKind, int],
) -> bool:
    if task.kind not in quotas:
        return False
    return counts.get(task.kind, 0) >= quotas[task.kind]


def _coverage_reservation_kinds(
    *,
    coverage_missing_by_kind: Mapping[MediaKind, int],
    focus_kinds: set[MediaKind],
) -> tuple[MediaKind, ...]:
    if not focus_kinds:
        return ()
    return tuple(
        kind
        for kind in _MEDIA_KIND_PRIORITY
        if kind in focus_kinds and kind is not MediaKind.FEED
    )
