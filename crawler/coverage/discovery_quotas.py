"""Quota and kind allocation logic for page discovery (coverage-driven)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from config.coverage.settings import CoverageSettings
from config.validation.coverage_settings import (
    nonnegative_int as _nonnegative_int,
)

if TYPE_CHECKING:
    from config.collection.processors import PageProcessorSettings


def resolve_kind_quotas(
    *,
    coverage_settings: CoverageSettings,
    settings: PageProcessorSettings,
    pressure_state: str,
    max_total: int,
    max_embedded_assets: int,
    max_non_page_media: int,
    coverage_missing_by_kind: Mapping[str, int],
) -> dict[str, int]:
    """Return per-kind media discovery caps for one page."""

    configured = settings.multimodal_reserved_slots_by_kind

    total_media_capacity = min(
        max(0, int(max_total)),
        max(0, int(max_embedded_assets) + int(max_non_page_media)),
    )
    if pressure_state == "critical" or total_media_capacity <= 0:
        return {kind: 0 for kind in _media_kind_priority(coverage_settings)}

    pressure_cap = (
        max(1, total_media_capacity)
        if pressure_state == "high"
        else total_media_capacity
    )
    missing = coverage_missing_by_kind
    priority = _priority_by_missing_coverage(
        base_priority=_media_kind_priority(coverage_settings),
        coverage_missing_by_kind=missing,
    )

    quotas: dict[str, int] = {}
    for kind in priority:
        raw_value = configured.get(kind, pressure_cap)
        try:
            quota = int(raw_value)
        except (TypeError, ValueError):
            quota = 0
        quotas[kind] = min(max(0, quota), pressure_cap)
    focused = _apply_focus(
        coverage_settings=coverage_settings,
        settings=settings,
        quotas=quotas,
        pressure_cap=pressure_cap,
        coverage_missing_by_kind=missing,
    )
    gap_adjusted = _apply_required_media_gap_quotas(
        coverage_settings=coverage_settings,
        settings=settings,
        quotas=focused,
        coverage_missing_by_kind=missing,
    )
    adjusted = _apply_dynamic_coverage_quotas(
        quotas=gap_adjusted,
        coverage_missing_by_kind=missing,
        pressure_cap=pressure_cap,
    )
    return _distribute_kind_quotas(
        adjusted=adjusted,
        max_total=max_total,
        coverage_missing=missing,
    )


def _apply_dynamic_coverage_quotas(
    *,
    quotas: dict[str, int],
    coverage_missing_by_kind: Mapping[str, int],
    pressure_cap: int,
) -> dict[str, int]:
    """Resolve per-kind discovery quotas from live coverage gaps."""

    adjusted: dict[str, int] = {}

    for kind in ("document", "audio", "video", "image"):
        missing = _nonnegative_int(coverage_missing_by_kind.get(kind, 0))
        base = max(0, int(quotas.get(kind, 0)))

        if missing <= 0:
            adjusted[kind] = 0
            continue

        boosted = max(1, base)
        if kind == "document":
            boosted = max(boosted, base * 3, 6)
        elif missing >= 10:
            boosted = max(boosted, base * 2, 3)

        adjusted[kind] = min(pressure_cap, boosted)

    adjusted["feed"] = (
        max(0, int(quotas.get("feed", 0)))
        if _nonnegative_int(coverage_missing_by_kind.get("audio", 0)) > 0
        else 0
    )

    return adjusted


def _apply_focus(
    *,
    coverage_settings: CoverageSettings,
    settings: PageProcessorSettings,
    quotas: dict[str, int],
    pressure_cap: int,
    coverage_missing_by_kind: Mapping[str, int],
) -> dict[str, int]:
    focus_kinds = {
        kind
        for kind, missing in coverage_missing_by_kind.items()
        if kind in _media_kind_priority(coverage_settings)
        and kind != "feed"
        and _nonnegative_int(missing) > 0
    }
    if not focus_kinds:
        return quotas

    multiplier = coverage_settings.focus.boost_multiplier
    non_target_slots = coverage_settings.focus.non_target_slots

    priority = _priority_by_missing_coverage(
        base_priority=_media_kind_priority(coverage_settings),
        coverage_missing_by_kind=coverage_missing_by_kind,
    )

    focused: dict[str, int] = {}
    for kind in priority:
        try:
            base_quota = max(0, int(quotas.get(kind, 0)))
        except (TypeError, ValueError):
            base_quota = 0

        if kind in focus_kinds:
            focused[kind] = min(
                pressure_cap,
                max(1, base_quota * multiplier),
            )
            continue

        if kind == "feed":
            focused[kind] = base_quota
            continue

        focused[kind] = min(base_quota, non_target_slots)

    return focused


def _apply_required_media_gap_quotas(
    *,
    coverage_settings: CoverageSettings,
    settings: PageProcessorSettings,
    quotas: dict[str, int],
    coverage_missing_by_kind: Mapping[str, int],
) -> dict[str, int]:
    focus_kinds = _media_focus_kinds(
        coverage_missing_by_kind=coverage_missing_by_kind
    )
    if not focus_kinds:
        return quotas

    focused = dict(quotas)
    min_focus = coverage_settings.focus.minimum_focus_slots_by_kind
    if "image" in focus_kinds:
        focused["image"] = max(
            focused.get("image", 0), int(min_focus.get("image", 3))
        )
    if "audio" in focus_kinds:
        focused["audio"] = max(
            focused.get("audio", 0), int(min_focus.get("audio", 2))
        )
    if "video" in focus_kinds:
        focused["video"] = max(
            focused.get("video", 0), int(min_focus.get("video", 2))
        )
    if "document" in focus_kinds:
        focused["document"] = max(
            focused.get("document", 0), int(min_focus.get("document", 3))
        )
    minimum_media = settings.min_media_assets_per_crawl_batch
    if minimum_media > 0:
        for kind in ("image", "audio", "video"):
            if kind in focus_kinds:
                focused[kind] = max(focused.get(kind, 0), minimum_media)
    return focused


def _media_focus_kinds(
    *, coverage_missing_by_kind: Mapping[str, int]
) -> set[str]:
    return {
        kind
        for kind, missing in coverage_missing_by_kind.items()
        if kind in {"image", "audio", "video", "document"}
        and _nonnegative_int(missing) > 0
    }


def _priority_by_missing_coverage(
    *,
    base_priority: tuple[str, ...],
    coverage_missing_by_kind: Mapping[str, int],
) -> tuple[str, ...]:
    order = {kind: index for index, kind in enumerate(base_priority)}
    return tuple(
        sorted(
            base_priority,
            key=lambda kind: _coverage_priority_key(
                kind=kind,
                missing=_nonnegative_int(
                    coverage_missing_by_kind.get(kind, 0)
                ),
                order=order,
            ),
        )
    )


def _coverage_priority_key(
    *,
    kind: str,
    missing: int,
    order: dict[str, int],
) -> tuple[bool, bool, int, int]:
    has_missing = missing > 0
    return (
        not has_missing,
        kind == "image" if has_missing else False,
        -missing,
        order[kind],
    )


def _distribute_kind_quotas(
    *,
    adjusted: dict[str, int],
    max_total: int,
    coverage_missing: Mapping[str, int],
) -> dict[str, int]:
    """Distribute max_total across media kinds as hard per-kind quotas."""

    quota_budget = max(0, int(max_total))
    if quota_budget <= 0:
        return {
            kind: 0 for kind in ("document", "audio", "video", "image", "feed")
        }

    desired = {
        kind: max(0, int(adjusted.get(kind, 0)))
        for kind in _media_kind_priority_for_dist()
    }
    distributed: dict[str, int] = {
        kind: 0 for kind in _media_kind_priority_for_dist()
    }
    remaining = quota_budget

    def _gap_priority_key(k: str) -> tuple[bool, int]:
        m = _nonnegative_int(coverage_missing.get(k, 0))
        return (k == "image", -m)

    positive_kinds = sorted(
        (
            kind
            for kind in ("document", "audio", "video", "image")
            if _nonnegative_int(coverage_missing.get(kind, 0)) > 0
        ),
        key=_gap_priority_key,
    )
    positive_kinds = list(positive_kinds)
    if (
        _nonnegative_int(coverage_missing.get("audio", 0)) > 0
        and desired.get("feed", 0) > 0
        and "feed" not in positive_kinds
    ):
        positive_kinds.append("feed")
    for kind in positive_kinds:
        if remaining <= 0:
            break
        distributed[kind] = 1
        remaining -= 1

    def _sort_key(k: str) -> tuple[bool, int]:
        m = _nonnegative_int(coverage_missing.get(k, 0))
        d = desired.get(k, 0)
        return (k == "image", -m if m > 0 else -d)

    weighted = sorted(
        positive_kinds
        or [
            kind
            for kind in _media_kind_priority_for_dist()
            if desired.get(kind, 0) > 0
        ],
        key=_sort_key,
    )

    while remaining > 0 and weighted:
        progressed = False
        for kind in weighted:
            if remaining <= 0:
                break
            if distributed[kind] >= desired.get(kind, 0):
                continue
            distributed[kind] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    return distributed


def _media_kind_priority(
    coverage_settings: CoverageSettings | None = None,
) -> tuple[str, ...]:
    if coverage_settings is not None:
        prio = coverage_settings.focus.media_priority
        if prio:
            return tuple(prio)
    return ("document", "audio", "video", "image", "feed")


def _media_kind_priority_for_dist() -> tuple[str, ...]:
    return ("document", "audio", "video", "image", "feed")
