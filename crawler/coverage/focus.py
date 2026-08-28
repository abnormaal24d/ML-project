"""Derive crawler focus policy from the canonical coverage-gap analysis."""

from __future__ import annotations

from config.collection.processors import PageProcessorSettings
from config.coverage.settings import CoverageFocusSettings, CoverageSettings


def focus_kinds(
    *,
    settings: CoverageSettings,
    missing_by_kind: dict[str, int],
) -> tuple[str, ...]:
    """Return focus kinds in a stable crawler media order.

    Document gaps get document-specific priority so that image/video do not
    simultaneously claim focus capacity.
    """

    effective_priority = settings.focus.media_priority
    priority_map = {
        kind: index for index, kind in enumerate(effective_priority)
    }
    effective_kinds = set(effective_priority) | set(settings.kinds.media_kinds)

    # Document gaps get priority ahead of equally urgent media gaps.
    has_document_gap = missing_by_kind.get("document", 0) > 0

    def sort_key(
        item: tuple[str, int],
    ) -> tuple[bool, bool, int, int, str]:
        kind, missing = item
        has_missing = missing > 0
        base = (
            not has_missing,
            kind == "image" if has_missing else False,
            -missing,
            priority_map.get(kind, len(priority_map)),
            kind,
        )
        if has_document_gap and kind == "document":
            # boost document to front when it's the missing one
            return (
                False,
                False,
                -9999,
                base[2],
                kind,
            )  # very high priority
        return base

    excluded = {"feed"} if settings.focus.exclude_feed_from_focus else set()
    return tuple(
        kind
        for kind, missing in sorted(
            missing_by_kind.items(),
            key=sort_key,
        )
        if kind in effective_kinds and kind not in excluded and missing > 0
    )


def focused_page_settings(
    *,
    page_settings: PageProcessorSettings,
    focus_settings: CoverageFocusSettings,
    focus_kinds: tuple[str, ...],
) -> PageProcessorSettings:
    """Return focused page-processor settings for the given focus kinds.

    Pure policy: page settings + coverage focus settings + focus kinds ->
    focused page settings. Invariants are re-validated through the settings
    model so a focused policy can never violate page-processor constraints.
    """
    focus_kinds_set = set(focus_kinds)

    if not focus_kinds_set:
        return page_settings

    updates: dict[str, object] = {}

    if "document" in focus_kinds_set:
        normal = max(
            page_settings.max_non_page_media_per_page,
            min(page_settings.max_discovered_tasks_per_page, 48),
        )
        under_pressure = min(
            normal,
            max(page_settings.max_non_page_media_per_page_under_pressure, 24),
        )
        updates["max_non_page_media_per_page"] = normal
        updates["max_non_page_media_per_page_under_pressure"] = under_pressure

    if "image" in focus_kinds_set:
        updates["max_embedded_assets_per_page"] = max(
            page_settings.max_embedded_assets_per_page,
            min(page_settings.max_discovered_tasks_per_page, 24),
        )

    if {"audio", "video"} & focus_kinds_set:
        configured_max = updates.get(
            "max_non_page_media_per_page",
            page_settings.max_non_page_media_per_page,
        )
        current_max = (
            configured_max
            if isinstance(configured_max, int)
            and not isinstance(configured_max, bool)
            else page_settings.max_non_page_media_per_page
        )
        updates["max_non_page_media_per_page"] = max(
            current_max,
            min(
                page_settings.max_discovered_tasks_per_page,
                page_settings.max_non_page_media_per_page * 2,
            ),
        )

    # Reserve focused discovery batches for direct modality candidates.
    page_limit = max(0, int(focus_settings.non_target_slots))
    updates["max_pages_per_page"] = min(
        page_settings.max_pages_per_page,
        page_limit,
    )
    updates["max_pages_per_page_under_pressure"] = min(
        page_settings.max_pages_per_page_under_pressure,
        page_limit,
    )
    updates["max_pages_per_page_critical"] = min(
        page_settings.max_pages_per_page_critical,
        page_limit,
    )

    focused = page_settings.model_copy(update=updates)
    # Validate to ensure invariants like under_pressure <= normal (per audit/defect 10)
    return type(focused).model_validate(focused.model_dump())
