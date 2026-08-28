"""Budget, caps, pressure and scan logic for page discovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from config.coverage.settings import CoverageSettings
from config.validation.coverage_settings import nonnegative_int
from crawler.coverage.discovery_quotas import (
    resolve_kind_quotas as _resolve_kind_quotas,
)
from crawler.coverage.snapshot import (
    CoverageSnapshot,
    CoverageSnapshotProvider,
    CoverageUnavailableError,
    freeze_count_mapping,
    normalize_snapshot_missing,
)

if TYPE_CHECKING:
    from config.collection.processors import PageProcessorSettings
    from crawler.scheduling.progress.scheduler_snapshot_reader import (
        DiscoveryCapacitySnapshot,
    )
    from crawler.scheduling.url_scheduler import UrlScheduler


type DiscoveryPressureState = Literal[
    "normal",
    "warm",
    "high",
    "critical",
    "complete",
]


_TARGET_MEDIA_KINDS = (
    "document",
    "audio",
    "video",
    "image",
)


@dataclass(frozen=True, slots=True)
class PageDiscoveryBudget:
    """Resolved discovery limits from one immutable coverage snapshot."""

    pressure_state: DiscoveryPressureState
    max_total: int
    max_pages: int
    max_embedded_assets: int
    max_non_page_media: int
    discovery_scan_budget: int
    coverage_missing_by_kind: Mapping[str, int]
    kind_quotas: Mapping[str, int]
    coverage_snapshot_version: int = 0
    coverage_snapshot_captured_at_monotonic: float = 0.0
    coverage_snapshot_source: str = "unknown"
    scheduler_capacity: DiscoveryCapacitySnapshot | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.coverage_missing_by_kind,
            MappingProxyType,
        ):
            object.__setattr__(
                self,
                "coverage_missing_by_kind",
                freeze_count_mapping(self.coverage_missing_by_kind),
            )
        object.__setattr__(
            self,
            "kind_quotas",
            freeze_count_mapping(self.kind_quotas),
        )


@dataclass(frozen=True, slots=True)
class PageDiscoveryCaps:
    """Caps for categories of discovered tasks."""

    total_task_cap: int
    page_task_cap: int
    embedded_asset_task_cap: int
    non_page_media_task_cap: int


_ZERO_CAPS = PageDiscoveryCaps(
    total_task_cap=0,
    page_task_cap=0,
    embedded_asset_task_cap=0,
    non_page_media_task_cap=0,
)


class PageDiscoveryCapResolver:
    """Resolve discovery caps and quotas from pressure and coverage."""

    def __init__(
        self,
        *,
        coverage_settings: CoverageSettings,
    ) -> None:
        self._coverage_settings = coverage_settings

    async def resolve_budget(
        self,
        *,
        settings: PageProcessorSettings,
        scheduler: UrlScheduler,
        coverage_tracker: CoverageSnapshotProvider | None,
        page_url: str | None = None,
        source_name: str | None = None,
    ) -> PageDiscoveryBudget:
        """Resolve one page budget from exactly one coverage snapshot."""

        coverage_snapshot = self._read_coverage_snapshot(
            tracker=coverage_tracker,
            page_url=page_url,
            source_name=source_name,
        )
        coverage_missing = normalize_snapshot_missing(
            snapshot=coverage_snapshot,
            media_kinds=self._coverage_settings.kinds.media_kinds,
        )

        if not _has_positive_gap(coverage_missing):
            return PageDiscoveryBudget(
                pressure_state="complete",
                max_total=0,
                max_pages=0,
                max_embedded_assets=0,
                max_non_page_media=0,
                discovery_scan_budget=0,
                coverage_missing_by_kind=coverage_missing,
                kind_quotas={kind: 0 for kind in coverage_missing},
                coverage_snapshot_version=coverage_snapshot.version,
                coverage_snapshot_captured_at_monotonic=(
                    coverage_snapshot.captured_at_monotonic
                ),
                coverage_snapshot_source=coverage_snapshot.source,
            )

        scheduler_snapshot = await scheduler.snapshot()
        scheduler_capacity = await self._read_scheduler_capacity(
            scheduler=scheduler,
        )

        pressure_state = self.pressure_state(
            settings=settings,
            queue_size=scheduler_snapshot.total_queued,
        )

        caps = self._resolve_caps(
            settings=settings,
            pressure_state=pressure_state,
            coverage_missing_by_kind=coverage_missing,
        )

        caps = _apply_live_coverage_gap_caps(
            settings=settings,
            pressure_state=pressure_state,
            caps=caps,
            coverage_missing_by_kind=coverage_missing,
            non_target_slots=(self._coverage_settings.focus.non_target_slots),
        )

        kind_quotas = self.resolve_kind_quotas(
            settings=settings,
            pressure_state=pressure_state,
            max_total=caps.total_task_cap,
            max_embedded_assets=caps.embedded_asset_task_cap,
            max_non_page_media=caps.non_page_media_task_cap,
            coverage_missing_by_kind=coverage_missing,
        )
        discovery_scan_budget = self.resolve_scan_budget(
            settings=settings,
            max_total=caps.total_task_cap,
            coverage_missing_by_kind=coverage_missing,
        )

        return PageDiscoveryBudget(
            pressure_state=pressure_state,
            max_total=caps.total_task_cap,
            max_pages=caps.page_task_cap,
            max_embedded_assets=caps.embedded_asset_task_cap,
            max_non_page_media=caps.non_page_media_task_cap,
            discovery_scan_budget=discovery_scan_budget,
            coverage_missing_by_kind=coverage_missing,
            kind_quotas=kind_quotas,
            coverage_snapshot_version=coverage_snapshot.version,
            coverage_snapshot_captured_at_monotonic=(
                coverage_snapshot.captured_at_monotonic
            ),
            coverage_snapshot_source=coverage_snapshot.source,
            scheduler_capacity=scheduler_capacity,
        )

    @staticmethod
    async def _read_scheduler_capacity(
        *, scheduler: UrlScheduler
    ) -> DiscoveryCapacitySnapshot | None:
        capacity_provider = getattr(
            scheduler,
            "discovery_capacity_snapshot",
            None,
        )
        if not callable(capacity_provider):
            return None
        try:
            return cast(
                "DiscoveryCapacitySnapshot",
                await capacity_provider(),
            )
        except Exception:
            return None

    def _read_coverage_snapshot(
        self,
        *,
        tracker: CoverageSnapshotProvider | None,
        page_url: str | None,
        source_name: str | None,
    ) -> CoverageSnapshot:
        focus_enabled = bool(self._coverage_settings.focus.enabled)
        tracker_type = (
            "none"
            if tracker is None
            else f"{type(tracker).__module__}.{type(tracker).__qualname__}"
        )

        if tracker is None:
            raise CoverageUnavailableError(
                "live coverage snapshot provider is required",
                operation="page_discovery_budget",
                focus_enabled=focus_enabled,
                tracker_type=tracker_type,
                page_url=page_url,
                source_name=source_name,
            )

        try:
            snapshot = tracker.snapshot()
        except CoverageUnavailableError:
            raise
        except Exception as exc:
            raise CoverageUnavailableError(
                "coverage snapshot provider failed",
                operation="page_discovery_budget",
                focus_enabled=focus_enabled,
                tracker_type=tracker_type,
                page_url=page_url,
                source_name=source_name,
            ) from exc

        if not isinstance(snapshot, CoverageSnapshot):
            raise CoverageUnavailableError(
                "coverage snapshot provider returned an invalid snapshot",
                operation="page_discovery_budget",
                focus_enabled=focus_enabled,
                tracker_type=tracker_type,
                page_url=page_url,
                source_name=source_name,
            )

        return snapshot

    def resolve(
        self,
        *,
        settings: PageProcessorSettings,
        queue_size: int,
        coverage_missing_by_kind: Mapping[str, int],
    ) -> PageDiscoveryCaps:
        """Resolve caps for a known queue size."""

        return self._resolve_caps(
            settings=settings,
            pressure_state=self.pressure_state(
                settings=settings,
                queue_size=queue_size,
            ),
            coverage_missing_by_kind=coverage_missing_by_kind,
        )

    def _resolve_caps(
        self,
        *,
        settings: PageProcessorSettings,
        pressure_state: DiscoveryPressureState,
        coverage_missing_by_kind: Mapping[str, int],
    ) -> PageDiscoveryCaps:
        if pressure_state == "critical":
            return _ZERO_CAPS

        if pressure_state == "high":
            caps = PageDiscoveryCaps(
                total_task_cap=(
                    settings.max_discovered_tasks_per_page_under_pressure
                ),
                page_task_cap=(settings.max_pages_per_page_under_pressure),
                embedded_asset_task_cap=(
                    settings.max_embedded_assets_per_page_under_pressure
                ),
                non_page_media_task_cap=(
                    settings.max_non_page_media_per_page_under_pressure
                ),
            )
        else:
            caps = PageDiscoveryCaps(
                total_task_cap=(settings.max_discovered_tasks_per_page),
                page_task_cap=settings.max_pages_per_page,
                embedded_asset_task_cap=(
                    settings.max_embedded_assets_per_page
                ),
                non_page_media_task_cap=(settings.max_non_page_media_per_page),
            )

        return _apply_focus_caps(
            settings=settings,
            caps=caps,
            focus_kinds=_positive_kinds(coverage_missing_by_kind),
        )

    @staticmethod
    def pressure_state(
        *,
        settings: PageProcessorSettings,
        queue_size: int,
    ) -> DiscoveryPressureState:
        """Classify discovery queue pressure."""

        high_threshold = settings.discovery_queue_high_watermark

        critical_threshold = max(
            high_threshold + 1,
            settings.discovery_queue_critical_watermark,
        )

        warm_threshold = max(
            1,
            high_threshold * 2 // 3,
        )

        if queue_size >= critical_threshold:
            return "critical"

        if queue_size >= high_threshold:
            return "high"

        if queue_size >= warm_threshold:
            return "warm"

        return "normal"

    async def apply_drain_budget(
        self,
        *,
        scheduler: UrlScheduler,
        pressure_state: DiscoveryPressureState,
        max_total: int,
        max_pages: int,
        max_embedded_assets: int,
        max_non_page_media: int,
        coverage_missing_by_kind: Mapping[str, int] | None,
    ) -> PageDiscoveryCaps:
        """Apply the scheduler drain allowance for direct callers."""

        caps = PageDiscoveryCaps(
            total_task_cap=max_total,
            page_task_cap=max_pages,
            embedded_asset_task_cap=max_embedded_assets,
            non_page_media_task_cap=max_non_page_media,
        )

        if (
            pressure_state != "high"
            or max_total <= 0
            or coverage_missing_by_kind is None
            or _has_positive_gap(coverage_missing_by_kind)
        ):
            return caps

        allowed_total = max(
            0,
            await scheduler.discovery_drain_budget(
                configured_cap=max_total,
                force=True,
            ),
        )

        return PageDiscoveryCaps(
            total_task_cap=allowed_total,
            page_task_cap=min(
                max_pages,
                allowed_total,
            ),
            embedded_asset_task_cap=min(
                max_embedded_assets,
                allowed_total,
            ),
            non_page_media_task_cap=min(
                max_non_page_media,
                allowed_total,
            ),
        )

    def resolve_kind_quotas(
        self,
        *,
        settings: PageProcessorSettings,
        pressure_state: DiscoveryPressureState,
        max_total: int,
        max_embedded_assets: int,
        max_non_page_media: int,
        coverage_missing_by_kind: Mapping[str, int],
    ) -> dict[str, int]:
        """Resolve per-kind discovery quotas."""

        return _resolve_kind_quotas(
            coverage_settings=self._coverage_settings,
            settings=settings,
            pressure_state=pressure_state,
            max_total=max_total,
            max_embedded_assets=max_embedded_assets,
            max_non_page_media=max_non_page_media,
            coverage_missing_by_kind=(coverage_missing_by_kind),
        )

    def resolve_scan_budget(
        self,
        *,
        settings: PageProcessorSettings,
        max_total: int,
        coverage_missing_by_kind: Mapping[str, int],
    ) -> int:
        """Return the number of discovered candidates to inspect."""

        if max_total <= 0 or settings.max_links_per_page <= 0:
            return 0

        multiplier = (
            self._coverage_settings.focus.focused_discovery_scan_multiplier
            if _has_positive_gap(coverage_missing_by_kind)
            else (self._coverage_settings.focus.discovery_scan_multiplier)
        )

        return min(
            settings.max_links_per_page,
            max_total * multiplier,
        )


def _positive_kinds(
    missing_by_kind: Mapping[str, int] | None,
) -> set[str]:
    return {
        kind
        for kind, missing in (missing_by_kind or {}).items()
        if nonnegative_int(missing) > 0
    }


def _has_positive_gap(
    missing_by_kind: Mapping[str, int] | None,
) -> bool:
    return any(
        nonnegative_int(missing) > 0
        for missing in (missing_by_kind or {}).values()
    )


def _apply_live_coverage_gap_caps(
    *,
    settings: PageProcessorSettings,
    pressure_state: DiscoveryPressureState,
    caps: PageDiscoveryCaps,
    coverage_missing_by_kind: Mapping[str, int],
    non_target_slots: int,
) -> PageDiscoveryCaps:
    """Reserve discovery capacity for missing media kinds."""

    if pressure_state == "critical" or caps.total_task_cap <= 0:
        return _ZERO_CAPS

    target_kinds = tuple(
        kind
        for kind in _TARGET_MEDIA_KINDS
        if nonnegative_int(coverage_missing_by_kind.get(kind, 0)) > 0
    )

    if not target_kinds:
        return caps

    max_total = caps.total_task_cap
    max_pages = caps.page_task_cap
    max_embedded_assets = caps.embedded_asset_task_cap
    max_non_page_media = caps.non_page_media_task_cap

    reserved_by_kind = settings.multimodal_reserved_slots_by_kind

    target_asset_slots = sum(
        min(
            coverage_missing_by_kind[kind],
            max(
                1,
                nonnegative_int(reserved_by_kind.get(kind, 1)),
            ),
        )
        for kind in target_kinds
    )

    required_total = target_asset_slots

    if settings.max_discovered_tasks_per_page > 0:
        required_total = min(
            required_total,
            settings.max_discovered_tasks_per_page,
        )

    if pressure_state != "high":
        max_total = max(
            max_total,
            required_total,
        )

    for kind in target_kinds:
        reserved_for_kind = min(
            coverage_missing_by_kind[kind],
            max(
                1,
                nonnegative_int(reserved_by_kind.get(kind, 1)),
            ),
        )

        if kind == "image":
            max_embedded_assets = max(
                max_embedded_assets,
                reserved_for_kind,
            )
            continue

        max_non_page_media = max(
            max_non_page_media,
            reserved_for_kind,
        )

        max_embedded_assets = max(
            max_embedded_assets,
            1,
        )

    # Pages are non-target discovery work whenever a modality is missing.
    # Keep a very small scouting allowance so pages can still reveal new
    # direct assets, without allowing a broad HTML frontier to crowd them
    # out of the same discovery batch.
    max_pages = min(
        max_pages,
        max(0, int(non_target_slots)),
    )

    if settings.max_media_assets_per_page > 0:
        max_embedded_assets = min(
            max_embedded_assets,
            settings.max_media_assets_per_page,
        )

    return PageDiscoveryCaps(
        total_task_cap=max_total,
        page_task_cap=min(
            max_pages,
            max_total,
        ),
        embedded_asset_task_cap=min(
            max_embedded_assets,
            max_total,
        ),
        non_page_media_task_cap=min(
            max_non_page_media,
            max_total,
        ),
    )


def _apply_focus_caps(
    *,
    settings: PageProcessorSettings,
    caps: PageDiscoveryCaps,
    focus_kinds: set[str],
) -> PageDiscoveryCaps:
    max_total = caps.total_task_cap
    max_pages = caps.page_task_cap
    max_embedded_assets = caps.embedded_asset_task_cap
    max_non_page_media = caps.non_page_media_task_cap
    if not focus_kinds or max_total <= 0:
        # Fallback to simple caps if no focus info.
        if not focus_kinds:
            return caps

    # Coverage-focus mag page discovery niet naar 2 links knijpen.
    if focus_kinds and focus_kinds.intersection(
        {"image", "audio", "video", "document"}
    ):
        reserved_page_slots = settings.multimodal_reserved_media_page_slots
        if reserved_page_slots <= 0:
            reserved_page_slots = 12

        max_pages = max(max_pages, min(max_total, reserved_page_slots))
        max_embedded_assets = max(max_embedded_assets, 24)
        max_non_page_media = max(max_non_page_media, 16)

    max_media_assets = settings.max_media_assets_per_page
    if max_media_assets > 0:
        max_embedded_assets = min(max_embedded_assets, max_media_assets)

    return PageDiscoveryCaps(
        total_task_cap=max_total,
        page_task_cap=min(max_pages, max_total),
        embedded_asset_task_cap=max_embedded_assets,
        non_page_media_task_cap=max_non_page_media,
    )
