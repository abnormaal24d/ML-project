"""Coverage-focus discovery-budget regressions."""

from __future__ import annotations

from config.collection.processors import PageProcessorSettings
from crawler.coverage.discovery_budget import (
    PageDiscoveryCaps,
    _apply_live_coverage_gap_caps,
)


def test_modality_focus_reserves_only_one_non_target_page_slot() -> None:
    caps = _apply_live_coverage_gap_caps(
        settings=PageProcessorSettings(),
        pressure_state="normal",
        caps=PageDiscoveryCaps(
            total_task_cap=20,
            page_task_cap=6,
            embedded_asset_task_cap=8,
            non_page_media_task_cap=8,
        ),
        coverage_missing_by_kind={"audio": 5, "video": 5},
        non_target_slots=1,
    )

    assert caps.page_task_cap == 1
    assert caps.non_page_media_task_cap >= 2
