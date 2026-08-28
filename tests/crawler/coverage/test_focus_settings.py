"""Regression contracts for coverage-gap focused page settings."""

from __future__ import annotations

from pathlib import Path

from config.load import load_settings
from crawler.coverage.focus import focus_kinds, focused_page_settings
from crawler.coverage.gaps import CoverageGapAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _dev_settings(tmp_path: Path):
    return load_settings(
        "dev",
        project_root=tmp_path / "project-root",
        config_root=PROJECT_ROOT,
        environment="dev",
    )


def _focused_page(settings, coverage_gaps: dict[str, int]):
    missing = CoverageGapAnalyzer(
        settings=settings.coverage
    ).missing_by_media_kind(coverage_gaps)
    kinds = focus_kinds(
        settings=settings.coverage,
        missing_by_kind=missing,
    )
    return focused_page_settings(
        page_settings=settings.collection.processors.page,
        focus_settings=settings.coverage.focus,
        focus_kinds=kinds,
    )


def _with_page_settings(tmp_path: Path, **page_fields: object):
    settings = _dev_settings(tmp_path)
    page_settings = settings.collection.processors.page.model_copy(
        update=page_fields
    )
    processors = settings.collection.processors.model_copy(
        update={"page": page_settings}
    )
    collection = settings.collection.model_copy(
        update={"processors": processors}
    )
    return settings.model_copy(update={"collection": collection})


def test_document_focus_clamps_under_pressure_to_normal(
    tmp_path: Path,
) -> None:
    settings = _with_page_settings(
        tmp_path,
        max_discovered_tasks_per_page=20,
        max_non_page_media_per_page=8,
        max_non_page_media_per_page_under_pressure=1,
        max_non_page_media_per_page_critical=0,
    )

    focused = _focused_page(
        settings,
        {"modality:document": 5},
    )

    assert focused.max_non_page_media_per_page == 20
    assert focused.max_non_page_media_per_page_under_pressure <= (
        focused.max_non_page_media_per_page
    )
    assert focused.max_non_page_media_per_page_critical <= (
        focused.max_non_page_media_per_page_under_pressure
    )


def test_document_focus_keeps_valid_existing_ratio(tmp_path: Path) -> None:
    settings = _with_page_settings(
        tmp_path,
        max_discovered_tasks_per_page=48,
        max_non_page_media_per_page=32,
        max_non_page_media_per_page_under_pressure=16,
        max_non_page_media_per_page_critical=4,
    )

    focused = _focused_page(
        settings,
        {"modality:document": 5},
    )

    assert focused.max_non_page_media_per_page == 48
    assert focused.max_non_page_media_per_page_under_pressure == 24
    assert focused.max_non_page_media_per_page_critical == 4
    assert focused.max_non_page_media_per_page_critical <= (
        focused.max_non_page_media_per_page_under_pressure
    )


def test_no_focus_kinds_returns_page_settings_unchanged(
    tmp_path: Path,
) -> None:
    settings = _dev_settings(tmp_path)
    page_settings = settings.collection.processors.page

    focused = focused_page_settings(
        page_settings=page_settings,
        focus_settings=settings.coverage.focus,
        focus_kinds=(),
    )

    assert focused is page_settings


def test_focus_does_not_mutate_the_canonical_tree(tmp_path: Path) -> None:
    """Focus must be exactly localised: no collateral mutation anywhere."""

    settings = _with_page_settings(
        tmp_path,
        max_discovered_tasks_per_page=48,
        max_non_page_media_per_page=8,
    )
    base_page = settings.collection.processors.page

    focused = focused_page_settings(
        page_settings=base_page,
        focus_settings=settings.coverage.focus,
        focus_kinds=("document",),
    )

    assert focused is not base_page
    assert settings.collection.processors.page is base_page
    assert settings.collection.processors.page == base_page

    # Non-focus fields stay canonical.
    assert focused.text_extraction == base_page.text_extraction
    assert focused.max_links_per_page == base_page.max_links_per_page
    assert focused.max_discovered_tasks_per_page == (
        base_page.max_discovered_tasks_per_page
    )

    # Only the discovery quotas the policy owns actually moved.
    assert (
        focused.max_non_page_media_per_page
        != base_page.max_non_page_media_per_page
    )
