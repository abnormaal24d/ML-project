"""Contracts for the crawler state persistence subgraph.

The state builder produces only storage primitives that exist at build
time. Runtime reader/writer services belong to the execution subgraph.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from config.settings.crawler import CrawlerSettings, CrawlStateStoreSettings
from crawler.runtime.control.crawler_control_directory import (
    CrawlerControlDirectory,
)
from orchestration.composition.runtime.crawler_state import (
    CrawlerStatePersistence,
    build_crawler_state,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class _Clock:
    def now(self):
        from datetime import UTC, datetime

        return datetime(2026, 1, 1, tzinfo=UTC)


def _logger_factory():
    from logger.factory import ProjectLoggerFactory

    return ProjectLoggerFactory(
        root_name="state-persistence-test",
        base_context={},
    )


def test_state_persistence_contract_contains_only_constructed_services() -> (
    None
):
    """No None-placeholders for services that are built elsewhere."""

    fields = CrawlerStatePersistence.__dataclass_fields__

    assert set(fields) == {
        "checkpoint_store",
        "dead_letter_writer",
        "dead_letter_path",
    }


def test_build_crawler_state_has_no_runtime_dependency_parameters() -> None:
    """The builder needs storage inputs only — no runtime graph objects."""

    parameters = inspect.signature(build_crawler_state).parameters

    assert set(parameters) == {
        "state_settings",
        "control_directory",
        "clock",
        "logger_factory",
        "crawl_session_id",
    }

    for forbidden in (
        "settings",
        "run_context",
        "scheduler",
        "worker_pool",
        "metrics",
        "host_normalizer",
        "id_generator",
        "shutdown_manager",
    ):
        assert forbidden not in parameters, forbidden


def test_disabled_state_settings_yield_an_empty_persistence(tmp_path: Path) -> (
    None
):
    control_directory = CrawlerControlDirectory(
        settings=CrawlerSettings(),
        project_root=tmp_path,
    )
    persistence = build_crawler_state(
        state_settings=CrawlStateStoreSettings(enabled=False),
        control_directory=control_directory,
        clock=_Clock(),
        logger_factory=_logger_factory(),
        crawl_session_id="session-1",
    )

    assert persistence.checkpoint_store is None
    assert persistence.dead_letter_writer is None
    assert persistence.dead_letter_path is None


def test_enabled_state_settings_build_real_storage_primitives(
    tmp_path: Path,
) -> None:
    from crawler.runtime.state.crawl_checkpoint_store import (
        CrawlerCheckpointStore,
    )
    from crawler.runtime.state.crawl_dead_letter_writer import (
        CrawlerDeadLetterWriter,
    )

    state_settings = CrawlStateStoreSettings(
        enabled=True,
        dead_letter_enabled=True,
    )
    control_directory = CrawlerControlDirectory(
        settings=CrawlerSettings(),
        project_root=tmp_path,
    )
    persistence = build_crawler_state(
        state_settings=state_settings,
        control_directory=control_directory,
        clock=_Clock(),
        logger_factory=_logger_factory(),
        crawl_session_id="session-1",
    )

    assert isinstance(persistence.checkpoint_store, CrawlerCheckpointStore)
    assert isinstance(persistence.dead_letter_writer, CrawlerDeadLetterWriter)
    assert persistence.dead_letter_path is not None
    assert persistence.dead_letter_path.is_relative_to(tmp_path)


def test_state_builder_module_does_not_import_root_settings() -> None:
    """State persistence receives one typed subconfig, never root Settings."""

    import ast

    source_path = (
        Path(__file__).resolve().parents[4]
        / "orchestration"
        / "composition"
        / "runtime"
        / "crawler_state.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "config.settings.root"
            if node.module:
                assert not node.module.startswith("config.settings.root")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "config.settings.root"


def test_state_persistence_is_frozen() -> None:
    assert CrawlerStatePersistence.__dataclass_params__.frozen is True
