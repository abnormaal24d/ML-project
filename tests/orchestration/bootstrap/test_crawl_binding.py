"""Bootstrap binding regression for the crawl ExecuteCrawl adapter.

Proves the workflow-facing ``execute_crawl`` capability forwards the
focused page policy as ``page_settings_override``, reuses the canonical
workflow settings, and never sends a ``page_settings`` keyword into
``execute_application``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from config.collection.processors import PageProcessorSettings
from config.load import load_settings
from crawler.runtime.loop.crawl_run_summary import CrawlTerminalOutcome
from datachecker.manifests.crawl_state_manifest_writer import (
    CrawlStateManifestWriter,
)
from orchestration.bootstrap.container import build_workflow_phase_executor
from orchestration.bootstrap.run_context import create_run_context

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.asyncio
async def test_crawl_adapter_forwards_override_and_canonical_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import orchestration.workflow.crawl.phase_runner as crawl_module

    settings = load_settings(
        "dev",
        project_root=tmp_path / "project-root",
        config_root=PROJECT_ROOT,
        environment="dev",
    )

    # Capture the real runner instance so its injected capability can be
    # invoked exactly the way CrawlPhaseRunner.run() invokes it.
    built: dict[str, object] = {}
    real_runner_cls = crawl_module.CrawlPhaseRunner

    class RecordingCrawlPhaseRunner(real_runner_cls):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            built["runner"] = self

    monkeypatch.setattr(
        crawl_module, "CrawlPhaseRunner", RecordingCrawlPhaseRunner
    )

    class _NoopLogger:
        def __getattr__(self, name: str):
            def _log(*_args: object, **_kwargs: object) -> None:
                return None

            return _log

    def _stub_logger_factory(*_args: object, **_kwargs: object):
        logger = _NoopLogger()
        return SimpleNamespace(
            get_logger=lambda *_a, **_k: logger,
            get_logger_for=lambda *_a, **_k: logger,
        )

    monkeypatch.setattr(
        "orchestration.bootstrap.logging.build_logger_factory",
        _stub_logger_factory,
    )
    monkeypatch.setattr(
        "orchestration.composition.runtime.dataset.build_data_checker",
        lambda **_k: SimpleNamespace(check=lambda **_k2: None),
    )
    manifest_writers = SimpleNamespace(
        crawl_state_manifest_writer=cast(
            CrawlStateManifestWriter,
            SimpleNamespace(),
        ),
        crawl_promotion=SimpleNamespace(commit=lambda **_k: None),
        preprocessing=SimpleNamespace(
            write_preprocessing_manifest=lambda: None
        ),
        augmentation=SimpleNamespace(write_augmentation_manifest=lambda: None),
        training=SimpleNamespace(
            write_training_manifests=lambda **_k: None,
            write_training_metrics=lambda **_k: None,
        ),
    )
    monkeypatch.setattr(
        "orchestration.composition.runtime.workflow_manifest_writers"
        ".build_workflow_manifest_writers",
        lambda **_k: manifest_writers,
    )
    monkeypatch.setattr(
        "orchestration.composition.runtime.augmentation"
        ".build_augmentation_workflow",
        lambda **_k: object(),
    )
    monkeypatch.setattr(
        "orchestration.composition.curated_snapshot"
        ".build_curated_snapshot_runtime",
        lambda **_k: SimpleNamespace(build=lambda **_k2: None),
    )
    monkeypatch.setattr(
        "orchestration.composition.preprocessing_dependencies"
        ".build_audio_materializer_factory",
        lambda **_k: lambda output_root: object(),
    )
    monkeypatch.setattr(
        "orchestration.composition.preprocessing_dependencies"
        ".build_video_materializer_factory",
        lambda **_k: lambda output_root: object(),
    )
    monkeypatch.setattr(
        "orchestration.composition.privacy.privacy_inspection_services"
        ".build_privacy_inspection_services",
        lambda **_k: SimpleNamespace(pii_detector=object()),
    )
    monkeypatch.setattr(
        "mmcrawler_datasets.snapshots.training_builder"
        ".build_training_snapshot",
        lambda **_k: object(),
    )

    received: dict[str, object] = {}

    async def recording_execute_crawl_application(
        **kwargs: object,
    ) -> object:
        received.update(kwargs)
        return SimpleNamespace(dataset_outcome=CrawlTerminalOutcome.SUCCESS)

    options = _make_options(tmp_path)

    build_workflow_phase_executor(
        options,
        workflow_context=create_run_context(stage="test"),
        settings=settings,
        runtime_readiness=cast(
            object,
            SimpleNamespace(
                processing_activity_registry=object(),
                dependency_report=object(),
            ),
        ),
        execute_crawl_application=recording_execute_crawl_application,
    )

    runner = built["runner"]
    assert isinstance(runner, real_runner_cls)

    crawl_state_manifest_writer = cast(
        CrawlStateManifestWriter,
        SimpleNamespace(),
    )
    focused_page = PageProcessorSettings(max_non_page_media_per_page=48)

    awaitable = runner._execute_crawl(  # type: ignore[attr-defined]
        crawl_attempt_id="attempt-1",
        crawl_state_manifest_writer=crawl_state_manifest_writer,
        page_settings=focused_page,
    )
    result = await awaitable

    assert result is not None
    assert received["settings"] is settings
    assert received["crawl_attempt_id"] == "attempt-1"
    assert (
        received["crawl_state_manifest_writer"] is crawl_state_manifest_writer
    )
    assert received["page_settings_override"] is focused_page
    assert "page_settings" not in received


def _make_options(tmp_path: Path):
    from orchestration.cli.argument_parser import RuntimeOptions

    return RuntimeOptions(
        command="run",
        project_root=tmp_path / "project-root",
        config_root=PROJECT_ROOT,
        environment="dev",
        profile="dev",
        fresh_run=False,
        resume=False,
    )
