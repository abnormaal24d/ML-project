"""Propagation contracts for the crawler page-settings override.

Proves: page_settings_override reaches exactly the PAGE handler while
image/audio/video/document keep their canonical processor settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from config.load import load_settings
from orchestration.composition.runtime import handlers as handlers_module
from orchestration.composition.runtime.handlers import build_task_processor
from orchestration.composition.runtime.handler_composition import registry as registry_module

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _canonical_settings():
    return load_settings(
        "dev",
        project_root=Path(PROJECT_ROOT) / "tmp-root",
        config_root=PROJECT_ROOT,
        environment="dev",
    )


def _build_with_override(
    monkeypatch: pytest.MonkeyPatch,
    settings,
    page_settings: Any,
) -> dict[str, dict[str, object]]:
    captured: dict[str, dict[str, object]] = []

    def record(name: str):
        def _build(**kwargs: object) -> object:
            captured.append({"handler": name, **kwargs})  # type: ignore[dict-item]
            return object()

        return _build

    monkeypatch.setattr(registry_module, "build_page_handler", record("page"))
    monkeypatch.setattr(registry_module, "build_feed_handler", record("feed"))
    monkeypatch.setattr(registry_module, "build_image_handler", record("image"))
    monkeypatch.setattr(registry_module, "build_audio_handler", record("audio"))
    monkeypatch.setattr(registry_module, "build_video_handler", record("video"))
    monkeypatch.setattr(registry_module, "build_document_handler", record("document"))

    from types import SimpleNamespace

    build_task_processor(
        settings=settings,
        page_settings=page_settings,
        logger_factory=SimpleNamespace(
            get_logger_for=lambda *_a, **_k: SimpleNamespace()
        ),
        id_generator=SimpleNamespace(),
        fetcher=SimpleNamespace(),
        coverage_tracker=SimpleNamespace(),
        scheduler=SimpleNamespace(),
        dataset_writer=SimpleNamespace(),
        url_filter=SimpleNamespace(),
        url_normalizer=SimpleNamespace(),
        host_normalizer=SimpleNamespace(),
        page_content_extractor=SimpleNamespace(),
        discovery_task_builder=SimpleNamespace(),
        network_access_guard=SimpleNamespace(),
        redirector=SimpleNamespace(),
    )
    return {entry["handler"]: entry for entry in captured}  # type: ignore[misc]


def test_override_reaches_only_the_page_handler(monkeypatch) -> None:
    settings = _canonical_settings()
    focused = settings.collection.processors.page.model_copy(
        update={"max_non_page_media_per_page": 48}
    )

    captured = _build_with_override(
        monkeypatch,
        settings=settings,
        page_settings=focused,
    )

    assert captured["page"]["page_settings"] is focused

    assert (
        captured["image"]["image_settings"]
        is settings.collection.processors.image
    )
    assert (
        captured["audio"]["audio_settings"]
        is settings.collection.processors.audio
    )
    assert (
        captured["video"]["video_settings"]
        is settings.collection.processors.video
    )
    assert (
        captured["document"]["document_settings"]
        is settings.collection.processors.document
    )


def test_canonical_page_settings_flow_without_override(monkeypatch) -> None:
    settings = _canonical_settings()

    captured = _build_with_override(
        monkeypatch,
        settings=settings,
        page_settings=settings.collection.processors.page,
    )

    assert (
        captured["page"]["page_settings"]
        is settings.collection.processors.page
    )
