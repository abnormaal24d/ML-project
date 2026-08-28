"""Logging contracts for crawler runtime composition."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.load import load_settings
from orchestration.runtime_dependency_preflight import OptionalDependencyReport
from orchestration.composition.runtime.crawler import (
    _log_runtime_composition_result,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _RecordingLogger:
    def __init__(self) -> None:
        self.infos: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.infos.append((event, fields))

    def warning(self, _event: str, **_fields: object) -> None:
        return None


@pytest.mark.parametrize(
    ("state_enabled", "dead_letter_setting", "expected_flags"),
    (
        (True, True, (True, True)),
        (False, True, (False, False)),
    ),
)
def test_runtime_composition_log_derives_state_flags_from_settings(
    tmp_path: Path,
    *,
    state_enabled: bool,
    dead_letter_setting: bool,
    expected_flags: tuple[bool, bool],
) -> None:
    settings = load_settings(
        "dev",
        project_root=tmp_path / "project-root",
        config_root=PROJECT_ROOT,
        environment="dev",
    )
    state_settings = settings.crawler.state.model_copy(
        update={
            "enabled": state_enabled,
            "dead_letter_enabled": dead_letter_setting,
        }
    )
    crawler_settings = settings.crawler.model_copy(
        update={"state": state_settings}
    )
    settings = settings.model_copy(update={"crawler": crawler_settings})
    logger = _RecordingLogger()

    _log_runtime_composition_result(
        settings=settings,
        logger=logger,  # type: ignore[arg-type]
        seed_tasks=(),
        dependency_report=OptionalDependencyReport(
            optional_dependency_status={
                "ffmpeg_available": True,
                "media_decoder_available": True,
            },
            summary="ok",
        ),
    )

    event, fields = logger.infos[0]
    assert event == "runtime_services_ready"
    assert (
        fields["state_persistence_enabled"],
        fields["dead_letter_persistence_enabled"],
    ) == expected_flags
