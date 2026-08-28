"""Preflight report tests for the new configuration surface."""

from __future__ import annotations

import os
from pathlib import Path

from config.preflight import IssueKind, PreflightResult, preflight

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_WHISPER_PINS = {
    "preprocessing.transcription.model_name": "/tmp/mmcrawler-test-whisper",
    "preprocessing.transcription.model_revision": "test-only-revision",
    "preprocessing.transcription.model_artifact_hash": "0" * 64,
    "preprocessing.transcription.backend_version": "1.1.1",
}


def _prod_env(pins: bool = True) -> dict[str, str]:
    env = dict(os.environ)
    env["DATA_ENGINE_PROJECT_ROOT"] = str(PROJECT_ROOT)
    if pins:
        for path, value in _WHISPER_PINS.items():
            env[f"APP_OVERRIDE__{path.replace('.', '__')}"] = value
    return env


def _issues(result: PreflightResult) -> dict[IssueKind, list[str]]:
    grouped: dict[IssueKind, list[str]] = {}
    for issue in result.issues:
        grouped.setdefault(issue.kind, []).append(issue.message)
    return grouped


def test_dev_preflight_is_ok() -> None:
    result = preflight("dev", project_root=str(PROJECT_ROOT))
    assert result.ok
    assert result.settings is not None


def test_test_preflight_is_ok() -> None:
    result = preflight("test", project_root=str(PROJECT_ROOT))
    assert result.ok


def test_prod_preflight_succeeds_with_pins() -> None:
    result = preflight("prod", env=_prod_env())
    assert result.ok
    assert result.settings is not None
    assert len(result.settings.release.tasks) == 14


def test_prod_without_pins_returns_error_issue() -> None:
    result = preflight("prod", env=_prod_env(pins=False))
    assert not result.ok
    assert result.settings is None
    messages = [issue.message for issue in result.issues]
    assert any(
        "production Whisper transcription" in message for message in messages
    )


def test_prod_without_root_returns_error_issue() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "DATA_ENGINE_PROJECT_ROOT"
    }
    result = preflight("prod", env=env)
    assert not result.ok
    assert any(
        "requires an explicit project root" in issue.message
        for issue in result.issues
    )


def test_load_errors_are_reported_not_raised() -> None:
    result = preflight("dev", overrides=["training.num_workers=many"])
    assert not result.ok
    assert result.settings is None


def test_missing_dirs_are_warnings() -> None:
    result = preflight("test", project_root=str(PROJECT_ROOT))
    grouped = _issues(result)
    assert all(
        "does not exist yet" in message
        for message in grouped.get(IssueKind.WARNING, ())
        if "does not exist yet" in message
    )


def test_dev_reports_cwd_fallback_warning() -> None:
    result = preflight("dev", env={})
    grouped = _issues(result)
    assert any(
        "current working directory" in message
        for message in grouped.get(IssueKind.WARNING, ())
    )


def test_baseline_limits_are_warned() -> None:
    result = preflight("prod", env=_prod_env())
    grouped = _issues(result)
    assert any(
        "initial baseline" in message
        for message in grouped.get(IssueKind.WARNING, ())
    )


def test_artifact_path_warnings_are_advisory_only(tmp_path: Path) -> None:
    missing_whisper = tmp_path / "missing_whisper"
    missing_ocr = tmp_path / "missing.traineddata"
    env = _prod_env()
    env["APP_OVERRIDE__preprocessing__transcription__model_name"] = str(
        missing_whisper
    )
    env["APP_OVERRIDE__preprocessing__ocr__model_artifact_path"] = str(
        missing_ocr
    )
    result = preflight(
        "prod",
        env=env,
        check_artifact_paths=True,
    )
    assert result.ok
    grouped = _issues(result)
    assert any(
        "pinned Whisper model directory not found" in message
        for message in grouped.get(IssueKind.WARNING, ())
    )
    assert any(
        "pinned OCR model artifact not found" in message
        for message in grouped.get(IssueKind.WARNING, ())
    )


def test_preflight_defaults_to_dev() -> None:
    result = preflight(env={})
    assert result.profile == "dev"
    assert result.ok
