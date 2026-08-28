"""Smoke coverage for the ``python -m orchestration.main`` module entrypoint."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from orchestration.errors import SettingsLoadError
from orchestration.main import STARTUP_CONFIGURATION_EXIT_CODE, main

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_PRODUCTION_WHISPER_PIN_ENV = (
    "APP_OVERRIDE__preprocessing__transcription__model_name",
    "APP_OVERRIDE__preprocessing__transcription__model_revision",
    "APP_OVERRIDE__preprocessing__transcription__model_artifact_hash",
    "APP_OVERRIDE__preprocessing__transcription__backend_version",
)


def test_orchestration_main_module_entrypoint_resolves() -> None:
    assert importlib.util.find_spec("orchestration.main") is not None


def test_orchestration_main_module_entrypoint_runs_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "orchestration.main",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "command",
    (("run",), ("control", "validate-config")),
)
def test_cli_reports_missing_processing_activity_safely(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: tuple[str, ...],
) -> None:
    config_root = tmp_path / "config-root"
    config_files = config_root / "config" / "files"
    shutil.copytree(PROJECT_ROOT / "config" / "files", config_files)
    (config_files / "governance" / "processing_activities.json").unlink()

    result = main(
        [
            *command,
            "--project-root",
            str(tmp_path / "project-root"),
            "--config-root",
            str(config_root),
            "--environment",
            "dev",
        ]
    )

    assert result == STARTUP_CONFIGURATION_EXIT_CODE
    error_output = capsys.readouterr().err

    assert "Startup configuration error:" in error_output
    assert "processing_activity_registry" in error_output
    assert "governance.processing_activities_file" in error_output
    assert "processing_activities.json" in error_output
    assert "processing_activity_config_missing" in error_output
    assert str(tmp_path) not in error_output


def test_cli_reports_settings_load_errors_without_traceback_or_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestration import main as main_module

    leaked_path = tmp_path / "private" / "profiles" / "dev.toml"

    def fail_load(**_kwargs: object) -> object:
        cause = ValueError(f"invalid profile at {leaked_path}")
        raise SettingsLoadError(
            str(cause),
            stage="bootstrap",
            component="settings",
            cause=cause,
        )

    monkeypatch.setattr(main_module, "load", fail_load)

    result = main(
        [
            "control",
            "status",
            "--project-root",
            str(tmp_path / "project-root"),
            "--config-root",
            str(PROJECT_ROOT),
            "--environment",
            "dev",
        ]
    )

    assert result == STARTUP_CONFIGURATION_EXIT_CODE
    error_output = capsys.readouterr().err
    assert "Startup configuration error:" in error_output
    assert "settings_load_error" in error_output
    assert "Traceback" not in error_output
    assert str(tmp_path) not in error_output


def test_cli_reports_missing_production_whisper_pins_safely(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _PRODUCTION_WHISPER_PIN_ENV:
        monkeypatch.delenv(name, raising=False)

    result = main(
        [
            "control",
            "validate-config",
            "--project-root",
            str(tmp_path / "project-root"),
            "--config-root",
            str(PROJECT_ROOT),
            "--environment",
            "prod",
        ]
    )

    assert result == STARTUP_CONFIGURATION_EXIT_CODE
    error_output = capsys.readouterr().err
    assert "Startup configuration error:" in error_output
    assert "settings_load_error" in error_output
    assert "preprocessing.transcription" in error_output
    assert "required_deployment_pins_missing" in error_output
    assert "Traceback" not in error_output
    assert str(tmp_path) not in error_output


@pytest.mark.parametrize(
    "command",
    (("run",), ("control", "validate-config")),
)
def test_cli_reports_backend_configuration_errors_safely(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    command: tuple[str, ...],
) -> None:
    from config.load import load_settings
    from orchestration import main as main_module
    from orchestration import settings_loader
    from training.runtime import preparation

    settings = load_settings(
        "dev",
        project_root=tmp_path / "project-root",
        config_root=PROJECT_ROOT,
        environment="dev",
    )
    leaked_path = tmp_path / "private" / "backend.json"

    monkeypatch.setattr(
        main_module,
        "load",
        lambda **_kwargs: settings,
    )
    monkeypatch.setattr(settings_loader, "world_size", lambda: 1)

    def fail_backend(**_kwargs: object) -> None:
        raise ValueError(f"invalid backend artifact at {leaked_path}")

    monkeypatch.setattr(
        preparation,
        "prepare_training_backend",
        fail_backend,
    )

    result = main(
        [
            *command,
            "--project-root",
            str(tmp_path / "project-root"),
            "--config-root",
            str(PROJECT_ROOT),
            "--environment",
            "dev",
        ]
    )

    assert result == STARTUP_CONFIGURATION_EXIT_CODE
    error_output = capsys.readouterr().err
    assert "Startup configuration error:" in error_output
    assert "training_backend" in error_output
    assert "backend_configuration_error" in error_output
    assert "Traceback" not in error_output
    assert str(tmp_path) not in error_output


@pytest.mark.parametrize(
    "command",
    (("run",), ("control", "validate-config")),
)
def test_cli_reports_invalid_world_size_safely(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    command: tuple[str, ...],
) -> None:
    from config.load import load_settings
    from orchestration import main as main_module

    settings = load_settings(
        "dev",
        project_root=tmp_path / "project-root",
        config_root=PROJECT_ROOT,
        environment="dev",
    )
    monkeypatch.setattr(
        main_module,
        "load",
        lambda **_kwargs: settings,
    )
    monkeypatch.setenv("WORLD_SIZE", "not-an-integer")

    result = main(
        [
            *command,
            "--project-root",
            str(tmp_path / "project-root"),
            "--config-root",
            str(PROJECT_ROOT),
            "--environment",
            "dev",
        ]
    )

    assert result == STARTUP_CONFIGURATION_EXIT_CODE
    error_output = capsys.readouterr().err
    assert "Startup configuration error:" in error_output
    assert "training_backend" in error_output
    assert "backend_configuration_error" in error_output
    assert "Traceback" not in error_output
    assert str(tmp_path) not in error_output


def test_validate_config_reports_missing_local_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.load import load_settings
    from orchestration import main as main_module

    settings = load_settings(
        "dev",
        project_root=tmp_path / "project-root",
        config_root=PROJECT_ROOT,
        environment="dev",
    )
    missing_model = tmp_path / "private" / "missing-whisper"
    transcription = settings.preprocessing.transcription.model_copy(
        update={
            "local_files_only": True,
            "model_name": str(missing_model),
        }
    )
    preprocessing = settings.preprocessing.model_copy(
        update={"transcription": transcription}
    )
    settings = settings.model_copy(update={"preprocessing": preprocessing})
    monkeypatch.setattr(main_module, "load", lambda **_kwargs: settings)

    result = main(
        [
            "control",
            "validate-config",
            "--project-root",
            str(tmp_path / "project-root"),
            "--config-root",
            str(PROJECT_ROOT),
            "--environment",
            "dev",
        ]
    )

    assert result == STARTUP_CONFIGURATION_EXIT_CODE
    error_output = capsys.readouterr().err
    assert "preprocessing.transcription.model_name" in error_output
    assert "missing-whisper" in error_output
    assert "required_local_artifact_missing" in error_output
    assert str(tmp_path) not in error_output


def test_validate_config_rejects_tampered_local_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.load import load_settings
    from orchestration import main as main_module
    from preprocessing.media.adapters.whisper_model_loader import (
        installed_backend_version,
    )

    settings = load_settings(
        "dev",
        project_root=tmp_path / "project-root",
        config_root=PROJECT_ROOT,
        environment="dev",
    )
    model_directory = tmp_path / "private" / "whisper"
    model_directory.mkdir(parents=True)
    (model_directory / "model.bin").write_bytes(b"tampered")
    transcription = settings.preprocessing.transcription.model_copy(
        update={
            "local_files_only": True,
            "production_mode": True,
            "model_name": str(model_directory),
            "model_revision": "immutable-test-revision",
            "model_artifact_hash": "0" * 64,
            "backend_version": installed_backend_version(),
        }
    )
    preprocessing = settings.preprocessing.model_copy(
        update={"transcription": transcription}
    )
    settings = settings.model_copy(update={"preprocessing": preprocessing})
    monkeypatch.setattr(main_module, "load", lambda **_kwargs: settings)

    result = main(
        [
            "control",
            "validate-config",
            "--project-root",
            str(tmp_path / "project-root"),
            "--config-root",
            str(PROJECT_ROOT),
            "--environment",
            "dev",
        ]
    )

    assert result == STARTUP_CONFIGURATION_EXIT_CODE
    error_output = capsys.readouterr().err
    assert "preprocessing.transcription.model_artifact_hash" in error_output
    assert "model.bin" in error_output
    assert "artifact_hash_mismatch" in error_output
    assert str(tmp_path) not in error_output
