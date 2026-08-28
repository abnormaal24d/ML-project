"""Contract tests for the installed and module CLI entrypoints.

Subprocess cases exercise both the installed console script and the module
entrypoint. Unit cases exercise argument resolution through
``parse_runtime_options``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from orchestration.cli.argument_parser import (
    RuntimeOptions,
    parse_runtime_options,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_SELECTOR_ENV_VARS = (
    "DATA_ENGINE_PROJECT_ROOT",
    "DATA_ENGINE_CONFIG_ROOT",
    "DATA_ENGINE_ENVIRONMENT",
    "APP_ENV",
)

_CLI_TIMEOUT_SECONDS = 30


def _run_cli_subprocess(
    *arguments: str,
    timeout_seconds: int = _CLI_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run the canonical CLI in a fresh subprocess with a hard timeout."""

    env = {
        name: value
        for name, value in os.environ.items()
        if name not in _SELECTOR_ENV_VARS
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "orchestration.main",
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
        env=env,
    )


def _installed_console_script() -> Path:
    """Return this interpreter's installed console script without using PATH."""

    scripts_directory = Path(sysconfig.get_path("scripts"))
    candidates = (
        scripts_directory / "multimodal-crawler",
        scripts_directory / "multimodal-crawler.exe",
    )
    console_script = next(
        (candidate for candidate in candidates if candidate.is_file()),
        None,
    )
    assert console_script is not None, (
        "multimodal-crawler console script is not installed for "
        f"{sys.executable}"
    )
    return console_script


def test_installed_console_script_runs_help() -> None:
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in _SELECTOR_ENV_VARS
    }
    result = subprocess.run(
        [_installed_console_script(), "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=_CLI_TIMEOUT_SECONDS,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "multimodal-crawler" in result.stdout


def test_cli_help_exits_zero() -> None:
    result = _run_cli_subprocess("--help")

    assert result.returncode == 0, result.stderr
    assert "multimodal-crawler" in result.stdout


def test_cli_unknown_command_exits_nonzero() -> None:
    result = _run_cli_subprocess("frobnicate")

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_cli_run_without_environment_exits_nonzero(tmp_path: Path) -> None:
    config_root = tmp_path / "config-root"
    (config_root / "config" / "files").mkdir(parents=True)

    result = _run_cli_subprocess(
        "run",
        "--project-root",
        str(tmp_path / "project-root"),
        "--config-root",
        str(config_root),
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "environment must be explicitly provided" in result.stderr


def test_cli_nonexistent_config_root_exits_nonzero(tmp_path: Path) -> None:
    result = _run_cli_subprocess(
        "run",
        "--project-root",
        str(tmp_path / "project-root"),
        "--config-root",
        str(tmp_path / "does-not-exist"),
        "--environment",
        "dev",
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "configuration root must contain config/files" in result.stderr
    assert str(tmp_path) not in result.stderr


def test_cli_control_without_action_exits_nonzero(tmp_path: Path) -> None:
    config_root = tmp_path / "config-root"
    (config_root / "config" / "files").mkdir(parents=True)

    result = _run_cli_subprocess(
        "control",
        "--project-root",
        str(tmp_path / "project-root"),
        "--config-root",
        str(config_root),
        "--environment",
        "dev",
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "required" in result.stderr


def _parse(arguments: tuple[str, ...], tmp_path: Path) -> RuntimeOptions:
    return parse_runtime_options(
        (
            *arguments,
            "--project-root",
            str(tmp_path),
            "--config-root",
            str(PROJECT_ROOT),
            "--environment",
            "dev",
        )
    )


def test_parse_runtime_options_environment_is_selected(tmp_path: Path) -> None:
    options = _parse(("run",), tmp_path)
    assert options.environment == "dev"


def test_parse_runtime_options_rejects_invalid_environment(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_runtime_options(
            (
                "run",
                "--project-root",
                str(tmp_path),
                "--config-root",
                str(PROJECT_ROOT),
                "--environment",
                "invalid-environment",
                "--checkpoint-headers",
                "--checkpoint-blob-storage",
                str(tmp_path / "blob-storage"),
                "--staging-lock",
                str(tmp_path / "staging.lock"),
            )
        )

    assert exc_info.value.code == 2


def test_parse_runtime_options_prod_is_selected(tmp_path: Path) -> None:
    options = parse_runtime_options(
        (
            "run",
            "--project-root",
            str(tmp_path),
            "--config-root",
            str(PROJECT_ROOT),
            "--environment",
            "prod",
            "--checkpoint-headers",
            "--checkpoint-blob-storage",
            str(tmp_path / "blob-storage"),
            "--staging-lock",
            str(tmp_path / "staging.lock"),
        )
    )
    assert options.environment == "prod"
    assert options.checkpoint_headers is True
    assert options.checkpoint_blob_storage == tmp_path / "blob-storage"
    assert options.staging_lock == tmp_path / "staging.lock"


def test_parse_runtime_options_use_cuda_default_is_false(
    tmp_path: Path,
) -> None:
    options = _parse(("run",), tmp_path)
    assert options.use_cuda is False


def test_parse_runtime_options_use_cuda_flag_is_parsed(tmp_path: Path) -> None:
    options = _parse(("run", "--use-cuda"), tmp_path)
    assert options.use_cuda is True


def test_parse_runtime_options_fresh_run_and_resume_defaults(
    tmp_path: Path,
) -> None:
    options = _parse(("run",), tmp_path)
    assert options.fresh_run is False
    assert options.resume is True


def test_parse_runtime_options_fresh_run_flag_is_parsed(
    tmp_path: Path,
) -> None:
    options = _parse(("run", "--fresh-run"), tmp_path)
    assert options.fresh_run is True
    assert options.resume is True


def test_parse_runtime_options_no_resume_is_parsed(tmp_path: Path) -> None:
    options = _parse(("run", "--no-resume"), tmp_path)
    assert options.resume is False


@pytest.mark.parametrize(
    ("arguments", "control_action"),
    (
        (("control", "pause"), "pause"),
        (("control", "resume"), "resume"),
        (("control", "stop"), "stop"),
        (("control", "status"), "status"),
        (("control", "validate-config"), "validate-config"),
    ),
)
def test_parse_runtime_options_control_actions(
    tmp_path: Path,
    arguments: tuple[str, ...],
    control_action: str,
) -> None:
    options = _parse(arguments, tmp_path)
    assert options.command == "control"
    assert options.control_action == control_action


def test_parse_runtime_options_requires_an_environment(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_runtime_options(
            (
                "run",
                "--project-root",
                str(tmp_path),
                "--config-root",
                str(PROJECT_ROOT),
            )
        )

    assert exc_info.value.code == 2


def test_parse_runtime_options_rejects_unknown_environment(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_runtime_options(
            (
                "run",
                "--project-root",
                str(tmp_path),
                "--config-root",
                str(PROJECT_ROOT),
                "--environment",
                "not-an-environment",
            )
        )

    assert exc_info.value.code == 2


def test_parse_runtime_options_rejects_missing_config_tree(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_runtime_options(
            (
                "run",
                "--project-root",
                str(tmp_path),
                "--config-root",
                str(tmp_path / "missing-config"),
                "--environment",
                "dev",
            )
        )

    assert exc_info.value.code == 2


def test_parse_runtime_options_rejects_incompatible_profile_override(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_runtime_options(
            (
                "run",
                "--project-root",
                str(tmp_path),
                "--config-root",
                str(PROJECT_ROOT),
                "--environment",
                "dev",
                "--profile",
                "test",
            )
        )

    assert exc_info.value.code == 2
