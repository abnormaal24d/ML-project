from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.load import load_settings
from config.path_resolution.project_paths import ProjectPaths
from crawler.runtime.control.crawler_control_directory import (
    CrawlerControlDirectory,
)
from orchestration.bootstrap.application import FAILURE_EXIT_CODE
from orchestration.cli.argument_parser import parse_runtime_options
from orchestration.main import main

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("arguments", "command", "control_action"),
    (
        (("run", "--fresh-run", "--use-cuda"), "run", None),
        (("control", "status"), "control", "status"),
        (("control", "validate-config"), "control", "validate-config"),
    ),
)
def test_runtime_options_resolve_every_command(
    tmp_path: Path,
    arguments: tuple[str, ...],
    command: str,
    control_action: str | None,
) -> None:
    options = parse_runtime_options(
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

    assert options.command == command
    assert options.control_action == control_action
    assert options.project_root == tmp_path.resolve()
    assert options.config_root == PROJECT_ROOT


@pytest.mark.parametrize("removed_command", ("ingest", "train"))
def test_removed_phase_commands_are_rejected(
    tmp_path: Path,
    removed_command: str,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_runtime_options(
            (
                removed_command,
                "--project-root",
                str(tmp_path),
                "--config-root",
                str(PROJECT_ROOT),
                "--environment",
                "dev",
            )
        )

    assert exc_info.value.code == 2


def test_control_lifecycle_runs_through_canonical_cli(tmp_path: Path) -> None:
    common = [
        "--project-root",
        str(tmp_path),
        "--config-root",
        str(PROJECT_ROOT),
        "--environment",
        "dev",
    ]

    assert main(["control", "pause", *common]) == 0
    assert main(["control", "status", *common]) == 0
    assert main(["control", "resume", *common]) == 0
    assert main(["control", "stop", *common]) == 0


def test_validate_config_is_side_effect_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"

    result = main(
        [
            "control",
            "validate-config",
            "--project-root",
            str(workspace),
            "--config-root",
            str(PROJECT_ROOT),
            "--environment",
            "dev",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "Configuration valid\n"
    assert captured.err == ""
    assert not workspace.exists()


def test_control_resume_clears_pause_and_unconsumed_stop(
    tmp_path: Path,
) -> None:
    common = [
        "--project-root",
        str(tmp_path),
        "--config-root",
        str(PROJECT_ROOT),
        "--environment",
        "dev",
    ]
    settings = load_settings(
        "dev",
        project_root=tmp_path,
        config_root=PROJECT_ROOT,
        environment="dev",
    )
    control_directory = CrawlerControlDirectory(
        settings=settings.crawler,
        project_root=tmp_path,
    )

    assert main(["control", "pause", *common]) == 0
    assert main(["control", "stop", *common]) == 0
    assert control_directory.should_pause() is True
    assert control_directory.should_stop() is True

    assert main(["control", "resume", *common]) == 0

    assert control_directory.should_pause() is False
    assert control_directory.should_stop() is False


def test_corrupt_resume_state_is_reported_as_cli_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = load_settings(
        "dev",
        project_root=tmp_path,
        config_root=PROJECT_ROOT,
        environment="dev",
    )
    state_path = ProjectPaths(project_root=tmp_path).resolve(
        Path(settings.datasets.paths.workflow_artifacts_directory)
        / settings.collection.datachecker.crawl_state_manifest_filename
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "status": "running",
                "workflow_id": 123,
                "generation_id": "generation-1",
            }
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "run",
            "--project-root",
            str(tmp_path),
            "--config-root",
            str(PROJECT_ROOT),
            "--environment",
            "dev",
        ]
    )

    assert result == int(FAILURE_EXIT_CODE)
    error_output = capsys.readouterr().err
    assert "Application failed:" in error_output
    assert (
        "manifest field workflow_id must be a non-empty string" in error_output
    )
