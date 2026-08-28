"""Checkpoint contract CLI: mandatory production options and defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestration.cli.argument_parser import parse_runtime_options

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _base(tmp_path: Path, environment: str) -> tuple[str, ...]:
    return (
        "run",
        "--project-root",
        str(tmp_path),
        "--config-root",
        str(PROJECT_ROOT),
        "--environment",
        environment,
    )


def test_dev_run_defaults_to_permissive_contract(tmp_path: Path) -> None:
    options = parse_runtime_options(_base(tmp_path, "dev"))

    assert options.checkpoint_headers is False
    assert options.checkpoint_blob_storage is None
    assert options.staging_lock is None


def test_prod_run_requires_checkpoint_headers(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_runtime_options(_base(tmp_path, "prod"))

    assert exc_info.value.code == 2


def test_prod_run_requires_blob_storage(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_runtime_options(
            (
                *_base(tmp_path, "prod"),
                "--checkpoint-headers",
            )
        )

    assert exc_info.value.code == 2


def test_prod_run_requires_staging_lock(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_runtime_options(
            (
                *_base(tmp_path, "prod"),
                "--checkpoint-headers",
                "--checkpoint-blob-storage",
                str(tmp_path / "blobs"),
            )
        )

    assert exc_info.value.code == 2


def test_prod_run_accepts_full_checkpoint_contract(tmp_path: Path) -> None:
    blob_storage = tmp_path / "blobs"
    staging_lock = tmp_path / "staging.lock"
    options = parse_runtime_options(
        (
            *_base(tmp_path, "prod"),
            "--checkpoint-headers",
            "--checkpoint-blob-storage",
            str(blob_storage),
            "--staging-lock",
            str(staging_lock),
        )
    )

    assert options.checkpoint_headers is True
    assert options.checkpoint_blob_storage == blob_storage
    assert options.staging_lock == staging_lock


def test_prod_control_does_not_require_run_checkpoint_contract(
    tmp_path: Path,
) -> None:
    options = parse_runtime_options(
        (
            "control",
            "status",
            "--project-root",
            str(tmp_path),
            "--config-root",
            str(PROJECT_ROOT),
            "--environment",
            "prod",
        )
    )

    assert options.command == "control"
    assert options.profile == "prod"


def test_contract_required_for_prod(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_runtime_options(_base(tmp_path, "prod"))

    assert exc_info.value.code == 2


def test_invalid_environment_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_runtime_options(_base(tmp_path, "invalid-environment"))

    assert exc_info.value.code == 2
