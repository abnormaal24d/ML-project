"""Unit tests for the production serving-artifact gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from release.serving_artifacts import (
    SERVING_FORMATS,
    ServingArtifactPolicy,
    check_serving_artifacts,
    inspect_serving_artifacts,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_safetensors(directory: Path) -> None:
    artifact = directory / "model.safetensors"
    save_file({"weight": torch.ones(2, 2)}, str(artifact))
    (directory / "safetensors_export_status.json").write_text(
        json.dumps({"status": "ok", "sha256": _sha256(artifact)}) + "\n",
        encoding="utf-8",
    )


def _write_valid_torchscript(directory: Path) -> None:
    artifact = directory / "model.torchscript.pt"
    traced = torch.jit.trace(torch.nn.Identity(), (torch.ones(1, 2),))
    traced.save(str(artifact))
    (directory / "torchscript_export_status.json").write_text(
        json.dumps({"status": "ok", "sha256": _sha256(artifact)}) + "\n",
        encoding="utf-8",
    )


def test_policy_rejects_unknown_formats() -> None:
    with pytest.raises(ValueError, match="unknown serving formats"):
        ServingArtifactPolicy(required_any_of=frozenset({"not_a_format"}))


def test_default_policy_requires_one_of_three_for_candidate_and_production() -> (
    None
):
    from release.serving_artifacts import default_serving_policy

    policy = default_serving_policy(mode="production_model")
    assert policy.required_any_of == frozenset(SERVING_FORMATS)
    assert policy.required_all_of == frozenset()
    assert default_serving_policy(mode="candidate") == policy


def test_check_passes_when_at_least_one_format_is_valid(
    tmp_path: Path,
) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    _write_valid_safetensors(export_directory)

    assert (
        check_serving_artifacts(
            export_directory=export_directory,
            policy=ServingArtifactPolicy(
                required_any_of=frozenset(SERVING_FORMATS)
            ),
        )
        == ()
    )


@pytest.mark.parametrize("status", ["skipped", "failed", "error"])
def test_check_fails_when_required_format_is_skipped(
    tmp_path: Path,
    status: str,
) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    (export_directory / "model.safetensors").write_bytes(b"stale")
    (export_directory / "safetensors_export_status.json").write_text(
        json.dumps({"status": status, "reason": "export failed"}) + "\n",
        encoding="utf-8",
    )

    violations = check_serving_artifacts(
        export_directory=export_directory,
        policy=ServingArtifactPolicy(
            required_any_of=frozenset(SERVING_FORMATS)
        ),
    )
    assert "serving_export_skipped:safetensors" in violations


def test_check_fails_when_all_required_formats_are_missing(
    tmp_path: Path,
) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()

    violations = check_serving_artifacts(
        export_directory=export_directory,
        policy=ServingArtifactPolicy(
            required_any_of=frozenset(SERVING_FORMATS)
        ),
    )
    assert any(
        item.startswith("serving_artifact_missing") for item in violations
    )


def test_required_all_of_reports_missing_and_invalid_formats(
    tmp_path: Path,
) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    (export_directory / "model.onnx").write_bytes(b"not-onnx")

    violations = check_serving_artifacts(
        export_directory=export_directory,
        policy=ServingArtifactPolicy(
            required_all_of=frozenset({"safetensors", "torchscript", "onnx"})
        ),
    )
    assert "serving_artifact_missing:safetensors" in violations
    assert "serving_artifact_missing:torchscript" in violations
    assert "invalid_serving_artifact:onnx" in violations


def test_inspection_marks_corrupt_status_file_invalid(tmp_path: Path) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    (export_directory / "model.onnx").write_bytes(b"model")
    (export_directory / "onnx_export_status.json").write_text(
        "not json", encoding="utf-8"
    )

    inspection = inspect_serving_artifacts(export_directory=export_directory)
    assert inspection.status_by_format["onnx"] == "invalid"
    assert inspection.invalid == frozenset({"onnx"})


def test_artifact_without_receipt_is_invalid(tmp_path: Path) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    artifact = export_directory / "model.safetensors"
    save_file({"weight": torch.ones(1)}, str(artifact))

    inspection = inspect_serving_artifacts(export_directory=export_directory)
    assert inspection.status_by_format["safetensors"] == "invalid"
    assert inspection.available == frozenset()


def test_digest_mismatch_is_invalid(tmp_path: Path) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    _write_valid_safetensors(export_directory)
    (export_directory / "model.safetensors").write_bytes(b"tampered")

    inspection = inspect_serving_artifacts(export_directory=export_directory)
    assert inspection.status_by_format["safetensors"] == "invalid"


def test_native_format_loader_is_required(tmp_path: Path) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    artifact = export_directory / "model.safetensors"
    artifact.write_bytes(b"not-safetensors")
    (export_directory / "safetensors_export_status.json").write_text(
        json.dumps({"status": "ok", "sha256": _sha256(artifact)}) + "\n",
        encoding="utf-8",
    )

    inspection = inspect_serving_artifacts(export_directory=export_directory)
    assert inspection.status_by_format["safetensors"] == "invalid"


def test_valid_torchscript_is_available(tmp_path: Path) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    _write_valid_torchscript(export_directory)

    inspection = inspect_serving_artifacts(export_directory=export_directory)
    assert inspection.available == frozenset({"torchscript"})
