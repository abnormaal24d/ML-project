from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from config.multimodal.model_settings import ModelSettings
from config.multimodal.training_settings import TrainingSettings
from evaluator.leakage.report import generate_report
from release.release_artifact_validation import check_model_card
from release.release_evidence_bundle import (
    EvaluationArtifacts,
    ReleaseEvidenceBundle,
)
from release.release_evidence_validation import (
    _artifact_violations,
    check_release,
)
from release.serving_artifacts import (
    SERVING_FORMATS,
    ServingArtifactPolicy,
    check_serving_artifacts,
    inspect_serving_artifacts,
)
from tests.support.release_fixtures import (
    fixture_policy,
    write_valid_reproducibility_report,
)
from training.export import export as export_module

_GENERATED_AT = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)


class MultimodalModel(nn.Module):
    """Small stand-in that preserves the production model-class contract."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)


def _disable_binary_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "_export_torchscript",
        "_export_onnx",
        "_export_safetensors",
    ):
        monkeypatch.setattr(export_module, name, lambda **_kwargs: {})


def _write_valid_safetensors(directory: Path) -> None:
    artifact = directory / "model.safetensors"
    save_file({"weight": torch.ones(2, 2)}, str(artifact))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (directory / "safetensors_export_status.json").write_text(
        json.dumps({"status": "ok", "sha256": digest}) + "\n",
        encoding="utf-8",
    )


def _build_production_evidence(
    *,
    candidate_directory: Path,
    dataset_root: Path,
) -> ReleaseEvidenceBundle:
    checkpoint_path = candidate_directory / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checksum-backed-checkpoint")
    metrics_path = candidate_directory / "training_metrics.json"
    metrics_path.write_text('{"train_loss": 0.5}\n', encoding="utf-8")
    dataset_card_path = dataset_root / "dataset_card.json"
    dataset_card_path.write_text(
        '{"name": "validated production dataset"}\n',
        encoding="utf-8",
    )
    leakage_path = dataset_root / "evaluation" / "leakage_report.json"
    generate_report(
        left_records=(),
        right_records=(),
        output_path=leakage_path,
        minimum_coverage={},
    )
    reproducibility_path = (
        candidate_directory / "evaluation" / "reproducibility_report.json"
    )
    reproducibility_path.parent.mkdir(parents=True)
    write_valid_reproducibility_report(
        reproducibility_path,
        policy=fixture_policy(),
    )

    artifacts = EvaluationArtifacts.from_release_outputs(
        candidate_directory=candidate_directory,
        dataset_directory=dataset_root,
        checkpoint=checkpoint_path,
        metrics=metrics_path,
    )
    return ReleaseEvidenceBundle.build(
        mode="production_model",
        artifacts=artifacts,
        release_requirements_id=fixture_policy().policy_id,
        release_requirements_sha256=fixture_policy().policy_sha256,
    )


def test_production_export_without_serving_artifact_fails_release_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_binary_exports(monkeypatch)
    tokenizer_sha256 = "a" * 64
    candidate_directory = tmp_path / "candidate"
    dataset_root = tmp_path / "dataset-snapshot"
    dataset_root.mkdir()

    export_module.export_model(
        model=MultimodalModel(),
        export_directory=candidate_directory,
        model_settings=ModelSettings(),
        training_settings=TrainingSettings(
            release_stage="production_model",
            training_backend="dense_transformer",
            text_tokenizer_sha256=tokenizer_sha256,
        ),
        dataset_root=dataset_root,
        generated_at=_GENERATED_AT,
    )

    model_card_path = candidate_directory / "model_card.md"
    inference_config_path = candidate_directory / "inference_config.json"
    inference_config = json.loads(
        inference_config_path.read_text(encoding="utf-8")
    )

    assert check_model_card(path=model_card_path) == ()
    assert "2026-08-26T09:00:00Z" in model_card_path.read_text(
        encoding="utf-8"
    )
    assert inference_config["tokenizer_sha256"] == tokenizer_sha256
    assert inference_config["training_backend"] == "dense_transformer"
    assert (
        _artifact_violations(
            path=inference_config_path,
            relative="inference_config.json",
            name="inference_config",
            mode="production_model",
        )
        == []
    )

    evidence = _build_production_evidence(
        candidate_directory=candidate_directory,
        dataset_root=dataset_root,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=fixture_policy(),
    )
    assert not gate.passed
    assert any(
        violation.startswith("serving_artifact_missing")
        for violation in gate.violations
    )


def test_production_export_with_serving_artifact_passes_release_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_binary_exports(monkeypatch)
    candidate_directory = tmp_path / "candidate"
    dataset_root = tmp_path / "dataset-snapshot"
    dataset_root.mkdir()

    export_module.export_model(
        model=MultimodalModel(),
        export_directory=candidate_directory,
        model_settings=ModelSettings(),
        training_settings=TrainingSettings(
            release_stage="production_model",
            training_backend="dense_transformer",
            text_tokenizer_sha256="a" * 64,
        ),
        dataset_root=dataset_root,
        generated_at=_GENERATED_AT,
    )

    # A candidate that cannot export a binary format is not serveable, but a
    # present safetensors artifact satisfies the deployment contract.
    _write_valid_safetensors(candidate_directory)

    evidence = _build_production_evidence(
        candidate_directory=candidate_directory,
        dataset_root=dataset_root,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=fixture_policy(),
    )
    assert gate.passed
    assert gate.violations == ()


def test_skipped_serving_status_fails_release_gate(tmp_path: Path) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()

    # An explicit skipped status is a gate failure even when a stale artifact
    # is left on disk.
    (export_directory / "model.safetensors").write_bytes(b"stale")
    (export_directory / "safetensors_export_status.json").write_text(
        json.dumps({"status": "skipped", "reason": "trace failed"}) + "\n",
        encoding="utf-8",
    )

    policy = ServingArtifactPolicy(
        required_any_of=frozenset(SERVING_FORMATS),
    )
    violations = check_serving_artifacts(
        export_directory=export_directory,
        policy=policy,
    )

    assert any(
        violation.startswith("serving_export_skipped")
        for violation in violations
    )


def test_serving_inspection_classifies_formats(tmp_path: Path) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()

    _write_valid_safetensors(export_directory)
    (export_directory / "onnx_export_status.json").write_text(
        json.dumps({"status": "skipped", "reason": "onnx unavailable"}) + "\n",
        encoding="utf-8",
    )

    inspection = inspect_serving_artifacts(
        export_directory=export_directory,
    )

    assert inspection.status_by_format["safetensors"] == "ok"
    assert inspection.status_by_format["torchscript"] == "missing"
    assert inspection.status_by_format["onnx"] == "skipped"
    assert inspection.available == frozenset({"safetensors"})
    assert inspection.skipped == frozenset({"onnx"})
    assert inspection.missing == frozenset({"torchscript"})


def test_packaged_model_card_template_is_canonical() -> None:
    project_root = Path(__file__).resolve().parents[2]
    canonical_path = (
        project_root / "training" / "export" / "model_card_template.md"
    )

    loaded_template = export_module._load_model_card_template()
    assert loaded_template == canonical_path.read_text(encoding="utf-8")


def test_export_rejects_missing_tokenizer_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_binary_exports(monkeypatch)

    with pytest.raises(
        ValueError,
        match="text_tokenizer_sha256 is required before model export",
    ):
        export_module.export_model(
            model=MultimodalModel(),
            export_directory=tmp_path / "candidate",
            model_settings=ModelSettings(),
            training_settings=TrainingSettings(release_stage="candidate"),
            dataset_root=tmp_path / "dataset-snapshot",
            generated_at=_GENERATED_AT,
        )
