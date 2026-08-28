from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from evaluator.leakage.report import generate_report
from release.current_release_pointer import resolve_current_release
from release.release_decision import ReleaseStatus
from release.release_directory_validation import _validate_release_directory
from release.release_manifest import (
    _manifest_payload,
    _reproducibility_requirements_from_manifest,
    _validate_manifest,
)
from release.release_utilities import (
    CURRENT_POINTER,
    RELEASES_DIRECTORY,
    ProductionPromotionValidationError,
    atomic_write_json,
)
from tests.support.release_fixtures import (
    fixture_policy,
    write_valid_model_card,
    write_valid_reproducibility_report,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: bytes | str = b"artifact") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        content = content.encode("utf-8")
    path.write_bytes(content)
    return _sha256(path)


def _valid_release_directory(tmp_path: Path) -> Path:
    release = tmp_path / "releases" / "release-abc"
    dataset = release / "dataset"
    for directory in (release, dataset):
        directory.mkdir(parents=True)

    from release.release_evidence_bundle import (
        EvaluationArtifacts,
        ReleaseEvidenceBundle,
    )
    from release.release_staging import _copy_checkpoint
    from training.runtime.checkpoint.io import atomic_torch_save

    source_checkpoint = tmp_path / "source" / "checkpoint.pt"
    atomic_torch_save(
        payload={"marker": b"checkpoint", "weight": torch.tensor([1.0])},
        checkpoint_path=source_checkpoint,
    )
    checkpoint = release / "checkpoint.pt"
    _copy_checkpoint(source=source_checkpoint, target=checkpoint)
    _write(
        release / "training_metrics.json",
        '{"train_loss": 0.5}\n',
    )
    _write(
        release / "model_card.md",
    )
    write_valid_model_card(release / "model_card.md")
    _write(dataset / "dataset_card.json", '{"name": "dataset"}\n')
    _write(
        release / "evaluation" / "acceptance_report.json",
        json.dumps(
            {
                "passed": True,
                "release_stage": "production_model",
                "expected_status": ReleaseStatus.MODEL_ACCEPTED.value,
            }
        )
        + "\n",
    )
    leakage = dataset / "evaluation" / "leakage_report.json"
    generate_report(
        left_records=(),
        right_records=(),
        output_path=leakage,
        minimum_coverage={},
    )
    write_valid_reproducibility_report(
        release / "evaluation" / "reproducibility_report.json",
        policy=fixture_policy(),
    )
    safetensors_path = release / "model.safetensors"
    save_file(
        {"weight": torch.tensor([1.0])},
        str(safetensors_path),
    )
    _write(
        release / "safetensors_export_status.json",
        json.dumps({"status": "ok", "sha256": _sha256(safetensors_path)})
        + "\n",
    )
    _write(
        release / "inference_config.json",
        json.dumps({"training_backend": "pipeline_smoke"}) + "\n",
    )

    artifacts = EvaluationArtifacts.from_release_outputs(
        candidate_directory=release,
        dataset_directory=dataset,
        checkpoint=checkpoint,
        metrics=release / "training_metrics.json",
    )
    evidence = ReleaseEvidenceBundle.build(
        mode="production_model",
        artifacts=artifacts,
        release_requirements_id=fixture_policy().policy_id,
        release_requirements_sha256=fixture_policy().policy_sha256,
    )
    evidence_path = release / "evaluation" / "release_evidence_bundle.json"
    evidence.write(evidence_path)
    payload = _manifest_payload(
        release_id="release-abc",
        evidence=evidence,
        evidence_bundle_path=evidence_path,
        acceptance_report_path=(
            release / "evaluation" / "acceptance_report.json"
        ),
        release_directory=release,
        reproducibility_requirements=fixture_policy(),
    )
    atomic_write_json(
        path=release / "release_manifest.json",
        payload=payload,
    )
    return release


def _policy_payload() -> dict[str, object]:
    return {
        "policy_id": "production_v1",
        "seeds": [42],
        "require_deterministic_execution": True,
        "metric_tolerances": {"validation_loss": 0.03, "test_loss": 0.03},
    }


def test_manifest_validation_passes_for_complete_release(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    _validate_manifest(
        release_directory=release,
        expected_release_id="release-abc",
    )


def test_manifest_validation_rejects_missing_fields(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    manifest_path = release / "release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["artifacts"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="manifest fields are invalid",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )


def test_manifest_validation_rejects_schema_and_identity(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    manifest_path = release / "release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "0.1"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="manifest schema is invalid",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )
    payload["schema_version"] = "production_release.v1"
    payload["release_id"] = "release-other"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="release identifier mismatch",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )


def test_manifest_validation_rejects_non_production_mode(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    manifest_path = release / "release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["release_mode"] = "candidate"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="release mode is invalid",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )


def test_manifest_validation_rejects_policy_identity_and_digest(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    manifest_path = release / "release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["release_requirements_id"] = ""
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="lacks release policy identity",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )
    payload["release_requirements_id"] = "production_v1"
    payload["release_requirements_sha256"] = "not-a-digest"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="must be SHA-256",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )
    payload["release_requirements_sha256"] = "0" * 64
    payload["reproducibility_policy"] = None
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="lacks reproducibility policy",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )


def test_manifest_validation_rejects_policy_digest_mismatch(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    manifest_path = release / "release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["release_requirements_sha256"] = "1" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="policy digest mismatch",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )


def test_manifest_validation_rejects_bad_artifacts(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    manifest_path = release / "release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"] = "not-a-list"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="artifacts are invalid",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )

    payload["artifacts"] = ["not-a-dict"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="artifact entry is invalid",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )

    payload["artifacts"] = [{"bad": "entry"}]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="artifact fields are invalid",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )


def test_manifest_validation_rejects_duplicate_and_missing_artifacts(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    manifest_path = release / "release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = list(payload["artifacts"])
    artifacts.append(dict(artifacts[0]))
    payload["artifacts"] = artifacts
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="artifact names are duplicated",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )

    payload["artifacts"] = [
        {"name": "checkpoint", "path": "missing.pt", "sha256": "0" * 64}
    ]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="artifact is missing: checkpoint",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )

    (release / "present.pt").write_bytes(b"artifact")
    payload["artifacts"] = [
        {
            "name": "checkpoint",
            "path": "present.pt",
            "sha256": "1" * 64,
        }
    ]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="artifact digest mismatch: checkpoint",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )

    payload["artifacts"] = [
        {
            "name": "checkpoint",
            "path": "present.pt",
            "sha256": _sha256(release / "present.pt"),
        }
    ]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="manifest is incomplete",
    ):
        _validate_manifest(
            release_directory=release,
            expected_release_id="release-abc",
        )


def test_reproducibility_requirements_reject_invalid_policy() -> None:
    with pytest.raises(
        ProductionPromotionValidationError,
        match="lacks release policy identity",
    ):
        _reproducibility_requirements_from_manifest(
            {"reproducibility_policy": _policy_payload()}
        )
    with pytest.raises(
        ProductionPromotionValidationError,
        match="lacks reproducibility policy",
    ):
        _reproducibility_requirements_from_manifest(
            {"release_requirements_id": "production_v1"}
        )
    invalid_seeds = dict(_policy_payload())
    invalid_seeds["seeds"] = ["42"]
    with pytest.raises(
        ProductionPromotionValidationError,
        match="seeds are invalid",
    ):
        _reproducibility_requirements_from_manifest(
            {
                "release_requirements_id": "production_v1",
                "reproducibility_policy": invalid_seeds,
            }
        )
    invalid_determinism = dict(_policy_payload())
    invalid_determinism["require_deterministic_execution"] = "yes"
    with pytest.raises(
        ProductionPromotionValidationError,
        match="determinism is invalid",
    ):
        _reproducibility_requirements_from_manifest(
            {
                "release_requirements_id": "production_v1",
                "reproducibility_policy": invalid_determinism,
            }
        )
    invalid_tolerances = dict(_policy_payload())
    invalid_tolerances["metric_tolerances"] = {"validation_loss": "0.03"}
    with pytest.raises(
        ProductionPromotionValidationError,
        match="tolerances are invalid",
    ):
        _reproducibility_requirements_from_manifest(
            {
                "release_requirements_id": "production_v1",
                "reproducibility_policy": invalid_tolerances,
            }
        )


def test_reproducibility_requirements_round_trip() -> None:
    policy = _reproducibility_requirements_from_manifest(
        {
            "release_requirements_id": "production_v1",
            "reproducibility_policy": _policy_payload(),
        }
    )
    assert policy.policy_id == "production_v1"
    assert policy.seeds == (42,)
    assert policy.require_deterministic_execution is True
    assert policy.metric_tolerances == {
        "validation_loss": 0.03,
        "test_loss": 0.03,
    }


def test_manifest_payload_requires_no_reproducibility_policy(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    from release.release_evidence_bundle import ReleaseEvidenceBundle

    evidence = ReleaseEvidenceBundle.load(
        release / "evaluation" / "release_evidence_bundle.json",
        approved_roots=(release,),
    )
    payload = _manifest_payload(
        release_id="release-abc",
        evidence=evidence,
        evidence_bundle_path=release
        / "evaluation"
        / "release_evidence_bundle.json",
        acceptance_report_path=release
        / "evaluation"
        / "acceptance_report.json",
        release_directory=release,
        reproducibility_requirements=None,
    )
    assert payload["release_requirements_id"] is None
    assert payload["reproducibility_policy"] is None


def test_resolve_current_release_rejects_invalid_pointers(
    tmp_path: Path,
) -> None:
    production = tmp_path / "prod"
    production.mkdir()
    with pytest.raises(
        ProductionPromotionValidationError,
        match="invalid JSON artifact",
    ):
        resolve_current_release(production)

    atomic_write_json(
        path=production / CURRENT_POINTER,
        payload={"unknown": "field"},
    )
    with pytest.raises(
        ProductionPromotionValidationError,
        match="pointer fields are invalid",
    ):
        resolve_current_release(production)

    atomic_write_json(
        path=production / CURRENT_POINTER,
        payload={
            "schema_version": "0.1",
            "release_id": "release-abc",
            "release_directory": "releases/release-abc",
            "release_manifest": "release_manifest.json",
            "release_manifest_sha256": "0" * 64,
        },
    )
    with pytest.raises(
        ProductionPromotionValidationError,
        match="pointer schema is invalid",
    ):
        resolve_current_release(production)


def test_resolve_current_release_rejects_escape_and_missing_targets(
    tmp_path: Path,
) -> None:
    _valid_release_directory(tmp_path / "runs")
    production = tmp_path / "prod"
    (production / RELEASES_DIRECTORY).mkdir(parents=True)
    payload = {
        "schema_version": "production_release_pointer.v1",
        "release_id": "release-abc",
        "release_directory": "../other/release-abc",
        "release_manifest": "release_manifest.json",
        "release_manifest_sha256": "0" * 64,
    }
    atomic_write_json(path=production / CURRENT_POINTER, payload=payload)
    with pytest.raises(
        ProductionPromotionValidationError,
        match="path is unsafe",
    ):
        resolve_current_release(production)

    payload["release_directory"] = "releases/sub/release-abc"
    atomic_write_json(path=production / CURRENT_POINTER, payload=payload)
    with pytest.raises(
        ProductionPromotionValidationError,
        match="escapes the releases directory",
    ):
        resolve_current_release(production)

    payload["release_directory"] = "releases/release-other"
    atomic_write_json(path=production / CURRENT_POINTER, payload=payload)
    with pytest.raises(
        ProductionPromotionValidationError,
        match="target is unavailable",
    ):
        resolve_current_release(production)


def test_resolve_current_release_accepts_complete_pointer(
    tmp_path: Path,
) -> None:
    _valid_release_directory(tmp_path)
    production = tmp_path
    target = production / RELEASES_DIRECTORY / "release-abc"
    manifest = target / "release_manifest.json"
    pointer_payload = {
        "schema_version": "production_release_pointer.v1",
        "release_id": "release-abc",
        "release_directory": "releases/release-abc",
        "release_manifest": "release_manifest.json",
        "release_manifest_sha256": _sha256(manifest),
    }
    atomic_write_json(
        path=production / CURRENT_POINTER, payload=pointer_payload
    )
    assert resolve_current_release(production) == target


def test_resolve_current_release_rejects_digest_mismatch(
    tmp_path: Path,
) -> None:
    _valid_release_directory(tmp_path)
    production = tmp_path
    payload = {
        "schema_version": "production_release_pointer.v1",
        "release_id": "release-abc",
        "release_directory": "releases/release-abc",
        "release_manifest": "release_manifest.json",
        "release_manifest_sha256": "0" * 64,
    }
    atomic_write_json(path=production / CURRENT_POINTER, payload=payload)
    with pytest.raises(
        ProductionPromotionValidationError,
        match="manifest digest mismatch",
    ):
        resolve_current_release(production)


def _validate_with_policy(
    release: Path,
    *,
    policy: object = None,
) -> None:
    from config.releases.release_requirements import (
        ReproducibilityRequirements,
    )

    resolved_policy = policy or ReproducibilityRequirements(
        policy_id="production_v1",
        seeds=(42,),
        require_deterministic_execution=True,
        metric_tolerances={"validation_loss": 0.03, "test_loss": 0.03},
    )
    _validate_release_directory(
        release_directory=release,
        expected_release_id="release-abc",
        reproducibility_requirements=resolved_policy,
    )


def _rewrite_evidence_bundle(
    release: Path, payload: dict[str, object]
) -> None:
    bundle_path = release / "evaluation" / "release_evidence_bundle.json"
    bundle_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    manifest_path = release / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["name"] == "release_evidence_bundle":
            artifact["sha256"] = _sha256(bundle_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def test_directory_validation_rejects_foreign_evidence_policy(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    bundle_path = release / "evaluation" / "release_evidence_bundle.json"
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["release_requirements_id"] = "foreign_policy_v1"
    _rewrite_evidence_bundle(release, payload)
    with pytest.raises(
        ProductionPromotionValidationError,
        match="bound to another policy identity",
    ):
        _validate_with_policy(release)


def test_directory_validation_rejects_foreign_evidence_policy_digest(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    bundle_path = release / "evaluation" / "release_evidence_bundle.json"
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["release_requirements_sha256"] = "1" * 64
    _rewrite_evidence_bundle(release, payload)
    with pytest.raises(
        ProductionPromotionValidationError,
        match="bound to another policy digest",
    ):
        _validate_with_policy(release)


def test_directory_validation_rejects_incomplete_checkpoint(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    (release / "checkpoint.pt.sha256").unlink()
    with pytest.raises(
        ProductionPromotionValidationError,
        match="production checkpoint is incomplete",
    ):
        _validate_with_policy(release)


def test_directory_validation_rejects_unpassed_acceptance_report(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    report_path = release / "evaluation" / "acceptance_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["passed"] = False
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    manifest_path = release / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["name"] == "acceptance_report":
            artifact["sha256"] = _sha256(report_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="acceptance report is not passed",
    ):
        _validate_with_policy(release)


def test_directory_validation_rejects_failed_evidence_gate(
    tmp_path: Path,
) -> None:
    release = _valid_release_directory(tmp_path)
    (release / "model.safetensors").unlink()
    with pytest.raises(
        ProductionPromotionValidationError,
        match="release evidence validation failed",
    ):
        _validate_with_policy(release)
