"""Checkpoint contract: headers, blob storage, and staging exclusivity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from filelock import FileLock

from evaluator.leakage.report import generate_report
from release import model_release_publisher as promotion_module
from release.acceptance_result import AcceptanceReport
from release.release_decision import ReleaseDecision
from release.release_evidence_bundle import (
    EvaluationArtifacts,
    ReleaseEvidenceBundle,
)
from release.release_utilities import (
    ProductionPromotionLockError,
)
from schemas.release import ReleaseStatus
from tests.support.release_fixtures import (
    fixture_policy,
    fixture_requirements,
    write_valid_model_card,
    write_valid_reproducibility_report,
)
from training.runtime.checkpoint.contract import (
    CheckpointContract,
    checkpoint_headers_path,
    checkpoint_headers_present,
    read_checkpoint_headers_payload,
    require_blob_checkpoint,
    require_checkpoint_headers,
    write_checkpoint_headers,
)
from training.runtime.checkpoint.io import (
    checkpoint_is_available,
    checkpoint_sha256,
    safe_torch_load,
)
from training.runtime.checkpoint.service import (
    load_model_weights,
    restore_checkpoint_if_requested,
    save_checkpoint,
)


class _ModelSettings:
    artifact_version = "contract-v1"
    model_family = "multimodal_model"

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {}


class _TrainingSettings:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {}

    def effective_min_task_samples(self) -> dict[str, int]:
        return {}


def _fingerprint_schema() -> dict[str, str]:
    return {
        "dataset_fingerprint": "a" * 64,
        "model_config_fingerprint": "b" * 64,
        "training_config_fingerprint": "c" * 64,
    }


def _metadata() -> dict[str, object]:
    return {
        "epochs": 3,
        "final_loss": 0.25,
        "sample_count": 10,
        "dataset_root": "dataset-root",
        "checkpoint_schema": _fingerprint_schema(),
    }


def _save_checkpoint(
    tmp_path: Path,
    *,
    checkpoint_contract: CheckpointContract | None = None,
) -> Path:
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        model=torch.nn.Linear(1, 1, bias=False),
        checkpoint_path=checkpoint_path,
        model_settings=_ModelSettings(),
        training_settings=_TrainingSettings(),
        metadata=_metadata(),
        checkpoint_contract=checkpoint_contract,
    )
    return checkpoint_path


def test_contract_defaults_are_permissive() -> None:
    contract = CheckpointContract()

    assert contract.checkpoint_headers is False
    assert contract.checkpoint_blob_storage is None
    assert contract.staging_lock is None


def test_write_headers_records_required_meta(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)
    headers_path = write_checkpoint_headers(
        checkpoint_path=checkpoint_path,
        metadata=_metadata(),
        sha256=checkpoint_sha256(checkpoint_path),
        artifact_version="contract-v1",
        model_family="multimodal_model",
    )

    assert headers_path == checkpoint_headers_path(checkpoint_path)
    payload = json.loads(headers_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["sha256"] == checkpoint_sha256(checkpoint_path)
    assert payload["artifact_version"] == "contract-v1"
    assert payload["model_family"] == "multimodal_model"
    assert payload["epochs"] == 3
    assert payload["final_loss"] == 0.25
    assert payload["sample_count"] == 10
    assert payload["checkpoint_schema"] == _fingerprint_schema()


def test_write_headers_rejects_invalid_sha256(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="64-character hex"):
        write_checkpoint_headers(
            checkpoint_path=checkpoint_path,
            metadata=_metadata(),
            sha256="not-a-digest",
            artifact_version="contract-v1",
            model_family="multimodal_model",
        )


def test_write_headers_requires_fingerprint_schema(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="checkpoint_schema"):
        write_checkpoint_headers(
            checkpoint_path=checkpoint_path,
            metadata={},
            sha256="a" * 64,
            artifact_version="contract-v1",
            model_family="multimodal_model",
        )


def test_headers_present_is_false_without_sidecar(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)

    assert not checkpoint_headers_present(checkpoint_path)
    with pytest.raises(FileNotFoundError, match="headers"):
        require_checkpoint_headers(checkpoint_path=checkpoint_path)


def test_headers_present_is_false_for_malformed_sidecar(
    tmp_path: Path,
) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)
    headers_path = checkpoint_headers_path(checkpoint_path)
    headers_path.write_text("not json", encoding="utf-8")

    assert not checkpoint_headers_present(checkpoint_path)
    with pytest.raises(FileNotFoundError, match="headers"):
        require_checkpoint_headers(checkpoint_path=checkpoint_path)


def test_require_headers_rejects_digest_mismatch(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)
    write_checkpoint_headers(
        checkpoint_path=checkpoint_path,
        metadata=_metadata(),
        sha256=checkpoint_sha256(checkpoint_path),
        artifact_version="contract-v1",
        model_family="multimodal_model",
    )
    headers_path = checkpoint_headers_path(checkpoint_path)
    payload = json.loads(headers_path.read_text(encoding="utf-8"))
    payload["sha256"] = "e" * 64
    headers_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        require_checkpoint_headers(checkpoint_path=checkpoint_path)


def test_save_with_headers_writes_and_loads_with_contract(
    tmp_path: Path,
) -> None:
    contract = CheckpointContract(checkpoint_headers=True)
    checkpoint_path = _save_checkpoint(tmp_path, checkpoint_contract=contract)

    assert checkpoint_headers_present(checkpoint_path)
    payload = read_checkpoint_headers_payload(checkpoint_path=checkpoint_path)
    assert payload["sha256"] == checkpoint_sha256(checkpoint_path)

    target = torch.nn.Linear(1, 1, bias=False)
    load_model_weights(
        model=target,
        checkpoint_path=checkpoint_path,
        model_settings=_ModelSettings(),
        contract=contract,
    )


def test_load_fails_closed_without_required_headers(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)

    with pytest.raises(FileNotFoundError, match="headers"):
        load_model_weights(
            model=torch.nn.Linear(1, 1, bias=False),
            checkpoint_path=checkpoint_path,
            model_settings=_ModelSettings(),
            contract=CheckpointContract(checkpoint_headers=True),
        )


def test_blob_save_persists_content_addressable(tmp_path: Path) -> None:
    blob_storage = tmp_path / "blobs"
    contract = CheckpointContract(checkpoint_blob_storage=blob_storage)
    checkpoint_path = _save_checkpoint(tmp_path, checkpoint_contract=contract)

    assert checkpoint_path.name == "model.pt"
    manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "blob"
    model_path = require_blob_checkpoint(
        checkpoint_path=checkpoint_path,
        blob_storage=blob_storage,
    )
    assert model_path == (
        blob_storage.resolve()
        / checkpoint_sha256(checkpoint_path)[:2]
        / checkpoint_sha256(checkpoint_path)
        / "checkpoint.pt"
    )
    assert model_path.is_file()
    assert checkpoint_is_available(checkpoint_path)

    payload = safe_torch_load(checkpoint_path)
    assert isinstance(payload, dict)
    assert "model_state" in payload
    assert payload["checkpoint_payload_version"] == 1
    assert payload["checkpoint_format"] == "single_file"


def test_blob_verify_rejects_foreign_store(tmp_path: Path) -> None:
    first_store = tmp_path / "blobs-a"
    second_store = tmp_path / "blobs-b"
    checkpoint_path = _save_checkpoint(
        tmp_path,
        checkpoint_contract=CheckpointContract(
            checkpoint_blob_storage=first_store
        ),
    )

    with pytest.raises(ValueError, match="does not match"):
        require_blob_checkpoint(
            checkpoint_path=checkpoint_path,
            blob_storage=second_store,
        )
    require_blob_checkpoint(
        checkpoint_path=checkpoint_path,
        blob_storage=first_store,
    )


def test_blob_contract_requires_pointer(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)

    with pytest.raises(FileNotFoundError, match="blob"):
        load_model_weights(
            model=torch.nn.Linear(1, 1, bias=False),
            checkpoint_path=checkpoint_path,
            model_settings=_ModelSettings(),
            contract=CheckpointContract(
                checkpoint_blob_storage=tmp_path / "blobs"
            ),
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(path: Path, payload: object) -> None:
    from training.runtime.checkpoint.io import atomic_torch_save

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload=payload, checkpoint_path=path)


def _build_candidate(root: Path) -> SimpleNamespace:
    candidate = root / "candidate"
    dataset = root / "dataset"
    candidate.mkdir(parents=True)
    dataset.mkdir(parents=True)

    checkpoint = root / "checkpoint.pt"
    _write_checkpoint(checkpoint, b"checkpoint-bytes")
    metrics = root / "training_metrics.json"
    metrics.write_text(
        json.dumps({"train_loss": 0.1}) + "\n", encoding="utf-8"
    )

    write_valid_model_card(candidate / "model_card.md")
    (candidate / "inference_config.json").write_text(
        json.dumps({"training_backend": "pipeline_smoke"}) + "\n",
        encoding="utf-8",
    )
    safetensors_path = candidate / "model.safetensors"
    from safetensors.torch import save_file

    save_file({"weight": torch.tensor([1.0])}, str(safetensors_path))
    (candidate / "safetensors_export_status.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "sha256": _sha256(safetensors_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reproducibility = candidate / "evaluation" / "reproducibility_report.json"
    write_valid_reproducibility_report(
        reproducibility,
        policy=fixture_policy(),
    )

    dataset_card = dataset / "dataset_card.json"
    dataset_card.write_text(
        json.dumps({"name": "dataset"}) + "\n", encoding="utf-8"
    )
    leakage = dataset / "evaluation" / "leakage_report.json"
    generate_report(
        left_records=(),
        right_records=(),
        output_path=leakage,
        minimum_coverage={},
    )

    artifacts = EvaluationArtifacts.from_release_outputs(
        candidate_directory=candidate,
        dataset_directory=dataset,
        checkpoint=checkpoint,
        metrics=metrics,
    )
    evidence = ReleaseEvidenceBundle.build(
        mode="candidate",
        artifacts=artifacts,
        release_requirements_id=fixture_policy().policy_id,
        release_requirements_sha256=fixture_policy().policy_sha256,
    )
    evidence_path = candidate / "evaluation" / "release_evidence_bundle.json"
    evidence.write(evidence_path)
    AcceptanceReport.build(
        release_stage="candidate",
        decision=ReleaseDecision(
            status=ReleaseStatus.MODEL_CANDIDATE,
            reasons=(),
        ),
        expected_status=ReleaseStatus.MODEL_CANDIDATE,
        evidence_paths={
            **evidence.evidence_paths,
            "release_evidence_bundle": evidence_path,
        },
    ).write(candidate / "evaluation" / "acceptance_report.json")

    training_result = SimpleNamespace(
        artifacts=SimpleNamespace(checkpoint_path=checkpoint)
    )
    return SimpleNamespace(
        candidate=candidate,
        dataset=dataset,
        checkpoint=checkpoint,
        metrics=metrics,
        evidence_path=evidence_path,
        training_result=training_result,
        evaluation_result=object(),
    )


def _promote(candidate: SimpleNamespace, production: Path) -> object:
    return promotion_module.promote_model(
        candidate_directory=candidate.candidate,
        production_directory=production,
        evidence_bundle_path=candidate.evidence_path,
        settings=object(),
        training_result=candidate.training_result,
        evaluation_result=candidate.evaluation_result,
        dataset_root=candidate.dataset,
        release_requirements=fixture_requirements(),
        metrics_path=candidate.metrics,
    )


def test_staging_lock_promotes_when_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        promotion_module,
        "decide_release",
        lambda **_kwargs: ReleaseDecision(
            status=ReleaseStatus.MODEL_ACCEPTED,
            reasons=(),
        ),
    )
    candidate = _build_candidate(tmp_path / "run")
    staging_lock = tmp_path / "staging.lock"

    report = promotion_module.promote_model(
        candidate_directory=candidate.candidate,
        production_directory=tmp_path / "production",
        evidence_bundle_path=candidate.evidence_path,
        settings=object(),
        training_result=candidate.training_result,
        evaluation_result=candidate.evaluation_result,
        dataset_root=candidate.dataset,
        release_requirements=fixture_requirements(),
        metrics_path=candidate.metrics,
        staging_lock=staging_lock,
    )

    assert report.to_payload()["passed"] is True
    assert (
        not staging_lock.exists()
        or FileLock(str(staging_lock)).is_locked is False
    )


def test_staging_lock_fails_closed_while_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        promotion_module,
        "decide_release",
        lambda **_kwargs: ReleaseDecision(
            status=ReleaseStatus.MODEL_ACCEPTED,
            reasons=(),
        ),
    )
    candidate = _build_candidate(tmp_path / "run")
    staging_lock = tmp_path / "staging.lock"

    held = FileLock(str(staging_lock))
    with held.acquire(timeout=0):
        with pytest.raises(ProductionPromotionLockError, match="staging lock"):
            promotion_module.promote_model(
                candidate_directory=candidate.candidate,
                production_directory=tmp_path / "production",
                evidence_bundle_path=candidate.evidence_path,
                settings=object(),
                training_result=candidate.training_result,
                evaluation_result=candidate.evaluation_result,
                dataset_root=candidate.dataset,
                release_requirements=fixture_requirements(),
                metrics_path=candidate.metrics,
                staging_lock=staging_lock,
            )


def _valid_payload(**overrides: object) -> dict[str, object]:
    base = {
        "checkpoint_payload_version": 1,
        "checkpoint_format": "single_file",
        "model_family": "multimodal_model",
        "artifact_version": "test-v1",
        "model_state": {"weight": torch.tensor([1.0])},
        "metadata": {
            "checkpoint_schema": {
                "model_config_fingerprint": "a" * 64,
                "training_config_fingerprint": "b" * 64,
                "dataset_fingerprint": "c" * 64,
                "tokenizer_fingerprint": "d" * 64,
                "label_mapping_fingerprint": "e" * 64,
                "dependency_fingerprint": "f" * 64,
            }
        },
    }
    base.update(overrides)
    return base


class _TestModelSettings:
    artifact_version = "test-v1"
    model_family = "multimodal_model"

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {}


def test_unknown_payload_version_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(
        checkpoint, _valid_payload(checkpoint_payload_version=999)
    )

    with pytest.raises(
        ValueError, match="unsupported checkpoint payload version"
    ):
        load_model_weights(
            model=torch.nn.Linear(1, 1),
            checkpoint_path=checkpoint,
            model_settings=_TestModelSettings(),
        )


@pytest.mark.parametrize(
    ("bad_format", "expected_error"),
    [
        ("distributed_sharded_v3", "unsupported checkpoint format"),
        ("foreign_format", "unsupported checkpoint format"),
        ("", "unsupported checkpoint format"),
        (None, "checkpoint_format must be a string"),
        (123, "checkpoint_format must be a string"),
    ],
)
def test_unknown_explicit_format_rejected(
    tmp_path: Path, bad_format: object, expected_error: str
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint, _valid_payload(checkpoint_format=bad_format))

    with pytest.raises(ValueError, match=expected_error):
        load_model_weights(
            model=torch.nn.Linear(1, 1),
            checkpoint_path=checkpoint,
            model_settings=_TestModelSettings(),
        )


def test_current_missing_family_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _valid_payload()
    del payload["model_family"]
    _write_checkpoint(checkpoint, payload)

    with pytest.raises(
        ValueError, match="missing required field: model_family"
    ):
        load_model_weights(
            model=torch.nn.Linear(1, 1),
            checkpoint_path=checkpoint,
            model_settings=_TestModelSettings(),
        )


def test_explicit_null_family_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint, _valid_payload(model_family=None))

    with pytest.raises(ValueError, match="must not be null"):
        load_model_weights(
            model=torch.nn.Linear(1, 1),
            checkpoint_path=checkpoint,
            model_settings=_TestModelSettings(),
        )


def test_wrong_family_rejected_in_load_model_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint, _valid_payload(model_family="other_model"))

    with pytest.raises(ValueError, match="model_family mismatch"):
        load_model_weights(
            model=torch.nn.Linear(1, 1),
            checkpoint_path=checkpoint,
            model_settings=_TestModelSettings(),
        )


def test_wrong_family_rejected_in_restore_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint, _valid_payload(model_family="other_model"))

    with pytest.raises(ValueError, match="model_family mismatch"):
        restore_checkpoint_if_requested(
            model=torch.nn.Linear(1, 1),
            optimizer=torch.optim.SGD(
                torch.nn.Linear(1, 1).parameters(), lr=0.1
            ),
            scheduler=None,
            settings=SimpleNamespace(resume_from_checkpoint=str(checkpoint)),
            model_settings=_TestModelSettings(),
            dataset_root=tmp_path,
            dataset_manifest_sha256="a" * 64,
        )


def test_headers_cross_validates_model_family(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint, _valid_payload(model_family="other_model"))
    write_checkpoint_headers(
        checkpoint_path=checkpoint,
        metadata=_valid_payload()["metadata"],
        sha256=checkpoint_sha256(checkpoint),
        artifact_version="test-v1",
        model_family="multimodal_model",
    )

    with pytest.raises(ValueError, match="model_family.*does not match"):
        require_checkpoint_headers(
            checkpoint_path=checkpoint,
            expected_model_family="multimodal_model",
            expected_artifact_version="test-v1",
        )


def test_headers_cross_validates_artifact_version(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint, _valid_payload(artifact_version="other-v1"))
    write_checkpoint_headers(
        checkpoint_path=checkpoint,
        metadata=_valid_payload()["metadata"],
        sha256=checkpoint_sha256(checkpoint),
        artifact_version="test-v1",
        model_family="multimodal_model",
    )

    with pytest.raises(ValueError, match="artifact_version.*does not match"):
        require_checkpoint_headers(
            checkpoint_path=checkpoint,
            expected_model_family="multimodal_model",
            expected_artifact_version="test-v1",
        )


def test_save_checkpoint_includes_payload_version_and_format(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        model=torch.nn.Linear(1, 1),
        checkpoint_path=checkpoint,
        model_settings=_TestModelSettings(),
        training_settings=_TrainingSettings(),
        metadata=_valid_payload()["metadata"],
    )
    payload = safe_torch_load(checkpoint)
    assert payload["checkpoint_payload_version"] == 1
    assert payload["checkpoint_format"] == "single_file"
    assert payload["model_family"] == "multimodal_model"


def test_blob_checkpoint_rejects_symlink_escape(tmp_path: Path) -> None:
    """Symlinks in blob store must not escape the blob root."""
    import hashlib

    from training.runtime.checkpoint.contract import require_blob_checkpoint
    from training.runtime.checkpoint.io import resolve_checkpoint_model_path

    blob_storage = tmp_path / "blobs"
    blob_storage.mkdir(parents=True)

    # Create a legitimate checkpoint in the blob store
    real_checkpoint = tmp_path / "real_checkpoint.pt"
    real_checkpoint.write_bytes(b"legitimate-checkpoint-data")
    real_sha256 = hashlib.sha256(b"legitimate-checkpoint-data").hexdigest()

    # Set up the blob store structure
    blob_dir = blob_storage / real_sha256[:2] / real_sha256
    blob_dir.mkdir(parents=True)
    real_blob_path = blob_dir / "checkpoint.pt"
    real_blob_path.write_bytes(b"legitimate-checkpoint-data")
    (blob_dir / "checkpoint.pt.sha256").write_text(
        f"{real_sha256}  checkpoint.pt\n"
    )

    # Create a malicious symlink pointing outside the blob store
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_checkpoint = outside / "evil.pt"
    outside_checkpoint.write_bytes(b"evil-data")
    outside_sha256 = hashlib.sha256(b"evil-data").hexdigest()

    # Create a manifest pointing to the outside path via symlink
    symlink_dir = blob_storage / outside_sha256[:2] / outside_sha256
    symlink_dir.mkdir(parents=True)
    symlink_path = symlink_dir / "checkpoint.pt"
    symlink_path.symlink_to(outside_checkpoint)
    (symlink_dir / "checkpoint.pt.sha256").write_text(
        f"{outside_sha256}  checkpoint.pt\n"
    )

    # Create a manifest that references the symlinked path
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": "blob",
        "blob_storage": str(blob_storage),
        "file": "checkpoint.pt",
        "sha256": outside_sha256,
    }
    manifest_path.write_text(json.dumps(manifest))

    # resolve_checkpoint_model_path should reject the symlink
    assert resolve_checkpoint_model_path(manifest_path) is None

    # require_blob_checkpoint should also reject
    with pytest.raises(ValueError, match="invalid|escapes"):
        require_blob_checkpoint(
            checkpoint_path=manifest_path,
            blob_storage=blob_storage,
        )
