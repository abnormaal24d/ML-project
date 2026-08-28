from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from filelock import FileLock
from safetensors.torch import save_file

from evaluator.leakage.report import generate_report
from release import model_release_publisher as promotion_module
from release.acceptance_result import AcceptanceReport
from release.release_decision import ReleaseDecision
from release.release_evidence_bundle import (
    EvaluationArtifacts,
    ReleaseEvidenceBundle,
)
from release.release_staging import _release_checkpoint_is_available
from schemas.release import ReleaseStatus
from tests.support.release_fixtures import (
    fixture_policy,
    fixture_requirements,
    write_valid_model_card,
    write_valid_reproducibility_report,
)
from training.runtime.checkpoint.io import (
    atomic_torch_save,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_checkpoint(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload={"marker": payload}, checkpoint_path=path)


def _build_candidate(
    root: Path,
    *,
    marker: str,
    exploding_file: bool = False,
    versioned_checkpoint: bool = False,
) -> SimpleNamespace:
    candidate = root / "candidate"
    dataset = root / "dataset"
    candidate.mkdir(parents=True)
    dataset.mkdir(parents=True)

    checkpoint = root / "checkpoint.pt"
    if versioned_checkpoint:
        atomic_torch_save(
            payload={"marker": marker, "weight": torch.tensor([1.0])},
            checkpoint_path=checkpoint,
        )
    else:
        _write_checkpoint(checkpoint, f"checkpoint-{marker}".encode())
    metrics = root / "training_metrics.json"
    metrics.write_text(
        json.dumps({"train_loss": 0.1, "marker": marker}) + "\n",
        encoding="utf-8",
    )

    write_valid_model_card(candidate / "model_card.md")
    (candidate / "inference_config.json").write_text(
        json.dumps({"training_backend": "pipeline_smoke", "marker": marker})
        + "\n",
        encoding="utf-8",
    )
    safetensors_path = candidate / "model.safetensors"
    save_file(
        {"weight": torch.tensor([float(len(marker))])}, str(safetensors_path)
    )
    (candidate / "safetensors_export_status.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "sha256": hashlib.sha256(
                    safetensors_path.read_bytes()
                ).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if exploding_file:
        (candidate / "zz-explode.bin").write_bytes(b"explode")

    reproducibility = candidate / "evaluation" / "reproducibility_report.json"
    write_valid_reproducibility_report(
        reproducibility,
        policy=fixture_policy(),
    )

    dataset_card = dataset / "dataset_card.json"
    dataset_card.write_text(
        json.dumps({"name": f"dataset-{marker}"}) + "\n",
        encoding="utf-8",
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
    evidence_path = candidate / "evaluation" / ("release_evidence_bundle.json")
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


def _accept_release(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        promotion_module,
        "decide_release",
        lambda **_kwargs: ReleaseDecision(
            status=ReleaseStatus.MODEL_ACCEPTED,
            reasons=(),
        ),
    )


def _promote(
    candidate: SimpleNamespace,
    production: Path,
):
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


def test_promotion_publishes_one_complete_version_and_atomic_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    production = tmp_path / "production"

    report = _promote(candidate, production)

    assert report.to_payload()["passed"] is True
    current = promotion_module.resolve_current_release(production)
    assert current.parent == production / "releases"
    assert current.name.startswith("release-")
    assert (current / "model.safetensors").is_file()
    assert (
        json.loads(
            (current / "inference_config.json").read_text(encoding="utf-8")
        )["marker"]
        == "one"
    )
    assert _release_checkpoint_is_available(current / "checkpoint.pt")
    assert (current / "evaluation" / "training_metrics.json").is_file()

    bundle = ReleaseEvidenceBundle.load(
        current / "evaluation" / "release_evidence_bundle.json",
        approved_roots=(current,),
    )
    assert all(
        Path(reference.path).is_relative_to(current)
        for reference in bundle.references
    )
    assert not list((production / "releases").glob(".staging-*"))


def test_promotion_materializes_real_versioned_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(
        tmp_path / "run-versioned",
        marker="versioned",
        versioned_checkpoint=True,
    )
    production = tmp_path / "production"

    _promote(candidate, production)

    current = promotion_module.resolve_current_release(production)
    published_checkpoint = current / "checkpoint.pt"
    assert _release_checkpoint_is_available(published_checkpoint)
    payload = torch.load(
        published_checkpoint, map_location="cpu", weights_only=True
    )
    assert payload["marker"] == "versioned"
    assert torch.equal(payload["weight"], torch.tensor([1.0]))


def test_copy_failure_keeps_previous_release_live_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    production = tmp_path / "production"
    first = _build_candidate(tmp_path / "run-one", marker="one")
    _promote(first, production)
    previous_pointer = (production / "current.json").read_bytes()
    previous_release = promotion_module.resolve_current_release(production)

    second = _build_candidate(
        tmp_path / "run-two",
        marker="two",
        exploding_file=True,
    )
    original_copy2 = promotion_module.shutil.copy2

    def fail_late(source: Path, target: Path) -> object:
        if Path(source).name == "zz-explode.bin":
            raise OSError("simulated copy failure")
        return original_copy2(source, target)

    monkeypatch.setattr(promotion_module.shutil, "copy2", fail_late)

    with pytest.raises(OSError, match="simulated copy failure"):
        _promote(second, production)

    assert (production / "current.json").read_bytes() == previous_pointer
    assert (
        promotion_module.resolve_current_release(production)
        == previous_release
    )
    assert not list((production / "releases").glob(".staging-*"))
    assert [
        path
        for path in (production / "releases").iterdir()
        if path.name.startswith("release-")
    ] == [previous_release]


def test_pointer_publication_failure_rolls_back_new_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    production = tmp_path / "production"
    first = _build_candidate(tmp_path / "run-one", marker="one")
    _promote(first, production)
    previous_pointer = (production / "current.json").read_bytes()
    previous_release = promotion_module.resolve_current_release(production)

    second = _build_candidate(tmp_path / "run-two", marker="two")
    original_replace = promotion_module.os.replace

    def fail_current_pointer(source: Path, target: Path) -> None:
        if Path(target) == production / "current.json":
            raise OSError("simulated pointer publication failure")
        original_replace(source, target)

    monkeypatch.setattr(
        promotion_module.os,
        "replace",
        fail_current_pointer,
    )

    with pytest.raises(
        OSError,
        match="simulated pointer publication failure",
    ):
        _promote(second, production)

    assert (production / "current.json").read_bytes() == previous_pointer
    assert (
        promotion_module.resolve_current_release(production)
        == previous_release
    )
    releases = [
        path
        for path in (production / "releases").iterdir()
        if path.name.startswith("release-")
    ]
    assert releases == [previous_release]
    assert not list((production / "releases").glob(".staging-*"))


def test_concurrent_promotion_is_rejected_by_existing_file_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    production = tmp_path / "production"
    production.mkdir()
    lock = FileLock(str(production / ".promotion.lock"))

    with lock:
        with pytest.raises(
            promotion_module.ProductionPromotionLockError,
            match="another production promotion is active",
        ):
            _promote(candidate, production)

    assert not (production / "current.json").exists()


def test_promotion_rejects_missing_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    (tmp_path / "metrics-directory").mkdir()
    with pytest.raises(FileNotFoundError, match="promotion metrics are"):
        promotion_module.promote_model(
            candidate_directory=candidate.candidate,
            production_directory=tmp_path / "production",
            evidence_bundle_path=candidate.evidence_path,
            settings=object(),
            training_result=candidate.training_result,
            evaluation_result=candidate.evaluation_result,
            dataset_root=candidate.dataset,
            release_requirements=fixture_requirements(),
            metrics_path=tmp_path / "metrics-directory",
        )


def test_promotion_rejects_wrong_evidence_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    production_evidence = ReleaseEvidenceBundle.build(
        mode="production_model",
        artifacts=EvaluationArtifacts.from_release_outputs(
            candidate_directory=candidate.candidate,
            dataset_directory=candidate.dataset,
            checkpoint=candidate.checkpoint,
            metrics=candidate.metrics,
        ),
        release_requirements_id=fixture_policy().policy_id,
        release_requirements_sha256=fixture_policy().policy_sha256,
    )
    production_evidence.write(candidate.evidence_path)
    with pytest.raises(
        ValueError,
        match="promotion requires candidate evidence",
    ):
        _promote(candidate, tmp_path / "production")


def test_promotion_requires_candidate_acceptance_bound_to_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    AcceptanceReport.build(
        release_stage="candidate",
        decision=ReleaseDecision(
            status=ReleaseStatus.MODEL_CANDIDATE,
            reasons=(),
        ),
        expected_status=ReleaseStatus.MODEL_CANDIDATE,
        evidence_paths={},
    ).write(candidate.candidate / "evaluation" / "acceptance_report.json")

    with pytest.raises(
        ValueError,
        match="candidate acceptance report is not bound to the evidence bundle",
    ):
        _promote(candidate, tmp_path / "production")


def test_promotion_rejects_candidate_without_serving_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    (candidate.candidate / "model.safetensors").unlink()
    (candidate.candidate / "safetensors_export_status.json").unlink()

    with pytest.raises(
        promotion_module.ProductionPromotionValidationError,
        match="serving_artifact_missing",
    ):
        _promote(candidate, tmp_path / "production")


def test_promotion_rejects_foreign_policy_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    from config.releases.release_requirements import (
        ReleaseRequirements,
        ReproducibilityRequirements,
    )

    foreign = ReleaseRequirements(
        release_id="foreign_policy",
        release_stage="production_model",
        required_modalities=("image",),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("image_tagging",),
        optional_tasks=(),
        blocked_capabilities=(),
        reproducibility=ReproducibilityRequirements(
            policy_id="foreign_policy_v1",
            seeds=(42,),
            require_deterministic_execution=True,
            metric_tolerances={"validation_loss": 0.03, "test_loss": 0.03},
        ),
    )
    with pytest.raises(
        ValueError,
        match="another release policy identity",
    ):
        promotion_module.promote_model(
            candidate_directory=candidate.candidate,
            production_directory=tmp_path / "production",
            evidence_bundle_path=candidate.evidence_path,
            settings=object(),
            training_result=candidate.training_result,
            evaluation_result=candidate.evaluation_result,
            dataset_root=candidate.dataset,
            release_requirements=foreign,
            metrics_path=candidate.metrics,
        )


def test_promotion_rejects_foreign_policy_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    from config.releases.release_requirements import (
        ReleaseRequirements,
        ReproducibilityRequirements,
    )

    foreign = ReleaseRequirements(
        release_id="production_v1",
        release_stage="production_model",
        required_modalities=("image",),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("image_tagging",),
        optional_tasks=(),
        blocked_capabilities=(),
        reproducibility=ReproducibilityRequirements(
            policy_id="production_v1",
            seeds=(43,),
            require_deterministic_execution=True,
            metric_tolerances={"validation_loss": 0.03, "test_loss": 0.03},
        ),
    )
    with pytest.raises(
        ValueError,
        match="another release policy digest",
    ):
        promotion_module.promote_model(
            candidate_directory=candidate.candidate,
            production_directory=tmp_path / "production",
            evidence_bundle_path=candidate.evidence_path,
            settings=object(),
            training_result=candidate.training_result,
            evaluation_result=candidate.evaluation_result,
            dataset_root=candidate.dataset,
            release_requirements=foreign,
            metrics_path=candidate.metrics,
        )


def test_promotion_rejects_checkpoint_bound_to_another_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    other_checkpoint = tmp_path / "other-checkpoint.pt"
    atomic_torch_save(
        payload={"marker": b"other"},
        checkpoint_path=other_checkpoint,
    )
    with pytest.raises(
        ValueError,
        match="bound to another checkpoint",
    ):
        promotion_module.promote_model(
            candidate_directory=candidate.candidate,
            production_directory=tmp_path / "production",
            evidence_bundle_path=candidate.evidence_path,
            settings=object(),
            training_result=SimpleNamespace(
                artifacts=SimpleNamespace(checkpoint_path=other_checkpoint)
            ),
            evaluation_result=candidate.evaluation_result,
            dataset_root=candidate.dataset,
            release_requirements=fixture_requirements(),
            metrics_path=candidate.metrics,
        )


def test_promotion_rejects_unaccepted_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        promotion_module,
        "decide_release",
        lambda **_kwargs: ReleaseDecision(
            status=ReleaseStatus.FAILED,
            reasons=(),
        ),
    )
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    with pytest.raises(RuntimeError, match="production promotion rejected"):
        _promote(candidate, tmp_path / "production")


def test_promotion_retry_reuses_existing_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    production = tmp_path / "production"

    first_report = _promote(candidate, production)
    current = promotion_module.resolve_current_release(production)

    second_report = _promote(candidate, production)

    assert first_report.to_payload() == second_report.to_payload()
    assert promotion_module.resolve_current_release(production) == current


def test_promotion_rejects_incomplete_staged_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    original_stage = promotion_module._stage_evidence

    def break_checkpoint(**kwargs: object) -> dict[str, Path]:
        paths = original_stage(**kwargs)
        (paths["checkpoint"].with_suffix(".pt.sha256")).unlink()
        return paths

    monkeypatch.setattr(
        promotion_module,
        "_stage_evidence",
        break_checkpoint,
    )
    with pytest.raises(
        promotion_module.ProductionPromotionValidationError,
        match="staged production checkpoint is incomplete",
    ):
        _promote(candidate, tmp_path / "production")


def test_promotion_rejects_staged_checkpoint_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    original_stage = promotion_module._stage_evidence

    def tamper_checkpoint(**kwargs: object) -> dict[str, Path]:
        paths = original_stage(**kwargs)
        paths["checkpoint"].write_bytes(b"tampered")
        return paths

    monkeypatch.setattr(
        promotion_module,
        "_stage_evidence",
        tamper_checkpoint,
    )
    with pytest.raises(
        promotion_module.ProductionPromotionValidationError,
        match="staged production checkpoint digest mismatch",
    ):
        _promote(candidate, tmp_path / "production")


def test_promotion_rejects_pointer_resolution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")

    def broken_resolve(_production: Path) -> Path:
        return tmp_path / "elsewhere"

    monkeypatch.setattr(
        promotion_module,
        "resolve_current_release",
        broken_resolve,
    )
    with pytest.raises(
        promotion_module.ProductionPromotionValidationError,
        match="did not resolve",
    ):
        _promote(candidate, tmp_path / "production")


def test_promotion_rejects_incomplete_published_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")

    class BrokenReport:
        def to_payload(self) -> dict[str, object]:
            return {"passed": False}

    monkeypatch.setattr(
        promotion_module.AcceptanceReport,
        "build",
        lambda **_kwargs: BrokenReport(),
    )
    with pytest.raises(
        promotion_module.ProductionPromotionValidationError,
        match="published production evidence is incomplete",
    ):
        _promote(candidate, tmp_path / "production")


def test_promotion_retry_restores_missing_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    production = tmp_path / "production"

    _promote(candidate, production)
    current = promotion_module.resolve_current_release(production)
    (production / "current.json").unlink()

    _promote(candidate, production)

    assert promotion_module.resolve_current_release(production) == current


def test_promotion_retry_rejects_unresolvable_existing_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_release(monkeypatch)
    candidate = _build_candidate(tmp_path / "run-one", marker="one")
    production = tmp_path / "production"

    _promote(candidate, production)
    (production / "current.json").unlink()

    def broken_resolve(_production: Path) -> Path:
        return tmp_path / "elsewhere"

    monkeypatch.setattr(
        promotion_module,
        "resolve_current_release",
        broken_resolve,
    )
    with pytest.raises(
        promotion_module.ProductionPromotionValidationError,
        match="did not resolve",
    ):
        _promote(candidate, production)


def test_promotion_rejects_symlink_in_candidate(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    target = candidate / "real.bin"
    target.write_bytes(b"x")
    link = candidate / "link.bin"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="symlink"):
        promotion_module._copy_candidate_tree(
            source_root=candidate,
            target_root=tmp_path / "staging",
        )


def test_promotion_rejects_unsupported_candidate_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "weird.bin").write_bytes(b"x")
    original_is_file = Path.is_file

    def fake_is_file(self: Path) -> bool:
        if self.name == "weird.bin":
            return False
        return original_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    with pytest.raises(ValueError, match="unsupported entry"):
        promotion_module._copy_candidate_tree(
            source_root=candidate,
            target_root=tmp_path / "staging",
        )
