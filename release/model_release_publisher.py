"""Transactional candidate-to-production release promotion."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock, Timeout

from release.acceptance_evaluator import decide_release
from release.acceptance_result import AcceptanceReport
from release.current_release_pointer import resolve_current_release
from release.release_decision import ReleaseDecision, ReleaseStatus
from release.release_directory_validation import _validate_evidence_bundle
from release.release_evidence_bundle import (
    EvaluationArtifacts,
    ReleaseEvidenceBundle,
)
from release.release_manifest import _manifest_payload, _validate_manifest
from release.release_staging import (
    _bundle_for_paths,
    _copy_regular_file,
    _release_checkpoint_is_available,
    _stage_evidence,
)
from release.release_utilities import (
    CURRENT_POINTER,
    POINTER_SCHEMA,
    PROMOTION_LOCK,
    RELEASES_DIRECTORY,
    STAGING_PREFIX,
    ProductionPromotionLockError,
    ProductionPromotionValidationError,
    atomic_write_json,
    cleanup_staging_directories,
    current_pointer_references,
    fsync_directory,
    fsync_tree,
    require_directory,
    require_separate_roots,
    same_file,
    sha256,
)
from training.runtime.checkpoint.io import (
    checkpoint_is_available,
    checkpoint_sha256,
)

if TYPE_CHECKING:
    from config.releases.release_requirements import (
        ReleaseRequirements,
        ReproducibilityRequirements,
    )
    from config.settings.datasets import DatasetValidatorSettings
    from evaluator.results import EvaluationResult
    from training.runtime.results import TrainingRunResult


def promote_model(
    *,
    candidate_directory: Path,
    production_directory: Path,
    evidence_bundle_path: Path,
    settings: DatasetValidatorSettings,
    training_result: TrainingRunResult,
    evaluation_result: EvaluationResult,
    dataset_root: Path,
    release_requirements: ReleaseRequirements | None = None,
    metrics_path: Path | None = None,
    run_mode: str = "full",
    staging_lock: Path | None = None,
) -> AcceptanceReport:
    """Promote one accepted candidate through an atomic release pointer.

    The live production location contains immutable version directories and a
    single atomically replaced ``current.json`` pointer. Candidate artifacts
    are assembled and validated in a private staging directory. The previous
    pointer remains authoritative until the new release is complete.

    Production promotion is fail-closed: the active release contract and its
    reproducibility policy are mandatory, and the evidence bundle must be
    bound to that exact policy identity.

    When *staging_lock* is provided, release staging is exclusive to that
    lock: a concurrent promotion that cannot acquire it fails closed.
    """

    if release_requirements is None:
        raise ValueError("production promotion requires release requirements")
    reproducibility_requirements = release_requirements.reproducibility
    if reproducibility_requirements is None:
        raise ValueError(
            "production promotion requires a reproducibility policy"
        )

    candidate_directory = require_directory(candidate_directory)
    dataset_root = require_directory(dataset_root)
    production_directory = production_directory.resolve(strict=False)
    require_separate_roots(
        candidate_directory=candidate_directory,
        production_directory=production_directory,
    )

    checkpoint = training_result.artifacts.checkpoint_path.resolve(
        strict=False
    )
    if not checkpoint_is_available(checkpoint):
        raise FileNotFoundError(
            f"promotion checkpoint is unavailable: {checkpoint}"
        )

    effective_metrics_path = (
        metrics_path
        if metrics_path is not None
        else candidate_directory / "training_metrics.json"
    ).resolve(strict=True)
    if not effective_metrics_path.is_file():
        raise FileNotFoundError(
            f"promotion metrics are unavailable: {effective_metrics_path}"
        )

    candidate_evidence = ReleaseEvidenceBundle.load(
        evidence_bundle_path,
        approved_roots=(
            candidate_directory,
            dataset_root,
            checkpoint.parent,
            effective_metrics_path.parent,
        ),
    )
    if candidate_evidence.release_mode != "candidate":
        raise ValueError("promotion requires candidate evidence")

    if (
        candidate_evidence.release_requirements_id
        != reproducibility_requirements.policy_id
    ):
        raise ValueError(
            "release evidence is bound to another release policy identity"
        )
    if (
        candidate_evidence.release_requirements_sha256
        != reproducibility_requirements.policy_sha256
    ):
        raise ValueError(
            "release evidence is bound to another release policy digest"
        )

    evidenced_checkpoint = candidate_evidence.path("checkpoint")
    if evidenced_checkpoint is None or not same_file(
        evidenced_checkpoint,
        checkpoint,
    ):
        raise ValueError("release evidence is bound to another checkpoint")
    evidenced_metrics = candidate_evidence.path("training_metrics")
    if evidenced_metrics is None or not same_file(
        evidenced_metrics,
        effective_metrics_path,
    ):
        raise ValueError("release evidence is bound to other training metrics")

    _validate_evidence_bundle(
        candidate_evidence,
        reproducibility_requirements=reproducibility_requirements,
    )
    candidate_report = AcceptanceReport.load(
        candidate_directory / "evaluation" / "acceptance_report.json"
    )
    if candidate_report.release_stage != "candidate":
        raise ValueError(
            "promotion requires a passed candidate acceptance report"
        )
    candidate_bundle_reference = next(
        (
            reference
            for reference in candidate_report.evidence
            if reference.name == "release_evidence_bundle"
        ),
        None,
    )
    if (
        candidate_bundle_reference is None
        or not same_file(
            Path(candidate_bundle_reference.path), evidence_bundle_path
        )
        or candidate_bundle_reference.sha256 != sha256(evidence_bundle_path)
    ):
        raise ValueError(
            "candidate acceptance report is not bound to the evidence bundle"
        )

    evidence = ReleaseEvidenceBundle.build(
        mode="production_model",
        artifacts=EvaluationArtifacts.from_release_outputs(
            candidate_directory=candidate_directory,
            dataset_directory=dataset_root,
            checkpoint=checkpoint,
            metrics=effective_metrics_path,
        ),
        release_requirements_id=reproducibility_requirements.policy_id,
        release_requirements_sha256=reproducibility_requirements.policy_sha256,
    )
    production_requirements = replace(
        release_requirements,
        release_stage="production_model",
    )

    decision = _decide_for_promotion(
        settings=settings,
        training_result=training_result,
        evaluation_result=evaluation_result,
        run_mode=run_mode,
        checkpoint_path=checkpoint,
        metrics_path=effective_metrics_path,
        dataset_root=dataset_root,
        release_requirements=production_requirements,
        evidence=evidence,
    )
    if decision.status is not ReleaseStatus.MODEL_ACCEPTED:
        raise RuntimeError("production promotion rejected")

    production_directory.mkdir(parents=True, exist_ok=True)
    releases_directory = production_directory / RELEASES_DIRECTORY
    releases_directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(production_directory / PROMOTION_LOCK))
    staging_lock_handle = (
        FileLock(str(staging_lock.resolve()))
        if staging_lock is not None
        else None
    )

    try:
        if staging_lock_handle is not None:
            try:
                with staging_lock_handle.acquire(timeout=0):
                    with lock.acquire(timeout=0):
                        return _promote_under_lock(
                            candidate_directory=candidate_directory,
                            production_directory=production_directory,
                            releases_directory=releases_directory,
                            evidence=evidence,
                            metrics_path=effective_metrics_path,
                            decision=decision,
                            checkpoint_sha256=checkpoint_sha256(checkpoint),
                            metricssha256=sha256(effective_metrics_path),
                            reproducibility_requirements=reproducibility_requirements,
                        )
            except Timeout as exc:
                raise ProductionPromotionLockError(
                    f"release staging lock is held by another promotion: "
                    f"{staging_lock}"
                ) from exc
        else:
            with lock.acquire(timeout=0):
                return _promote_under_lock(
                    candidate_directory=candidate_directory,
                    production_directory=production_directory,
                    releases_directory=releases_directory,
                    evidence=evidence,
                    metrics_path=effective_metrics_path,
                    decision=decision,
                    checkpoint_sha256=checkpoint_sha256(checkpoint),
                    metricssha256=sha256(effective_metrics_path),
                    reproducibility_requirements=reproducibility_requirements,
                )
    except Timeout as exc:
        raise ProductionPromotionLockError(
            f"another production promotion is active: {production_directory}"
        ) from exc


def _decide_for_promotion(
    *,
    settings: DatasetValidatorSettings,
    training_result: TrainingRunResult,
    evaluation_result: EvaluationResult,
    run_mode: str,
    checkpoint_path: Path,
    metrics_path: Path,
    dataset_root: Path,
    release_requirements: ReleaseRequirements | None,
    evidence: ReleaseEvidenceBundle,
) -> ReleaseDecision:
    return decide_release(
        settings=settings,
        training_result=training_result,
        evaluation_result=evaluation_result,
        run_mode=run_mode,
        release_stage="production_model",
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        dataset_root=dataset_root,
        release_requirements=release_requirements,
        evidence=evidence,
    )


def _promote_under_lock(
    *,
    candidate_directory: Path,
    production_directory: Path,
    releases_directory: Path,
    evidence: ReleaseEvidenceBundle,
    metrics_path: Path,
    decision: ReleaseDecision,
    checkpoint_sha256: str,
    metricssha256: str,
    reproducibility_requirements: ReproducibilityRequirements,
) -> AcceptanceReport:
    cleanup_staging_directories(releases_directory)
    release_id = _release_id_for(
        checkpoint_sha256=checkpoint_sha256,
        metricssha256=metricssha256,
        evidence=evidence,
    )
    final_directory = releases_directory / release_id
    if final_directory.exists():
        return _load_existing_release(
            production_directory=production_directory,
            releases_directory=releases_directory,
            final_directory=final_directory,
            release_id=release_id,
        )

    staging_directory = Path(
        tempfile.mkdtemp(
            dir=str(releases_directory),
            prefix=STAGING_PREFIX,
        )
    )

    try:
        _copy_candidate_tree(
            source_root=candidate_directory,
            target_root=staging_directory,
        )
        staged_paths = _stage_evidence(
            evidence=evidence,
            candidate_directory=candidate_directory,
            staging_directory=staging_directory,
        )
        staged_metrics_path = staged_paths["training_metrics"]

        staged_evidence = _bundle_for_paths(
            source=evidence,
            paths=staged_paths,
        )
        _validate_evidence_bundle(
            staged_evidence,
            reproducibility_requirements=reproducibility_requirements,
        )
        staged_checkpoint = staged_evidence.path("checkpoint")
        if staged_checkpoint is None or not _release_checkpoint_is_available(
            staged_checkpoint
        ):
            raise ProductionPromotionValidationError(
                "staged production checkpoint is incomplete"
            )
        if sha256(staged_checkpoint) != checkpoint_sha256:
            raise ProductionPromotionValidationError(
                "staged production checkpoint digest mismatch"
            )

        final_paths = {
            name: final_directory / path.relative_to(staging_directory)
            for name, path in staged_paths.items()
        }
        final_metrics_path = final_directory / staged_metrics_path.relative_to(
            staging_directory
        )
        final_evidence = _bundle_for_paths(
            source=evidence,
            paths=final_paths,
            digest_paths=staged_paths,
        )
        staged_bundle_path = (
            staging_directory / "evaluation" / "release_evidence_bundle.json"
        )
        atomic_write_json(
            path=staged_bundle_path,
            payload=final_evidence.to_payload(),
        )
        fsync_tree(staging_directory)

        os.replace(staging_directory, final_directory)
        fsync_tree(final_directory)
        fsync_directory(final_directory.parent)

        final_bundle_path = (
            final_directory / "evaluation" / "release_evidence_bundle.json"
        )
        published_evidence = ReleaseEvidenceBundle.load(
            final_bundle_path,
            approved_roots=(final_directory,),
        )
        _validate_evidence_bundle(
            published_evidence,
            reproducibility_requirements=reproducibility_requirements,
        )

        report = AcceptanceReport.build(
            release_stage="production_model",
            decision=decision,
            expected_status=ReleaseStatus.MODEL_ACCEPTED,
            evidence_paths={
                **published_evidence.evidence_paths,
                "release_evidence_bundle": final_bundle_path,
                "training_metrics": final_metrics_path,
            },
        )
        report_path = final_directory / "evaluation" / "acceptance_report.json"
        atomic_write_json(path=report_path, payload=report.to_payload())
        if not report.to_payload()["passed"]:
            raise ProductionPromotionValidationError(
                "published production evidence is incomplete"
            )

        manifest_path = final_directory / "release_manifest.json"
        atomic_write_json(
            path=manifest_path,
            payload=_manifest_payload(
                release_id=release_id,
                evidence=published_evidence,
                evidence_bundle_path=final_bundle_path,
                acceptance_report_path=report_path,
                release_directory=final_directory,
                reproducibility_requirements=reproducibility_requirements,
            ),
        )
        fsync_tree(final_directory)
        _validate_manifest(
            release_directory=final_directory,
            expected_release_id=release_id,
        )

        _write_current_pointer(
            production_directory=production_directory,
            release_id=release_id,
            final_directory=final_directory,
            manifest_path=manifest_path,
        )
        if resolve_current_release(production_directory) != final_directory:
            raise ProductionPromotionValidationError(
                "published production pointer did not resolve"
            )
        return report
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory, ignore_errors=True)
        if (
            final_directory is not None
            and final_directory.exists()
            and not current_pointer_references(
                production_directory=production_directory,
                release_directory=final_directory,
            )
        ):
            shutil.rmtree(final_directory, ignore_errors=True)


def _release_id_for(
    *,
    checkpoint_sha256: str,
    metricssha256: str,
    evidence: ReleaseEvidenceBundle,
) -> str:
    """Return the deterministic release id for one promoted candidate.

    The id is derived entirely from content digests (checkpoint, metrics,
    and the verified evidence bundle), so retrying the same promotion
    resolves to the same directory and stays idempotent.
    """

    evidence_digest = hashlib.sha256(
        json.dumps(
            evidence.to_payload(),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return (
        "release-"
        f"{checkpoint_sha256[:16]}-"
        f"{metricssha256[:12]}-"
        f"{evidence_digest[:12]}"
    )


def _load_existing_release(
    *,
    production_directory: Path,
    releases_directory: Path,
    final_directory: Path,
    release_id: str,
) -> AcceptanceReport:
    """Reuse one already published release idempotently.

    A retry of a successful promotion finds the deterministic release
    directory already present. Its manifest and acceptance report are
    validated and the current pointer is restored when needed, without
    re-copying any candidate artifact.
    """

    _validate_manifest(
        release_directory=final_directory,
        expected_release_id=release_id,
    )
    report_path = final_directory / "evaluation" / "acceptance_report.json"
    report = AcceptanceReport.load(report_path)
    manifest_path = final_directory / "release_manifest.json"
    if not current_pointer_references(
        production_directory=production_directory,
        release_directory=final_directory,
    ):
        _write_current_pointer(
            production_directory=production_directory,
            release_id=release_id,
            final_directory=final_directory,
            manifest_path=manifest_path,
        )
    if resolve_current_release(production_directory) != final_directory:
        raise ProductionPromotionValidationError(
            "existing production pointer did not resolve"
        )
    return report


def _write_current_pointer(
    *,
    production_directory: Path,
    release_id: str,
    final_directory: Path,
    manifest_path: Path,
) -> None:
    atomic_write_json(
        path=production_directory / CURRENT_POINTER,
        payload={
            "schema_version": POINTER_SCHEMA,
            "release_id": release_id,
            "release_directory": (
                final_directory.relative_to(production_directory).as_posix()
            ),
            "release_manifest": manifest_path.relative_to(
                final_directory
            ).as_posix(),
            "release_manifest_sha256": sha256(manifest_path),
        },
    )


def _copy_candidate_tree(*, source_root: Path, target_root: Path) -> None:
    for source in sorted(source_root.rglob("*")):
        if source.is_symlink():
            raise ValueError(f"candidate release contains a symlink: {source}")
        relative = source.relative_to(source_root)
        target = target_root / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            _copy_regular_file(source=source, target=target)
        else:
            raise ValueError(
                f"candidate release contains an unsupported entry: {source}"
            )


__all__ = [
    "ProductionPromotionLockError",
    "ProductionPromotionValidationError",
    "decide_release",
    "promote_model",
]
