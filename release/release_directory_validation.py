"""Release directory and report validation."""

from __future__ import annotations

from pathlib import Path

from config.releases.release_requirements import (
    ReproducibilityRequirements,
)
from release.release_decision import ReleaseStatus
from release.release_evidence_bundle import ReleaseEvidenceBundle
from release.release_manifest import _validate_manifest
from release.release_staging import _release_checkpoint_is_available
from release.release_utilities import (
    ProductionPromotionValidationError,
    read_json_object,
    required_string,
)


def _validate_release_directory(
    *,
    release_directory: Path,
    expected_release_id: str,
    reproducibility_requirements: ReproducibilityRequirements,
) -> None:
    """Validate full release directory contents including manifest and report."""
    release_directory = release_directory.resolve(strict=True)

    _validate_manifest(
        release_directory=release_directory,
        expected_release_id=expected_release_id,
    )

    bundle_path = (
        release_directory / "evaluation" / "release_evidence_bundle.json"
    )
    evidence = ReleaseEvidenceBundle.load(
        bundle_path,
        approved_roots=(release_directory,),
    )
    if evidence.release_requirements_id != (
        reproducibility_requirements.policy_id
    ):
        raise ProductionPromotionValidationError(
            "published release evidence is bound to another policy identity"
        )
    if evidence.release_requirements_sha256 != (
        reproducibility_requirements.policy_sha256
    ):
        raise ProductionPromotionValidationError(
            "published release evidence is bound to another policy digest"
        )
    _validate_evidence_bundle(
        evidence,
        reproducibility_requirements=reproducibility_requirements,
    )
    checkpoint = evidence.path("checkpoint")
    if checkpoint is None or not _release_checkpoint_is_available(checkpoint):
        raise ProductionPromotionValidationError(
            "published production checkpoint is incomplete"
        )

    report_path = release_directory / "evaluation" / "acceptance_report.json"
    report = read_json_object(report_path)

    if (
        report.get("passed") is not True
        or required_string(report, "release_stage") != "production_model"
        or required_string(report, "expected_status")
        != ReleaseStatus.MODEL_ACCEPTED.value
    ):
        raise ProductionPromotionValidationError(
            "production acceptance report is not passed"
        )


def _validate_evidence_bundle(
    evidence: ReleaseEvidenceBundle,
    *,
    reproducibility_requirements: ReproducibilityRequirements,
) -> None:
    from release.release_evidence_validation import check_release

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=reproducibility_requirements,
    )
    if not gate.passed:
        raise ProductionPromotionValidationError(
            "release evidence validation failed: " + ", ".join(gate.violations)
        )


__all__ = ["_validate_release_directory"]
