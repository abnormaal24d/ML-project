"""Release manifest payload generation and validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from config.releases.release_requirements import ReproducibilityRequirements
from release.release_evidence_bundle import ReleaseEvidenceBundle
from release.release_utilities import (
    RELEASE_SCHEMA,
    ProductionPromotionValidationError,
    contained_relative_path,
    read_json_object,
    required_sha256,
    required_string,
    sha256,
)


def _manifest_payload(
    *,
    release_id: str,
    evidence: ReleaseEvidenceBundle,
    evidence_bundle_path: Path,
    acceptance_report_path: Path,
    release_directory: Path,
    reproducibility_requirements: ReproducibilityRequirements | None = None,
) -> dict[str, object]:
    """Build the release manifest payload dict."""
    artifacts: dict[str, tuple[Path, str]] = {
        reference.name: (
            Path(reference.path),
            reference.sha256,
        )
        for reference in evidence.references
    }
    artifacts.update(
        {
            "release_evidence_bundle": (
                evidence_bundle_path,
                sha256(evidence_bundle_path),
            ),
            "acceptance_report": (
                acceptance_report_path,
                sha256(acceptance_report_path),
            ),
        }
    )
    reproducibility_policy = (
        reproducibility_requirements.to_payload()
        if reproducibility_requirements is not None
        else None
    )
    return {
        "schema_version": RELEASE_SCHEMA,
        "release_id": release_id,
        "release_mode": evidence.release_mode,
        "release_requirements_id": (
            reproducibility_requirements.policy_id
            if reproducibility_requirements is not None
            else None
        ),
        "release_requirements_sha256": (
            reproducibility_requirements.policy_sha256
            if reproducibility_requirements is not None
            else None
        ),
        "reproducibility_policy": reproducibility_policy,
        "artifacts": [
            {
                "name": name,
                "path": path.relative_to(release_directory).as_posix(),
                "sha256": sha256,
            }
            for name, (path, sha256) in sorted(artifacts.items())
        ],
    }


def _validate_manifest(
    *,
    release_directory: Path,
    expected_release_id: str,
) -> None:
    """Validate manifest structure and digest against directory contents."""
    manifest_path = release_directory / "release_manifest.json"
    manifest = read_json_object(manifest_path)
    if set(manifest) != {
        "schema_version",
        "release_id",
        "release_mode",
        "release_requirements_id",
        "release_requirements_sha256",
        "reproducibility_policy",
        "artifacts",
    }:
        raise ProductionPromotionValidationError(
            "production release manifest fields are invalid"
        )
    if manifest.get("schema_version") != RELEASE_SCHEMA:
        raise ProductionPromotionValidationError(
            "production release manifest schema is invalid"
        )
    if manifest.get("release_id") != expected_release_id:
        raise ProductionPromotionValidationError(
            "production release identifier mismatch"
        )
    if manifest.get("release_mode") != "production_model":
        raise ProductionPromotionValidationError(
            "production release mode is invalid"
        )

    release_requirements_id = manifest.get("release_requirements_id")
    release_requirements_sha256 = manifest.get("release_requirements_sha256")
    reproducibility_policy = manifest.get("reproducibility_policy")
    if (
        not isinstance(release_requirements_id, str)
        or not release_requirements_id.strip()
    ):
        raise ProductionPromotionValidationError(
            "production release manifest lacks release policy identity"
        )
    if not isinstance(release_requirements_sha256, str):
        raise ProductionPromotionValidationError(
            "production release manifest lacks release policy digest"
        )
    required_sha256(manifest, "release_requirements_sha256")
    if not isinstance(reproducibility_policy, dict):
        raise ProductionPromotionValidationError(
            "production release manifest lacks reproducibility policy"
        )
    _validate_reproducibility_policy(
        release_requirements_id=release_requirements_id,
        release_requirements_sha256=release_requirements_sha256,
        reproducibility_policy=reproducibility_policy,
    )

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ProductionPromotionValidationError(
            "production release artifacts are invalid"
        )
    artifact_names: set[str] = set()
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, dict):
            raise ProductionPromotionValidationError(
                "production release artifact entry is invalid"
            )
        if set(raw_artifact) != {"name", "path", "sha256"}:
            raise ProductionPromotionValidationError(
                "production release artifact fields are invalid"
            )
        name = required_string(raw_artifact, "name")
        if name in artifact_names:
            raise ProductionPromotionValidationError(
                "production release artifact names are duplicated"
            )
        artifact_names.add(name)
        artifact_path = contained_relative_path(
            root=release_directory,
            relative=required_string(raw_artifact, "path"),
        )
        if not artifact_path.is_file():
            raise ProductionPromotionValidationError(
                f"production release artifact is missing: {name}"
            )
        if sha256(artifact_path) != required_sha256(
            raw_artifact,
            "sha256",
        ):
            raise ProductionPromotionValidationError(
                f"production release artifact digest mismatch: {name}"
            )

    required = {
        "checkpoint",
        "leakage",
        "dataset_card",
        "model_card",
        "reproducibility",
        "run_receipts",
        "release_evidence_bundle",
        "acceptance_report",
        "training_metrics",
    }
    if not required.issubset(artifact_names):
        raise ProductionPromotionValidationError(
            "production release manifest is incomplete"
        )


def _validate_reproducibility_policy(
    *,
    release_requirements_id: str,
    release_requirements_sha256: str,
    reproducibility_policy: dict[str, object],
) -> None:
    """Fail closed when the recorded policy is not cryptographically bound."""
    policy = _reproducibility_requirements_from_manifest(
        {
            "release_requirements_id": release_requirements_id,
            "reproducibility_policy": reproducibility_policy,
        }
    )
    if policy.policy_sha256 != release_requirements_sha256:
        raise ProductionPromotionValidationError(
            "production release policy digest mismatch"
        )
    if policy.to_payload() != reproducibility_policy:
        raise ProductionPromotionValidationError(
            "production release policy payload mismatch"
        )


def _reproducibility_requirements_from_manifest(
    manifest: Mapping[str, object],
) -> ReproducibilityRequirements:
    """Reconstruct the trusted reproducibility policy recorded in a manifest.

    The manifest is already digest-verified before this is called, so the
    recorded policy is the authoritative promotion-time contract.
    """

    release_requirements_id = manifest.get("release_requirements_id")
    reproducibility_policy = manifest.get("reproducibility_policy")
    if (
        not isinstance(release_requirements_id, str)
        or not release_requirements_id.strip()
    ):
        raise ProductionPromotionValidationError(
            "production release manifest lacks release policy identity"
        )
    if not isinstance(reproducibility_policy, dict):
        raise ProductionPromotionValidationError(
            "production release manifest lacks reproducibility policy"
        )
    raw_seeds = reproducibility_policy.get("seeds")
    if not isinstance(raw_seeds, list) or not all(
        isinstance(seed, int) and not isinstance(seed, bool)
        for seed in raw_seeds
    ):
        raise ProductionPromotionValidationError(
            "production release reproducibility policy seeds are invalid"
        )
    deterministic = reproducibility_policy.get(
        "require_deterministic_execution"
    )
    if not isinstance(deterministic, bool):
        raise ProductionPromotionValidationError(
            "production release reproducibility policy determinism is invalid"
        )
    raw_tolerances = reproducibility_policy.get("metric_tolerances")
    if not isinstance(raw_tolerances, dict) or not all(
        isinstance(name, str)
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        for name, value in raw_tolerances.items()
    ):
        raise ProductionPromotionValidationError(
            "production release reproducibility policy tolerances are invalid"
        )
    return ReproducibilityRequirements(
        policy_id=release_requirements_id,
        seeds=tuple(int(seed) for seed in raw_seeds),
        require_deterministic_execution=deterministic,
        metric_tolerances={
            str(name): float(value) for name, value in raw_tolerances.items()
        },
    )


__all__ = [
    "_manifest_payload",
    "_reproducibility_requirements_from_manifest",
    "_validate_manifest",
]
