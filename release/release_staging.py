"""Staging logic for release promotion."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from release.release_evidence_bundle import (
    EvidenceReference,
    ReleaseEvidenceBundle,
)
from release.release_utilities import (
    ProductionPromotionValidationError,
    safe_segment,
    sha256,
)
from training.runtime.checkpoint.io import (
    checkpoint_checksum_path,
    resolve_checkpoint_model_path,
)


def _stage_evidence(
    *,
    evidence: ReleaseEvidenceBundle,
    candidate_directory: Path,
    staging_directory: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    targets: set[Path] = set()
    for reference in evidence.references:
        source = Path(reference.path).resolve(strict=True)
        if reference.name == "checkpoint":
            target = staging_directory / "checkpoint.pt"
            _copy_checkpoint(source=source, target=target)
        elif reference.name == "training_metrics":
            target = staging_directory / "evaluation" / source.name
            _copy_regular_file(source=source, target=target)
        else:
            try:
                relative = source.relative_to(candidate_directory)
            except ValueError:
                relative = (
                    Path("evidence")
                    / safe_segment(reference.name)
                    / source.name
                )
            target = staging_directory / relative
            _copy_regular_file(source=source, target=target)

        resolved_target = target.resolve(strict=True)
        if resolved_target in targets:
            raise ProductionPromotionValidationError(
                f"multiple evidence references target {target}"
            )
        targets.add(resolved_target)
        paths[reference.name] = resolved_target
    return paths


def _copy_checkpoint(*, source: Path, target: Path) -> None:
    model_source = resolve_checkpoint_model_path(source)
    if model_source is None or not model_source.is_file():
        raise FileNotFoundError(f"checkpoint model is unavailable: {source}")
    checksum_source = checkpoint_checksum_path(source)
    if not checksum_source.is_file():
        raise FileNotFoundError(
            f"checkpoint checksum is unavailable: {checksum_source}"
        )

    _copy_regular_file(source=model_source, target=target)
    _copy_regular_file(
        source=checksum_source,
        target=target.with_name(target.name + ".sha256"),
    )
    expected = _source_sha256_from_sidecar(checksum_source)
    if sha256(target) != expected:
        raise ProductionPromotionValidationError(
            f"copied release checkpoint digest mismatch: {target}"
        )


def _release_checkpoint_is_available(model_path: Path) -> bool:
    """A release checkpoint is an immutable model file with a checksum sidecar."""
    return (
        model_path.is_file()
        and model_path.with_name(model_path.name + ".sha256").is_file()
    )


def _source_sha256_from_sidecar(sidecar: Path) -> str:
    parts = sidecar.read_text(encoding="ascii").strip().split()
    if not parts:
        raise ProductionPromotionValidationError(
            f"checkpoint checksum sidecar is empty: {sidecar}"
        )
    return parts[0].lower()


def _copy_regular_file(*, source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha256(source)
    if target.exists():
        if not target.is_file() or sha256(target) != source_sha256:
            raise ProductionPromotionValidationError(
                f"release artifact collision: {target}"
            )
        return
    shutil.copy2(source, target)
    with target.open("r+b") as handle:
        os.fsync(handle.fileno())
    if sha256(target) != source_sha256:
        raise ProductionPromotionValidationError(
            f"release artifact copy digest mismatch: {target}"
        )


def _bundle_for_paths(
    *,
    source: ReleaseEvidenceBundle,
    paths: dict[str, Path],
    digest_paths: dict[str, Path] | None = None,
) -> ReleaseEvidenceBundle:
    return ReleaseEvidenceBundle(
        release_mode=source.release_mode,
        references=tuple(
            EvidenceReference(
                name=reference.name,
                path=paths[reference.name].as_posix(),
                sha256=sha256((digest_paths or paths)[reference.name]),
            )
            for reference in source.references
        ),
        leakage_report=source.leakage_report,
        release_requirements_id=source.release_requirements_id,
        release_requirements_sha256=source.release_requirements_sha256,
    )


__all__ = [
    "_stage_evidence",
    "_copy_checkpoint",
    "_copy_regular_file",
    "_release_checkpoint_is_available",
    "_bundle_for_paths",
]
