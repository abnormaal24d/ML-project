"""Verify privacy clearance before a sample can enter training."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mmcrawler_datasets.training_samples.models import TrainingSample
from preprocessing.privacy.clearance import (
    ApprovedObjectRole,
    PrivacyClearance,
)
from preprocessing.privacy.text_privacy import PiiDetector


@dataclass(frozen=True, slots=True)
class ClearanceVerification:
    """Outcome of checking one sample against its bound clearance."""

    valid: bool
    reasons: tuple[str, ...] = ()


def verify_sample_clearance(
    sample: TrainingSample,
    *,
    project_root: Path,
    residual_pii_detector: PiiDetector,
) -> ClearanceVerification:
    """Verify completeness, approved text, and media-byte integrity."""

    project_root = project_root.resolve(strict=True)
    clearance = sample.privacy_clearance
    if clearance is None:
        return ClearanceVerification(False, ("privacy_clearance_missing",))

    reasons: list[str] = []
    if not clearance.permits_training:
        reasons.append("privacy_clearance_not_approved")
    if not clearance.inspection_digest:
        reasons.append("inspection_digest_missing")
    if not clearance.assessment_digest:
        reasons.append("assessment_digest_missing")
    if not _text_is_approved(sample, clearance):
        reasons.append("training_text_not_approved")

    for field_path, value in _iter_strings(sample.to_dict()):
        result = residual_pii_detector.detect(text=value)
        if not result.inspection_complete:
            reasons.append(f"residual_privacy_incomplete:{field_path}")
        elif result.has_findings:
            reasons.append(f"residual_pii:{field_path}")

    primary_digests = frozenset(
        approved.output_digest
        for approved in clearance.approved_objects
        if approved.role is ApprovedObjectRole.PRIMARY_MEDIA
    )
    for training_object in sample.objects:
        actual_digest = training_object.current_digest()
        if actual_digest is None:
            reasons.append("cleared_media_unavailable")
            continue
        if actual_digest != training_object.object_sha256:
            reasons.append("cleared_media_digest_mismatch")
            continue
        approved = clearance.approved_object(
            object_id=training_object.object_id,
            role=training_object.role,
        )
        if approved is None:
            reasons.append("object_clearance_missing")
            continue
        if actual_digest != approved.output_digest:
            reasons.append("cleared_media_digest_mismatch")
            continue
        if training_object.role is ApprovedObjectRole.PRIMARY_MEDIA:
            if approved.derived_from_digest is None:
                if training_object.derived_from_sha256 is not None:
                    reasons.append("primary_object_provenance_invalid")
            elif (
                clearance.derivation_digest is None
                or training_object.derived_from_sha256
                != approved.derived_from_digest
                or approved.derived_from_digest != clearance.input_digest
                or approved.output_digest != clearance.output_digest
            ):
                reasons.append("primary_object_provenance_invalid")
        elif (
            approved.derived_from_digest is None
            or approved.derived_from_digest not in primary_digests
            or training_object.derived_from_sha256
            != approved.derived_from_digest
        ):
            reasons.append("derived_object_provenance_invalid")

    for target_path, target_role in _target_bindings(sample):
        actual_digest = _file_digest(
            target_path,
            project_root=project_root,
        )
        target_approval = (
            clearance.approved_object(
                object_id=sample.object_id,
                role=target_role,
            )
            if sample.object_id
            else None
        )
        if target_approval is None:
            reasons.append("target_clearance_missing")
        elif actual_digest is None:
            reasons.append("cleared_media_unavailable")
        elif actual_digest != target_approval.output_digest:
            reasons.append("cleared_media_digest_mismatch")

    return ClearanceVerification(not reasons, tuple(dict.fromkeys(reasons)))


def _text_is_approved(
    sample: TrainingSample,
    clearance: PrivacyClearance,
) -> bool:
    if not sample.text:
        return False
    approved = clearance.approved_text("training_text")
    return approved is not None and sample.text == approved


def _target_bindings(
    sample: TrainingSample,
) -> tuple[tuple[str, ApprovedObjectRole], ...]:
    bindings: list[tuple[str, ApprovedObjectRole]] = []
    target = sample.task_target
    for path, role in (
        (target.target_image_path, ApprovedObjectRole.PRIMARY_MEDIA),
        (target.target_audio_path, ApprovedObjectRole.PRIMARY_MEDIA),
        (target.target_video_path, ApprovedObjectRole.PRIMARY_MEDIA),
        (target.source_image_path, ApprovedObjectRole.EDIT_SOURCE),
        (target.edit_mask_path, ApprovedObjectRole.EDIT_MASK),
    ):
        if path:
            bindings.append((path, role))
    return tuple(dict.fromkeys(bindings))


def _iter_strings(
    value: object,
    path: str = "sample",
) -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        if value.strip():
            yield path, value
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_strings(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            yield from _iter_strings(child, f"{path}[{index}]")


def _file_digest(path: str, *, project_root: Path) -> str | None:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    try:
        root = project_root.resolve(strict=True)
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["ClearanceVerification", "verify_sample_clearance"]
