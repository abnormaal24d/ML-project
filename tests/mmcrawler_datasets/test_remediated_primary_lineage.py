from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from mmcrawler_datasets.selection.privacy_clearance import (
    verify_sample_clearance,
)
from mmcrawler_datasets.training_samples.artifact_path import (
    ValidatedArtifactPath,
)
from mmcrawler_datasets.training_samples.models import (
    TrainingObject,
    TrainingSample,
)
from preprocessing.privacy.clearance import (
    ApprovedObject,
    ApprovedObjectRole,
    ApprovedTextField,
    PrivacyClearance,
    PrivacyClearanceStatus,
)
from tests.support.privacy import build_test_pii_detector


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sample(tmp_path: Path) -> TrainingSample:
    media = tmp_path / "artifact.bin"
    media.write_bytes(b"sanitized media")
    source_digest = _sha(b"source media")
    output_digest = _sha(b"sanitized media")
    text = "A public scientific image."
    text_digest = _sha(text.encode("utf-8"))
    clearance = PrivacyClearance(
        status=PrivacyClearanceStatus.REMEDIATED,
        input_digest=source_digest,
        output_digest=output_digest,
        checked_fields=frozenset({"body"}),
        required_fields=frozenset({"body"}),
        approved_text_fields=(
            ApprovedTextField(
                name="training_text",
                value=text,
                input_digest=text_digest,
                output_digest=text_digest,
            ),
        ),
        approved_objects=(
            ApprovedObject(
                object_id="media-1",
                role=ApprovedObjectRole.PRIMARY_MEDIA,
                output_digest=output_digest,
                derived_from_digest=source_digest,
            ),
        ),
        inspection_digest=_sha(b"inspection"),
        assessment_digest=_sha(b"assessment"),
        remediation_verified=True,
        derivation_digest=_sha(b"receipt"),
    )
    validated = ValidatedArtifactPath(
        relative_path="artifact.bin",
        resolved_path=media,
        project_root=tmp_path,
    )
    return TrainingSample(
        sample_id="sample-1",
        snapshot_id="snapshot-1",
        modality="image",
        object_id="media-1",
        text=text,
        objects=(
            TrainingObject(
                object_id="media-1",
                object_path=validated,
                object_sha256=output_digest,
                object_mime_type="application/octet-stream",
                role=ApprovedObjectRole.PRIMARY_MEDIA,
                derived_from_sha256=source_digest,
            ),
        ),
        privacy_clearance=clearance,
    )


def test_remediated_primary_lineage_is_accepted_end_to_end(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)

    result = verify_sample_clearance(
        sample,
        project_root=tmp_path,
        residual_pii_detector=build_test_pii_detector(),
    )

    assert result.valid
    assert result.reasons == ()


def test_remediated_primary_lineage_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)
    bad_object = replace(sample.objects[0], derived_from_sha256=None)
    sample = replace(sample, objects=(bad_object,))

    result = verify_sample_clearance(
        sample,
        project_root=tmp_path,
        residual_pii_detector=build_test_pii_detector(),
    )

    assert not result.valid
    assert "primary_object_provenance_invalid" in result.reasons
