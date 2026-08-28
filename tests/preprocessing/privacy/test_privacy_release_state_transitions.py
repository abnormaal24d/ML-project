"""Characterization of the media privacy clearance state machine.

These tests pin the *current* transition behavior of
``inspect_media_privacy`` before any ownership refactor.  In particular they
document that the clearance status is overwritten by later checks in the
coordinator (e.g. a text REJECTED clearance can become INCOMPLETE when the
media subject turns out to be unavailable).  A dedicated state-machine
correction may then change these exact outcomes in ``clearance.py`` only.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from config.preprocessing.text_settings import PrivacyDetectionSettings
from preprocessing.media.privacy_inspection import inspect_media_privacy
from preprocessing.preprocessing_input import (
    LanguageEvidence,
    PreprocessingInput,
)
from preprocessing.privacy.artifacts import (
    PrivacyArtifactWorkspace,
    PublishedPrivacyArtifact,
    build_receipt,
    canonical_sha256,
    file_sha256,
    write_exclusive_bytes,
)
from preprocessing.privacy.clearance import (
    ApprovedObjectRole,
    PrivacyClearanceStatus,
)
from preprocessing.privacy.inspection.detector import DetectorRun
from preprocessing.privacy.inspection.detector_registry import DetectorRegistry
from preprocessing.privacy.inspection.evidence_location import (
    BoundingBox,
    EvidenceLocation,
)
from preprocessing.privacy.inspection.finding import PrivacyFinding
from preprocessing.privacy.inspection.finding_type import FindingType
from preprocessing.privacy.inspection.inspection_coverage import (
    InspectionCoverage,
)
from preprocessing.privacy.inspection.inspection_result import InspectionResult
from preprocessing.privacy.text_privacy import PiiDetector
from tests.support.privacy import build_test_pii_detector


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _item(
    path: Path, *, payload: dict[str, object] | None = None
) -> PreprocessingInput:
    return PreprocessingInput(
        source_id="item",
        source_url="https://example.test/item.png",
        normalized_url="https://example.test/item.png",
        domain="example.test",
        path="/item.png",
        language_evidence=LanguageEvidence(language="en"),
        title="Public item",
        modality="image",
        mime_type="image/png",
        media_path=str(path),
        byte_size=path.stat().st_size if path.exists() else 0,
        payload={"caption_text": "Public scene", **(payload or {})},
    )


def _finding(
    finding_type: FindingType,
    *,
    field_name: str,
    bounding_box: BoundingBox | None = None,
) -> PrivacyFinding:
    return PrivacyFinding(
        finding_id=f"{finding_type.value}-{field_name}",
        finding_type=finding_type,
        confidence=0.99,
        location=EvidenceLocation(
            field_name=field_name,
            bounding_box=bounding_box,
        ),
        detector_name="characterization",
        detector_version="1",
    )


def _inspection(
    path: Path,
    *,
    findings: tuple[PrivacyFinding, ...] = (),
    checked_fields: frozenset[str] = frozenset(),
    required_fields: frozenset[str] = frozenset(),
    completed: bool = True,
) -> InspectionResult:
    return InspectionResult(
        subject_digest=(
            _sha(path.read_bytes()) if path.exists() else _sha(b"missing")
        ),
        findings=findings,
        coverage=InspectionCoverage(
            checked_fields=checked_fields,
            required_fields=required_fields,
        ),
        detector_runs=(
            DetectorRun(
                detector_name="char",
                detector_version="1",
                completed=True,
                finding_count=0,
                elapsed_ms=0,
            ),
        ),
        completed=completed,
    )


def _publish(source: Path, *, content: bytes) -> PublishedPrivacyArtifact:
    """Publish one derivative whose receipt binds the exact source bytes."""
    workspace = PrivacyArtifactWorkspace(
        source_path=source,
        stage="sanitized",
        run_id=f"state-transition-{content.hex()[:8]}",
    )
    temporary = workspace.new_bytes_temp(suffix=".bin")
    write_exclusive_bytes(temporary, content)
    digest = file_sha256(temporary)
    receipt = build_receipt(
        workspace=workspace,
        source_path=source,
        source_sha256=workspace.source_snapshot.sha256,
        transform_input_sha256=workspace.source_snapshot.sha256,
        output_path=temporary,
        output_sha256=digest,
        source_mime_type="application/octet-stream",
        output_mime_type="application/octet-stream",
        transform_id="characterization",
        transform_version="1",
        transform_artifact_path=Path(__file__),
        configuration={"mode": "characterization"},
        residual_inspection_sha256=canonical_sha256({"clean": True}),
        created_at=datetime(2026, 8, 26, 9, 0, tzinfo=UTC),
    )
    artifact = workspace.publish(
        temporary_path=temporary,
        receipt=receipt,
        final_name=f"source.{content.hex()[:4]}.bin",
    )
    workspace.cleanup()
    return artifact


def _media_clean_inspection(path: Path) -> InspectionResult:
    return _inspection(
        path,
        checked_fields=frozenset({"media_decode", "visual_analysis"}),
        required_fields=frozenset({"media_decode", "visual_analysis"}),
    )


def _residual_clean_inspection(
    artifact: PublishedPrivacyArtifact,
) -> InspectionResult:
    return InspectionResult(
        subject_digest=artifact.sha256,
        findings=(),
        coverage=InspectionCoverage(
            checked_fields=frozenset({"residual_check"}),
            required_fields=frozenset({"residual_check"}),
        ),
        detector_runs=(
            DetectorRun(
                detector_name="char",
                detector_version="1",
                completed=True,
                finding_count=0,
                elapsed_ms=0,
            ),
        ),
        completed=True,
    )


def _publish_fresh(
    tmp_path: Path, *, content: bytes
) -> PublishedPrivacyArtifact:
    """Publish one derivative bound to a fresh source with the same bytes."""
    source = tmp_path / "source.bin"
    source.write_bytes(content)
    return _publish(source, content=content)


def _invoke(
    *,
    item: PreprocessingInput,
    inspection: InspectionResult,
    media_path: str,
    fields: dict[str, str] | None = None,
    inspected_artifact: PublishedPrivacyArtifact | None = None,
    remediated_artifact: PublishedPrivacyArtifact | None = None,
    residual_inspection: InspectionResult | None = None,
    detector=None,
):
    return inspect_media_privacy(
        item=item,
        object_id=item.source_id,
        detector=detector or build_test_pii_detector(),
        fields=fields or {"caption_text": "Public scene"},
        inspection=inspection,
        media_path=media_path,
        content_field_prefixes=("ocr_text", "transcript"),
        inspected_artifact=inspected_artifact,
        remediated_artifact=remediated_artifact,
        residual_inspection=residual_inspection,
    )


def test_text_rejected_then_media_unavailable_stays_rejected(
    tmp_path: Path,
) -> None:
    media = tmp_path / "missing.png"
    item = _item(media, payload={"caption_text": "4111 1111 1111 1111"})

    result = _invoke(
        item=item,
        inspection=_media_clean_inspection(media),
        media_path=str(media),
        fields={"caption_text": "4111 1111 1111 1111"},
    )

    assert "restricted_finding:caption_text" in result.clearance.reasons
    assert result.clearance.status is PrivacyClearanceStatus.REJECTED
    assert "media_subject_unavailable" in result.clearance.reasons
    assert not result.clearance.permits_training


def test_text_review_required_then_media_unavailable_stays_review_required(
    tmp_path: Path,
) -> None:
    media = tmp_path / "missing.png"
    item = _item(media)
    text = "The patient's diagnosis requires immediate treatment"

    result = _invoke(
        item=item,
        inspection=_media_clean_inspection(media),
        media_path=str(media),
        fields={"caption_text": text},
    )

    assert "review_required:caption_text" in result.clearance.reasons
    assert result.clearance.status is PrivacyClearanceStatus.REVIEW_REQUIRED
    assert "media_subject_unavailable" in result.clearance.reasons
    assert not result.clearance.permits_training


def test_approved_fields_then_media_unavailable_is_incomplete(
    tmp_path: Path,
) -> None:
    media = tmp_path / "missing.png"
    item = _item(media)

    result = _invoke(
        item=item,
        inspection=_media_clean_inspection(media),
        media_path=str(media),
    )

    assert result.clearance.status is PrivacyClearanceStatus.INCOMPLETE
    assert "media_subject_unavailable" in result.clearance.reasons
    assert not result.clearance.permits_training


def test_text_remediated_then_media_unavailable_becomes_incomplete(
    tmp_path: Path,
) -> None:
    media = tmp_path / "missing.png"
    item = _item(
        media, payload={"caption_text": "Contact jan.peeters@example.com"}
    )

    result = _invoke(
        item=item,
        inspection=_media_clean_inspection(media),
        media_path=str(media),
        fields={"caption_text": "Contact jan.peeters@example.com"},
    )

    assert "residual_findings" not in " ".join(result.clearance.reasons)
    assert result.clearance.status is PrivacyClearanceStatus.INCOMPLETE
    assert not result.clearance.permits_training


def test_incomplete_text_then_policy_reject_upgrades_to_rejected(
    tmp_path: Path,
) -> None:
    media = tmp_path / "item.png"
    media.write_bytes(b"source bytes")
    item = _item(media)
    inspection = InspectionResult(
        subject_digest=_sha(media.read_bytes()),
        findings=(
            _finding(FindingType.IDENTITY_DOCUMENT, field_name="visual"),
        ),
        coverage=InspectionCoverage(
            checked_fields=frozenset({"media_decode", "visual_analysis"}),
            required_fields=frozenset({"media_decode", "visual_analysis"}),
        ),
        detector_runs=(
            DetectorRun(
                detector_name="char",
                detector_version="1",
                completed=True,
                finding_count=0,
                elapsed_ms=0,
            ),
        ),
        completed=True,
    )

    incomplete_detector = PiiDetector(
        settings=PrivacyDetectionSettings(),
        registry=DetectorRegistry(text_detectors=()),
    )

    result = _invoke(
        item=item,
        inspection=inspection,
        media_path=str(media),
        fields={"caption_text": "Public scene"},
        detector=incomplete_detector,
    )

    assert "inspection_incomplete:caption_text" in result.clearance.reasons
    assert result.clearance.status is PrivacyClearanceStatus.REJECTED
    assert result.rejection_reason == "identity_document_detected"


def test_review_required_then_policy_reject_upgrades_to_rejected(
    tmp_path: Path,
) -> None:
    media = tmp_path / "item.png"
    media.write_bytes(b"source bytes")
    item = _item(media)
    inspection = InspectionResult(
        subject_digest=_sha(media.read_bytes()),
        findings=(
            _finding(FindingType.IDENTITY_DOCUMENT, field_name="visual"),
        ),
        coverage=InspectionCoverage(
            checked_fields=frozenset({"media_decode", "visual_analysis"}),
            required_fields=frozenset({"media_decode", "visual_analysis"}),
        ),
        detector_runs=(
            DetectorRun(
                detector_name="char",
                detector_version="1",
                completed=True,
                finding_count=0,
                elapsed_ms=0,
            ),
        ),
        completed=True,
    )

    result = _invoke(
        item=item,
        inspection=inspection,
        media_path=str(media),
        fields={"caption_text": "The patient's diagnosis requires treatment"},
    )

    assert result.clearance.status is PrivacyClearanceStatus.REJECTED
    assert result.rejection_reason == "identity_document_detected"


def test_policy_rejection_precedes_text_rejection_in_reason(
    tmp_path: Path,
) -> None:
    media = tmp_path / "item.png"
    media.write_bytes(b"source bytes")
    item = _item(media)
    inspection = InspectionResult(
        subject_digest=_sha(media.read_bytes()),
        findings=(
            _finding(FindingType.IDENTITY_DOCUMENT, field_name="visual"),
        ),
        coverage=InspectionCoverage(
            checked_fields=frozenset({"media_decode", "visual_analysis"}),
            required_fields=frozenset({"media_decode", "visual_analysis"}),
        ),
        detector_runs=(
            DetectorRun(
                detector_name="char",
                detector_version="1",
                completed=True,
                finding_count=0,
                elapsed_ms=0,
            ),
        ),
        completed=True,
    )

    result = _invoke(
        item=item,
        inspection=inspection,
        media_path=str(media),
        fields={"caption_text": "4111 1111 1111 1111"},
    )

    assert result.clearance.status is PrivacyClearanceStatus.REJECTED
    assert result.rejection_reason == "identity_document_detected"
    assert "restricted_finding:caption_text" in result.clearance.reasons


def test_approved_then_artifact_receipt_invalid_is_rejected(
    tmp_path: Path,
) -> None:
    artifact = _publish_fresh(tmp_path, content=b"artifact bytes")
    item = _item(artifact.path)
    artifact.receipt_path.chmod(0o600)
    artifact.receipt_path.write_bytes(b"{}")

    result = _invoke(
        item=item,
        inspection=_media_clean_inspection(artifact.path),
        media_path=str(artifact.path),
        inspected_artifact=artifact,
    )

    assert result.clearance.status is PrivacyClearanceStatus.REJECTED
    assert "media_derivation_receipt_invalid" in result.clearance.reasons
    assert not result.clearance.permits_training


def test_approved_then_artifact_output_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    artifact = _publish_fresh(tmp_path, content=b"artifact bytes")
    other = tmp_path / "other.png"
    other.write_bytes(b"other bytes")
    item = _item(artifact.path)

    result = _invoke(
        item=item,
        inspection=_media_clean_inspection(other),
        media_path=str(other),
        inspected_artifact=artifact,
    )

    assert result.clearance.status is PrivacyClearanceStatus.REJECTED
    assert "media_derivative_digest_mismatch" in result.clearance.reasons
    assert not result.clearance.permits_training


def test_metadata_derived_artifact_with_clean_media_stays_approved(
    tmp_path: Path,
) -> None:
    artifact = _publish_fresh(tmp_path, content=b"artifact bytes")
    item = _item(artifact.path)

    result = _invoke(
        item=item,
        inspection=_media_clean_inspection(artifact.path),
        media_path=str(artifact.path),
        inspected_artifact=artifact,
    )

    assert result.clearance.status is PrivacyClearanceStatus.APPROVED
    assert result.clearance.permits_training
    assert result.media_path == str(artifact.path)
    approved = result.clearance.approved_object(
        object_id=item.source_id,
        role=ApprovedObjectRole.PRIMARY_MEDIA,
    )
    assert approved is not None
    assert approved.output_digest == artifact.sha256
    assert approved.derived_from_digest is None


def test_content_findings_with_valid_residual_become_remediated(
    tmp_path: Path,
) -> None:
    media = tmp_path / "item.png"
    media.write_bytes(b"source bytes")
    item = _item(media)
    inspection = InspectionResult(
        subject_digest=_sha(media.read_bytes()),
        findings=(
            _finding(
                FindingType.PERSON_NAME,
                field_name="ocr_text",
                bounding_box=BoundingBox(0, 0, 10, 10),
            ),
        ),
        coverage=InspectionCoverage(
            checked_fields=frozenset({"media_decode", "visual_analysis"}),
            required_fields=frozenset({"media_decode", "visual_analysis"}),
        ),
        detector_runs=(
            DetectorRun(
                detector_name="char",
                detector_version="1",
                completed=True,
                finding_count=0,
                elapsed_ms=0,
            ),
        ),
        completed=True,
    )
    remediated = _publish(media, content=b"remediated bytes")

    result = _invoke(
        item=item,
        inspection=inspection,
        media_path=str(media),
        remediated_artifact=remediated,
        residual_inspection=_residual_clean_inspection(remediated),
    )

    assert result.clearance.status is PrivacyClearanceStatus.REMEDIATED
    assert result.clearance.remediation_verified
    assert result.clearance.derivation_digest == remediated.derivation_sha256
    assert result.media_path == str(remediated.path)
    approved = result.clearance.approved_object(
        object_id=item.source_id,
        role=ApprovedObjectRole.PRIMARY_MEDIA,
    )
    assert approved is not None
    assert approved.output_digest == remediated.sha256
    assert approved.derived_from_digest == _sha(media.read_bytes())


def test_content_findings_with_invalid_residual_are_rejected(
    tmp_path: Path,
) -> None:
    media = tmp_path / "item.png"
    media.write_bytes(b"source bytes")
    item = _item(media)
    inspection = InspectionResult(
        subject_digest=_sha(media.read_bytes()),
        findings=(
            _finding(
                FindingType.PERSON_NAME,
                field_name="ocr_text",
                bounding_box=BoundingBox(0, 0, 10, 10),
            ),
        ),
        coverage=InspectionCoverage(
            checked_fields=frozenset({"media_decode", "visual_analysis"}),
            required_fields=frozenset({"media_decode", "visual_analysis"}),
        ),
        detector_runs=(
            DetectorRun(
                detector_name="char",
                detector_version="1",
                completed=True,
                finding_count=0,
                elapsed_ms=0,
            ),
        ),
        completed=True,
    )
    remediated = _publish(media, content=b"remediated bytes")
    residual = InspectionResult(
        subject_digest=remediated.sha256,
        findings=(
            _finding(FindingType.PERSON_NAME, field_name="residual_field"),
        ),
        coverage=InspectionCoverage(
            checked_fields=frozenset({"residual_check"}),
            required_fields=frozenset({"residual_check"}),
        ),
        detector_runs=(
            DetectorRun(
                detector_name="char",
                detector_version="1",
                completed=True,
                finding_count=0,
                elapsed_ms=0,
            ),
        ),
        completed=True,
    )

    result = _invoke(
        item=item,
        inspection=inspection,
        media_path=str(media),
        remediated_artifact=remediated,
        residual_inspection=residual,
    )

    assert result.clearance.status is PrivacyClearanceStatus.REJECTED
    assert (
        "media_contains_unremediated_private_data" in result.clearance.reasons
    )
    assert not result.clearance.permits_training


def test_content_findings_without_remediated_artifact_are_rejected(
    tmp_path: Path,
) -> None:
    media = tmp_path / "item.png"
    media.write_bytes(b"source bytes")
    item = _item(media)
    inspection = InspectionResult(
        subject_digest=_sha(media.read_bytes()),
        findings=(
            _finding(
                FindingType.PERSON_NAME,
                field_name="ocr_text",
                bounding_box=BoundingBox(0, 0, 10, 10),
            ),
        ),
        coverage=InspectionCoverage(
            checked_fields=frozenset({"media_decode", "visual_analysis"}),
            required_fields=frozenset({"media_decode", "visual_analysis"}),
        ),
        detector_runs=(
            DetectorRun(
                detector_name="char",
                detector_version="1",
                completed=True,
                finding_count=0,
                elapsed_ms=0,
            ),
        ),
        completed=True,
    )

    result = _invoke(
        item=item,
        inspection=inspection,
        media_path=str(media),
    )

    assert result.clearance.status is PrivacyClearanceStatus.REJECTED
    assert (
        "media_contains_unremediated_private_data" in result.clearance.reasons
    )
    assert not result.clearance.permits_training


def test_text_review_required_blocks_remediation_binding(
    tmp_path: Path,
) -> None:
    media = tmp_path / "item.png"
    media.write_bytes(b"source bytes")
    item = _item(media)
    inspection = InspectionResult(
        subject_digest=_sha(media.read_bytes()),
        findings=(
            _finding(
                FindingType.PERSON_NAME,
                field_name="ocr_text",
                bounding_box=BoundingBox(0, 0, 10, 10),
            ),
        ),
        coverage=InspectionCoverage(
            checked_fields=frozenset({"media_decode", "visual_analysis"}),
            required_fields=frozenset({"media_decode", "visual_analysis"}),
        ),
        detector_runs=(
            DetectorRun(
                detector_name="char",
                detector_version="1",
                completed=True,
                finding_count=0,
                elapsed_ms=0,
            ),
        ),
        completed=True,
    )
    remediated = _publish(media, content=b"remediated bytes")

    result = _invoke(
        item=item,
        inspection=inspection,
        media_path=str(media),
        fields={"caption_text": "The patient's diagnosis requires treatment"},
        remediated_artifact=remediated,
        residual_inspection=_residual_clean_inspection(remediated),
    )

    assert result.clearance.status is PrivacyClearanceStatus.REVIEW_REQUIRED
    assert not result.clearance.permits_training
    assert result.clearance.derivation_digest is None
    assert result.media_path == str(remediated.path)


def test_remediated_clearance_round_trips_through_dict(tmp_path: Path) -> None:
    media = tmp_path / "item.png"
    media.write_bytes(b"source bytes")
    item = _item(media)
    inspection = InspectionResult(
        subject_digest=_sha(media.read_bytes()),
        findings=(
            _finding(
                FindingType.PERSON_NAME,
                field_name="ocr_text",
                bounding_box=BoundingBox(0, 0, 10, 10),
            ),
        ),
        coverage=InspectionCoverage(
            checked_fields=frozenset({"media_decode", "visual_analysis"}),
            required_fields=frozenset({"media_decode", "visual_analysis"}),
        ),
        detector_runs=(
            DetectorRun(
                detector_name="char",
                detector_version="1",
                completed=True,
                finding_count=0,
                elapsed_ms=0,
            ),
        ),
        completed=True,
    )
    remediated = _publish(media, content=b"remediated bytes")
    result = _invoke(
        item=item,
        inspection=inspection,
        media_path=str(media),
        remediated_artifact=remediated,
        residual_inspection=_residual_clean_inspection(remediated),
    )

    canonical_payload = result.clearance.to_dict()
    restored = type(result.clearance).from_dict(canonical_payload)
    retired_payload = dict(canonical_payload)
    retired_payload["derivation_receipt_digest"] = retired_payload.pop(
        "derivation_digest"
    )

    assert restored == result.clearance
    assert "derivation_digest" in canonical_payload
    assert "derivation_receipt_digest" not in canonical_payload
    with pytest.raises(ValueError, match="fields are incomplete"):
        type(result.clearance).from_dict(retired_payload)
    assert restored.status is PrivacyClearanceStatus.REMEDIATED
