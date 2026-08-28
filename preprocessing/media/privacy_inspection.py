"""Local privacy release boundary shared by image, audio, and video.

Multimodal clearance is derived exclusively from ``InspectionResult`` values
created inside the preprocessing process by ``inspect_image``, ``inspect_audio``
or ``inspect_video``.  Generic payload dictionaries are metadata only and can
never declare privacy checks complete.
"""

from __future__ import annotations

from dataclasses import dataclass

from preprocessing.preprocessing_input import PreprocessingInput
from preprocessing.privacy.artifacts import (
    PrivacyArtifactOutputMismatch,
    PrivacyArtifactVerificationError,
    PublishedPrivacyArtifact,
    optional_file_sha256,
    verify_artifact_binding,
)
from preprocessing.privacy.clearance import (
    ApprovedObjectRole,
    PrivacyClearance,
)
from preprocessing.privacy.field_inspection import (
    inspect_media_fields_for_release,
)
from preprocessing.privacy.inspection.inspection_result import (
    InspectionResult,
    MediaAnalysisEvidence,
    media_analysis_evidence,
)
from preprocessing.privacy.text_privacy import PiiDetector


@dataclass(frozen=True, slots=True)
class MediaPrivacyResult:
    media_path: str
    fields: dict[str, str]
    clearance: PrivacyClearance
    rejection_reason: str | None
    analysis_evidence: MediaAnalysisEvidence
    residual_evidence: MediaAnalysisEvidence | None


def inspect_media_privacy(
    *,
    item: PreprocessingInput,
    object_id: str,
    detector: PiiDetector,
    fields: dict[str, str | None],
    inspection: InspectionResult,
    media_path: str,
    content_field_prefixes: tuple[str, ...],
    source_media_path: str | None = None,
    inspected_artifact: PublishedPrivacyArtifact | None = None,
    remediated_artifact: PublishedPrivacyArtifact | None = None,
    residual_inspection: InspectionResult | None = None,
) -> MediaPrivacyResult:
    """Evaluate local inspection evidence and bind clearance to exact bytes."""

    source_path = source_media_path or item.media_path or media_path
    source_digest = optional_file_sha256(source_path)
    inspected_digest = optional_file_sha256(media_path)
    analysis = media_analysis_evidence(
        inspection,
        expected_digest=inspected_digest or "",
    )

    inspected_fields = inspect_media_fields_for_release(
        item=item,
        fields=fields,
        detector=detector,
        evidence_fields=(
            analysis.completed_checks if analysis.valid else frozenset()
        ),
        required_evidence_fields=inspection.coverage.required_fields,
        input_digest=source_digest or inspected_digest,
        output_digest=inspected_digest,
    )
    clearance = inspected_fields.clearance

    if not source_digest or not inspected_digest:
        clearance = clearance.mark_incomplete(
            reason="media_subject_unavailable"
        )
    if not analysis.valid:
        for reason in analysis.reasons:
            clearance = clearance.mark_incomplete(reason=reason)

    rules_rejection = _media_rules_rejection(item=item, evidence=analysis)
    if rules_rejection is not None:
        clearance = clearance.reject(reason=rules_rejection)

    content_findings = _requires_media_remediation(
        item=item,
        evidence=analysis,
    ) or inspected_fields.has_findings_for_prefixes(content_field_prefixes)

    final_path = media_path
    final_digest = inspected_digest
    final_artifact = inspected_artifact
    residual: MediaAnalysisEvidence | None = None

    if inspected_artifact is not None:
        try:
            verify_artifact_binding(
                inspected_artifact,
                expected_source_sha256=source_digest,
                expected_output_sha256=inspected_digest,
            )
        except PrivacyArtifactOutputMismatch:
            clearance = clearance.reject(
                reason="media_derivative_digest_mismatch"
            )
        except PrivacyArtifactVerificationError:
            clearance = clearance.reject(
                reason="media_derivation_receipt_invalid"
            )

    if content_findings and rules_rejection is None:
        residual_digest = (
            remediated_artifact.sha256
            if remediated_artifact is not None
            else ""
        )
        if residual_inspection is not None:
            residual = media_analysis_evidence(
                residual_inspection,
                expected_digest=residual_digest,
            )
        residual_verified = (
            remediated_artifact is not None
            and bool(residual_digest)
            and residual is not None
            and residual_inspection is not None
            and residual.valid
            and not residual.findings
            and residual_inspection.coverage.required_fields.issubset(
                residual.completed_checks
            )
        )
        if residual_verified and remediated_artifact is not None:
            try:
                verify_artifact_binding(
                    remediated_artifact,
                    expected_source_sha256=source_digest,
                    expected_output_sha256=remediated_artifact.sha256,
                    parent_artifact=inspected_artifact,
                )
            except PrivacyArtifactVerificationError:
                residual_verified = False
        if not residual_verified or remediated_artifact is None:
            clearance = clearance.reject(
                reason="media_contains_unremediated_private_data"
            )
            if residual is not None:
                for reason in residual.reasons:
                    clearance = clearance.reject(reason=reason)
        else:
            final_artifact = remediated_artifact
            final_path = str(remediated_artifact.path)
            final_digest = residual_digest
    elif inspected_artifact is not None:
        final_path = str(inspected_artifact.path)
        final_digest = inspected_artifact.sha256

    if (
        source_digest
        and final_digest
        and source_digest != final_digest
        and final_artifact is not None
    ):
        clearance = clearance.bind_verified_remediation(
            input_digest=source_digest,
            output_digest=final_digest,
            derivation_digest=final_artifact.derivation_sha256,
        )
    elif clearance.permits_training:
        clearance = clearance.bind_output(digest=final_digest)

    rejection_reason = _release_rejection_reason(
        clearance,
        rules_rejection=rules_rejection,
        analysis=analysis,
    )
    if clearance.permits_training:
        output_digest = clearance.output_digest
        if output_digest is None:
            raise RuntimeError(
                "training-permitted privacy clearance lacks an output digest"
            )
        clearance = clearance.approve_object(
            object_id=object_id,
            role=ApprovedObjectRole.PRIMARY_MEDIA,
            output_digest=output_digest,
            derived_from_digest=(
                source_digest if source_digest != output_digest else None
            ),
        )

    return MediaPrivacyResult(
        media_path=final_path,
        fields=inspected_fields.values,
        clearance=clearance,
        rejection_reason=rejection_reason,
        analysis_evidence=analysis,
        residual_evidence=residual,
    )


def _release_rejection_reason(
    clearance: PrivacyClearance,
    *,
    rules_rejection: str | None,
    analysis: MediaAnalysisEvidence,
) -> str | None:
    """Pick the most specific recorded cause for a blocked release."""

    if clearance.permits_training:
        return None
    # Prefer the policy-level cause over incidental unchecked-field
    # diagnostics produced while the same inspection was failing closed.
    return (
        rules_rejection
        or analysis.primary_failure_reason
        or (clearance.reasons[0] if clearance.reasons else None)
        or f"privacy_{clearance.status.value}"
    )


def _media_rules_rejection(
    *,
    item: PreprocessingInput,
    evidence: MediaAnalysisEvidence,
) -> str | None:
    if "identifiable_voice_without_authorization" in evidence.errors:
        return "identifiable_voice_without_authorization"
    if not evidence.valid:
        return None
    finding_types = evidence.finding_types()
    if "identity_document" in finding_types:
        return "identity_document_detected"
    if item.modality == "image":
        supported = {
            "face",
            "license_plate",
            "machine_readable_code",
            "person_name",
            "email_address",
            "phone_number",
            "postal_address",
            "date_of_birth",
            "belgian_national_number",
            "passport_number",
            "iban",
            "payment_card",
            "signature",
        }
        for finding in evidence.findings:
            if finding.field_name.startswith("metadata:"):
                continue
            if finding.finding_type not in supported:
                return (
                    f"unsupported_image_privacy_finding:{finding.finding_type}"
                )
            if finding.bounding_box is None:
                return (
                    f"image_finding_missing_mask_region:{finding.finding_type}"
                )
    return None


def _requires_media_remediation(
    *,
    item: PreprocessingInput,
    evidence: MediaAnalysisEvidence,
) -> bool:
    if not evidence.findings:
        return False
    if item.modality == "audio":
        return any(
            finding.finding_type != "voice_identity"
            for finding in evidence.findings
        )
    return any(
        not finding.field_name.startswith("metadata:")
        for finding in evidence.findings
    )


__all__ = [
    "MediaPrivacyResult",
    "inspect_media_privacy",
]
