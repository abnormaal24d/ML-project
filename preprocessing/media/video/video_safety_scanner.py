"""Production-safe video training safety gate.

The scanner is deterministic and conservative. It does not claim to run NSFW,
face, watermark, or copyright-matching models. It blocks concrete PII/secret
findings, non-training licenses, explicit copyright-reserved signals, and —
when requested by the caller — records without enough evidence to scan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from preprocessing.privacy.inspection.content_readers.text_content import (
    TextContent,
)
from preprocessing.privacy.inspection.detector_registry import DetectorRegistry
from preprocessing.privacy.inspection.inspect_text import inspect_text

_SCANNER_VERSION = "central-privacy-video-safety-v2"

_CONFIDENTIAL_TERMS = (
    "confidential",
    "strictly confidential",
    "internal only",
    "private document",
    "do not distribute",
    "not for distribution",
)
_COPYRIGHT_TERMS = (
    "all rights reserved",
    "copyright protected",
    "copyright ©",
    "©",
    "licensed footage",
    "unauthorized reproduction",
)
_TRAINING_ALLOWED_LICENSE_TOKENS = (
    "public domain",
    "cc0",
    "creative commons zero",
    "cc-by",
    "cc by",
    "cc-by-sa",
    "cc by-sa",
    "open license",
    "public",
)
_TRAINING_BLOCKED_LICENSE_TOKENS = (
    "all rights reserved",
    "rights reserved",
    "editorial only",
    "personal use only",
    "non-commercial",
    "noncommercial",
    "no derivatives",
    "licensed footage",
    "copyright",
)


class VideoSafetyScanner:
    """Apply deterministic video safety checks with an injected registry."""

    def __init__(self, *, registry: DetectorRegistry) -> None:
        self._registry = registry

    def scan(
        self,
        *,
        video_path: str | None = None,
        body: bytes | None = None,
        frame_ocr_text: str | None = None,
        transcript: str | None = None,
        metadata: dict[str, Any] | None = None,
        require_decisive_scan: bool = False,
    ) -> dict[str, Any]:
        """Return a fail-closed video safety decision.

        ``require_decisive_scan=True`` is intended for product ingestion
        gates. In that mode, records without OCR/transcript/metadata/body/path
        evidence are rejected as ``insufficient_safety_evidence`` instead of
        silently passing.
        """

        text_parts = [
            part
            for part in (frame_ocr_text, transcript)
            if isinstance(part, str) and part.strip()
        ]
        metadata = metadata if isinstance(metadata, dict) else {}
        combined_text = "\n".join(text_parts)
        lower_text = combined_text.lower()

        findings: list[dict[str, object]] = []
        reasons: list[str] = []
        review_reasons: list[str] = []

        _collect_pii_findings(
            text=combined_text,
            findings=findings,
            reasons=reasons,
            registry=self._registry,
        )
        _collect_confidential_findings(
            text=lower_text, findings=findings, reasons=reasons
        )
        copyright_findings = _copyright_findings(text=lower_text)
        if copyright_findings:
            findings.extend(copyright_findings)
            reasons.append("copyright_or_rights_reserved_signal")

        license_status, license_reason = _license_status(metadata=metadata)
        if license_status == "blocked":
            reasons.append(license_reason)
        elif license_status == "review":
            review_reasons.append(license_reason)

        evidence = _evidence_summary(
            video_path=video_path,
            body=body,
            frame_ocr_text=frame_ocr_text,
            transcript=transcript,
            metadata=metadata,
        )
        decisive_evidence = _has_decisive_analysis(evidence)
        if require_decisive_scan and not decisive_evidence:
            reasons.append("insufficient_safety_evidence")

        is_safe = not reasons
        status = "passed" if is_safe else "blocked"
        if is_safe and review_reasons:
            status = "review"

        return {
            "is_safe": is_safe,
            "safety_status": status,
            "is_decisive": bool(reasons or decisive_evidence),
            "safety_reasons": tuple(dict.fromkeys(reasons)),
            "review_reasons": tuple(dict.fromkeys(review_reasons)),
            "pii_detected": any(
                finding.get("category") == "pii" for finding in findings
            ),
            "pii_findings": tuple(
                finding
                for finding in findings
                if finding.get("category") == "pii"
            ),
            "copyright_risk": any(
                finding.get("category") == "copyright" for finding in findings
            )
            or license_status == "blocked",
            "copyright_findings": tuple(
                finding
                for finding in findings
                if finding.get("category") == "copyright"
            ),
            "license_status": license_status,
            "evidence": evidence,
            "scanner_version": _SCANNER_VERSION,
            "model_backed": False,
        }


def _collect_pii_findings(
    *,
    text: str,
    findings: list[dict[str, object]],
    reasons: list[str],
    registry: DetectorRegistry,
) -> None:
    if not text:
        return
    inspection = inspect_text(
        TextContent(text=text, field_name="video_text"),
        registry,
    )
    if not inspection.safe_to_assess:
        reasons.append("privacy_inspection_incomplete")
    for finding in inspection.findings:
        span = finding.location.text_span
        if span is None:
            continue
        label = _VIDEO_FINDING_LABELS.get(
            finding.finding_type.value,
            finding.finding_type.value,
        )
        findings.append(
            {
                "category": "pii",
                "type": label,
                "canonical_type": finding.finding_type.value,
                "span": (span.start, span.end),
                "confidence": finding.confidence,
                "detector": finding.detector_name,
            }
        )
        reasons.append(f"pii_{label}_detected")


# These are presentation labels for detector categories, not credential values.
_VIDEO_FINDING_LABELS = {
    "email_address": "email",
    "phone_number": "phone",
    "payment_card": "credit_card",
    "api_credential": "secret",
    "cloud_credential": "secret",
    "oauth_token": "secret",  # nosec: B105
    "jwt_token": "secret",  # nosec: B105
    "session_credential": "secret",
    "private_key": "secret",
    "basic_auth_credential": "secret",
    "database_credential": "secret",
}


def _collect_confidential_findings(
    *,
    text: str,
    findings: list[dict[str, object]],
    reasons: list[str],
) -> None:
    for term in _CONFIDENTIAL_TERMS:
        index = text.find(term)
        if index >= 0:
            findings.append(
                {
                    "category": "pii",
                    "type": "confidential_marker",
                    "span": (index, index + len(term)),
                }
            )
            reasons.append("confidential_marker_detected")


def _copyright_findings(*, text: str) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for term in _COPYRIGHT_TERMS:
        index = text.find(term)
        if index >= 0:
            found.append(
                {
                    "category": "copyright",
                    "type": "rights_reserved_text",
                    "span": (index, index + len(term)),
                }
            )
    return found


def _license_status(*, metadata: dict[str, Any]) -> tuple[str, str]:
    allow_training = metadata.get("allow_training")
    if allow_training is False:
        return "blocked", "metadata_allow_training_false"

    license_value = (
        metadata.get("license")
        or metadata.get("rights")
        or metadata.get("usage_rights")
    )
    if license_value is None:
        return "unknown", "license_metadata_missing"

    license_text = str(license_value).strip().lower()
    if not license_text:
        return "unknown", "license_metadata_empty"
    if any(
        token in license_text for token in _TRAINING_BLOCKED_LICENSE_TOKENS
    ):
        return "blocked", "license_not_training_safe"
    if any(
        token in license_text for token in _TRAINING_ALLOWED_LICENSE_TOKENS
    ):
        return "passed", "license_training_allowed"
    return "review", "license_requires_review"


def _evidence_summary(
    *,
    video_path: str | None,
    body: bytes | None,
    frame_ocr_text: str | None,
    transcript: str | None,
    metadata: dict[str, Any],
) -> dict[str, bool]:
    """Separate technical availability from completed privacy analysis."""

    return {
        "video_path_exists": bool(video_path and Path(video_path).is_file()),
        "body_available": bool(body),
        "frame_ocr_text_available": bool(
            frame_ocr_text and frame_ocr_text.strip()
        ),
        "transcript_available": bool(transcript and transcript.strip()),
        "metadata_available": bool(metadata),
        "decoder_verified": _completed(
            metadata.get("privacy_decoder_verified")
        ),
        "visual_analysis_completed": _completed(
            metadata.get("privacy_visual_analysis_completed")
        ),
        "audio_analysis_completed": (
            _completed(metadata.get("privacy_audio_analysis_completed"))
            or _completed(metadata.get("privacy_audio_track_absent"))
        ),
        "frame_text_analysis_completed": (
            _completed(metadata.get("privacy_frame_ocr_analysis_completed"))
            or _completed(metadata.get("privacy_text_free_video_verified"))
        ),
        "metadata_inspection_completed": _completed(
            metadata.get("privacy_metadata_inspection_completed")
        ),
    }


def _has_decisive_analysis(evidence: dict[str, bool]) -> bool:
    return all(
        evidence.get(name, False)
        for name in (
            "decoder_verified",
            "visual_analysis_completed",
            "audio_analysis_completed",
            "frame_text_analysis_completed",
            "metadata_inspection_completed",
        )
    )


def _completed(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "passed",
        "complete",
        "completed",
        "success",
    }


__all__ = ["VideoSafetyScanner"]
