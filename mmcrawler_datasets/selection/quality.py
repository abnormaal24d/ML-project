"""Quality enrichment and rejection checks for training samples."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from mmcrawler_datasets.similarity.text import build_text_shingle_profile
from mmcrawler_datasets.training_samples.fingerprints import (
    ContentFingerprints,
)
from preprocessing.privacy.text_privacy import PiiDetector

from .contracts import RejectReason
from .privacy_clearance import verify_sample_clearance

if TYPE_CHECKING:
    from config.settings.datasets import (
        DatasetValidatorSettings,
        TrainingSnapshotAssemblerSettings,
    )
from mmcrawler_datasets.training_samples.models import TrainingSample

_BLOCKED_SAFETY_TERMS = frozenset(
    {"nsfw", "porn", "exploit", "gore", "terrorist"}
)
_GENERIC_CAPTIONS = frozenset(
    {
        "image",
        "photo",
        "picture",
        "logo",
        "icon",
        "click here",
        "read more",
        "video",
        "audio",
    }
)


def finalize_sample(
    *,
    sample: TrainingSample,
    settings: TrainingSnapshotAssemblerSettings,
    validator_settings: DatasetValidatorSettings,
    snapshot_id: str,
    project_root: Path,
    pii_detector: PiiDetector,
) -> TrainingSample:
    """Populate lineage and enforce configured release requirements."""

    project_root = project_root.resolve(strict=True)
    content_hash = sample.content_hash or _content_hash(sample)
    language = _normalize_language(sample.language)
    confidence = sample.language_confidence
    if confidence is None:
        confidence = 1.0 if language is not None else 0.0

    detected_safety, safety_labels = _classify_safety(sample.text)
    safety_status = _resolved_safety_status(
        sample=sample,
        detected_status=detected_safety,
        required=validator_settings.require_safety_passed,
    )
    if validator_settings.require_pii_passed:
        clearance = verify_sample_clearance(
            sample,
            project_root=project_root,
            residual_pii_detector=pii_detector,
        )
        pii_status = "passed" if clearance.valid else "blocked"
    else:
        pii_status = _classify_pii(sample.text, pii_detector)

    training_usage_status = _resolve_training_usage_status(
        sample,
        require_known_license=validator_settings.require_known_license,
        require_license_evidence=validator_settings.require_license_evidence,
        require_license_url_or_terms=(
            validator_settings.require_license_url_or_terms
        ),
    )
    quarantine = _quarantine_reason(
        sample=sample,
        safety_status=safety_status,
        pii_status=pii_status,
        training_usage_status=training_usage_status,
    )
    alignment_score = _alignment_score(sample)
    return replace(
        sample,
        language=language,
        language_confidence=confidence,
        language_script=sample.language_script or _script_for(sample.text),
        task_target=replace(
            sample.task_target,
            alignment_score=alignment_score,
        ),
        source_object_ids=_source_object_ids(sample),
        dataset_version=(
            sample.dataset_version
            or f"{settings.dataset_version_prefix}:{snapshot_id}"
        ),
        content_hash=content_hash,
        processing_version=(
            sample.processing_version or settings.processing_version
        ),
        governance=replace(
            sample.governance,
            usage_rules=training_usage_status,
            robots_status=sample.governance.robots_status or "unknown",
        ),
        safety_status=safety_status,
        safety_labels=safety_labels,
        pii_status=pii_status,
        quarantine_reason=quarantine,
        content_fingerprints=_content_fingerprints(sample),
    )


_PAIR_TASKS = frozenset(
    {
        "document_text_pair",
        "image_text_pair",
        "audio_text_pair",
        "video_text_pair",
    }
)


def quality_reject(
    sample: TrainingSample,
    settings: TrainingSnapshotAssemblerSettings,
    validator_settings: DatasetValidatorSettings,
) -> RejectReason | None:
    """Return why quality rules rejects a permitted sample."""

    if sample.task_target.task_type in _PAIR_TASKS:
        if sample.pairability_score is None:
            return RejectReason.ALIGNMENT
        if sample.pairability_score < settings.min_pairability_score:
            return RejectReason.ALIGNMENT

    minimum_quality = validator_settings.min_quality_score_by_modality.get(
        sample.modality
    )
    if minimum_quality is not None and sample.quality_score < minimum_quality:
        return RejectReason.LOW_QUALITY

    minimum_context = validator_settings.min_context_score_by_modality.get(
        sample.modality
    )
    if minimum_context is not None:
        if sample.context_score is None:
            return RejectReason.LOW_CONTEXT
        if sample.context_score < minimum_context:
            return RejectReason.LOW_CONTEXT

    minimum_modality_alignment = (
        validator_settings.min_alignment_score_by_modality.get(
            sample.modality,
            0.0,
        )
    )

    minimum_task_alignment = (
        validator_settings.min_alignment_score_by_task.get(
            sample.task_target.task_type,
            0.0,
        )
    )

    required_alignment = max(
        settings.min_alignment_score,
        minimum_modality_alignment,
        minimum_task_alignment,
    )

    if sample.task_target.alignment_score < required_alignment:
        return RejectReason.ALIGNMENT

    if not _language_allowed(sample, settings):
        return RejectReason.LANGUAGE
    if sample.safety_status == "blocked" or sample.pii_status == "blocked":
        return RejectReason.SAFETY
    if not _fingerprint_evidence_complete(sample):
        return RejectReason.FINGERPRINT
    if sample.quarantine_reason:
        return _quarantine_reject(sample.quarantine_reason)
    if (
        sample.modality == "image"
        and sample.quality_score < settings.min_caption_quality_score
    ):
        return RejectReason.LOW_CAPTION
    if is_generic_caption(sample.text):
        return RejectReason.CAPTION
    return None


def is_generic_caption(text: str) -> bool:
    """Return whether text is generic or dominated by repeated tokens."""

    normalized = " ".join(str(text).casefold().strip().split())
    if not normalized or normalized in _GENERIC_CAPTIONS:
        return True
    tokens = normalized.split()
    if len(tokens) <= 2 and any(
        token in _GENERIC_CAPTIONS for token in tokens
    ):
        return True
    return bool(tokens) and 1.0 - (len(set(tokens)) / len(tokens)) > 0.6


def _language_allowed(
    sample: TrainingSample,
    settings: TrainingSnapshotAssemblerSettings,
) -> bool:
    if settings.language_rules == "multilingual":
        return True
    language = _normalize_language(sample.language)
    if language is None:
        return False
    accepted = {item.lower() for item in settings.accepted_languages}
    confidence = sample.language_confidence
    if confidence is None:
        confidence = 1.0
    return (
        language in accepted and confidence >= settings.min_language_confidence
    )


def _alignment_score(sample: TrainingSample) -> float:
    if sample.task_target.task_type in _PAIR_TASKS:
        candidates = [sample.quality_score]

        if sample.pairability_score is not None:
            candidates.append(sample.pairability_score)

        if sample.context_score is not None:
            candidates.append(sample.context_score)

        return round(
            max(
                0.0,
                min(1.0, *candidates),
            ),
            4,
        )

    candidates = [
        sample.task_target.alignment_score,
        sample.quality_score,
    ]

    if sample.context_score is not None:
        candidates.append(sample.context_score)

    return round(
        max(
            0.0,
            min(
                1.0,
                max(candidates),
            ),
        ),
        4,
    )


def _content_hash(sample: TrainingSample) -> str:
    object_path = (
        sample.objects[0].object_path.relative_path if sample.objects else ""
    )
    payload = "\n".join(
        value
        for value in (
            sample.text,
            sample.object_id or "",
            object_path,
            sample.source_url,
        )
        if value
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_object_ids(sample: TrainingSample) -> tuple[str, ...]:
    values = [
        sample.document_id,
        sample.chunk_id,
        sample.object_id,
        *(obj.object_id for obj in sample.objects),
    ]
    return tuple(dict.fromkeys(value for value in values if value))


def _content_fingerprints(
    sample: TrainingSample,
) -> ContentFingerprints:
    inputs = sample.fingerprint_inputs
    emitted_text_sha256 = (
        hashlib.sha256(sample.text.encode("utf-8")).hexdigest()
        if sample.text
        else None
    )
    normalized_text_sha256 = (
        hashlib.sha256(
            _normalize_text(sample.text).encode("utf-8")
        ).hexdigest()
        if sample.text.strip()
        else None
    )
    text_shingle_profile = (
        tuple(sorted(build_text_shingle_profile(sample.text)))
        if sample.text.strip()
        else None
    )
    document_layout_sha256 = (
        _document_layout_fingerprint(sample)
        if sample.modality == "document"
        else None
    )
    return ContentFingerprints(
        emitted_text_sha256=emitted_text_sha256,
        normalized_text_sha256=normalized_text_sha256,
        text_shingle_profile=text_shingle_profile,
        image_ahash=inputs.image_ahash,
        image_dhash=inputs.image_dhash,
        image_phash=inputs.image_phash,
        audio_chromaprint=inputs.audio_chromaprint,
        video_keyframe_phashes=inputs.video_keyframe_phashes,
        document_layout_sha256=document_layout_sha256,
        document_page_phashes=inputs.document_page_phashes,
    )


def _document_layout_fingerprint(sample: TrainingSample) -> str:
    layout = "|".join(
        f"{span.page_number or 0}:{len(span.text)}"
        for span in sample.text_spans
    )
    return hashlib.sha256(layout.encode("utf-8")).hexdigest()


def _fingerprint_evidence_complete(sample: TrainingSample) -> bool:
    fingerprints = sample.content_fingerprints
    if fingerprints is None:
        return False
    if sample.text.strip() and (
        not fingerprints.emitted_text_sha256
        or not fingerprints.normalized_text_sha256
        or not fingerprints.text_shingle_profile
    ):
        return False
    modalities = {sample.modality, *sample.task_target.output_modalities}
    if "image" in modalities and (
        not fingerprints.image_dhash or not fingerprints.image_phash
    ):
        return False
    if "audio" in modalities and not fingerprints.audio_chromaprint:
        return False
    if "video" in modalities and not fingerprints.video_keyframe_phashes:
        return False
    if "document" in modalities and not fingerprints.document_layout_sha256:
        return False
    return True


def _classify_safety(text: str) -> tuple[str, tuple[str, ...]]:
    lowered = text.casefold()
    labels = tuple(
        term for term in sorted(_BLOCKED_SAFETY_TERMS) if term in lowered
    )
    return ("blocked", labels) if labels else ("passed", ())


def _classify_pii(
    text: str,
    detector: PiiDetector,
) -> str:
    result = detector.detect(text=text)
    return (
        "passed"
        if result.inspection_complete and result.assessment_outcome == "accept"
        else "blocked"
    )


def _quarantine_reason(
    *,
    sample: TrainingSample,
    safety_status: str,
    pii_status: str,
    training_usage_status: str,
) -> str | None:
    if safety_status == "blocked":
        return RejectReason.SAFETY_CONTENT.value
    if pii_status == "blocked":
        return RejectReason.PII_CONTENT.value
    if training_usage_status == "blocked":
        return RejectReason.USAGE.value
    if is_generic_caption(sample.text):
        return RejectReason.GENERIC_TEXT.value
    return None


def _quarantine_reject(reason: str) -> RejectReason:
    mapping = {
        RejectReason.SAFETY_CONTENT.value: RejectReason.SAFETY_CONTENT,
        RejectReason.PII_CONTENT.value: RejectReason.PII_CONTENT,
        RejectReason.GENERIC_TEXT.value: RejectReason.GENERIC_TEXT,
        RejectReason.USAGE.value: RejectReason.USAGE,
    }
    return mapping.get(reason, RejectReason.SAFETY)


def _resolved_safety_status(
    *,
    sample: TrainingSample,
    detected_status: str,
    required: bool,
) -> str:
    if detected_status == "blocked" or sample.safety_status == "blocked":
        return "blocked"
    if sample.safety_status == "passed":
        return "passed"
    if sample.modality in {"text", "document"}:
        return "passed"
    return "blocked" if required else detected_status


def _resolve_training_usage_status(
    sample: TrainingSample,
    *,
    require_known_license: bool,
    require_license_evidence: bool,
    require_license_url_or_terms: bool,
) -> str:
    governance = sample.governance
    license_name = (governance.license or "").strip()
    known = bool(license_name) and license_name.casefold() not in {
        "unknown",
        "none",
        "unspecified",
    }
    has_url = bool((governance.license_url or "").strip())
    has_terms = bool((governance.terms_source or "").strip())
    requirements_met = (
        (not require_known_license or known)
        and (not require_license_evidence or has_url)
        and (not require_license_url_or_terms or has_url or has_terms)
    )
    has_license = bool(license_name)
    license_allowed = has_license and (not require_known_license or known)
    if (
        governance.allow_training is True
        and license_allowed
        and requirements_met
    ):
        return "training_allowed"
    if governance.allow_training is False or not requirements_met:
        return "blocked"
    return "unknown"


def _normalize_language(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().lower()
    if not text or text in {"unknown", "und", "none"}:
        return None
    return text.split("-", maxsplit=1)[0]


def _script_for(text: str) -> str:
    for char in text:
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = ""
        return "Latin" if "LATIN" in name else "Other"
    return "Unknown"


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().strip().split())


__all__ = ["finalize_sample", "quality_reject"]
