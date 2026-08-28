"""Image preprocessing orchestration."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger
from preprocessing.media.adapters.pillow_image import inspect_image_dimensions
from preprocessing.media.base_media_preprocessor import BaseMediaPreprocessor
from preprocessing.media.media_input_validation import (
    MediaValidationResult,
    accepted_media_result,
    as_optional_float,
    as_optional_int,
    as_optional_text,
    modality_preprocessing_limit,
    payload_field,
    rejected_media_result,
    resolve_media_path,
    resolve_path_object,
    validate_common_media_fields,
)
from preprocessing.media.privacy_inspection import (
    inspect_media_privacy,
)
from preprocessing.preprocessed_media import PreprocessedImage
from preprocessing.preprocessing_input import PreprocessingInput
from preprocessing.preprocessing_result import PreprocessingQuarantineRecord
from preprocessing.privacy.artifacts import (
    PrivacyArtifactWorkspace,
    PublishedPrivacyArtifact,
    build_receipt,
    canonical_sha256,
    file_sha256,
    write_exclusive_bytes,
)
from preprocessing.privacy.field_inspection import text_payload_fields
from preprocessing.privacy.inspection.detector_registry import (
    DetectorRegistry,
)
from preprocessing.privacy.inspection.inspect_image import inspect_image
from preprocessing.privacy.inspection.inspection_result import (
    InspectionResult,
    media_analysis_evidence,
)
from preprocessing.privacy.inspection.local_content_factories import (
    ImagePrivacyContentFactory,
)
from preprocessing.privacy.remediation.images.mask_sensitive_regions import (
    mask_sensitive_regions,
)
from preprocessing.privacy.text_privacy import PiiDetector

if TYPE_CHECKING:
    from config.collection.modality_acceptance import (
        ModalityAcceptanceSettings,
    )
    from config.preprocessing.media_settings import ImageValidationSettings
    from preprocessing.media.ports import EmbeddedMetadataAdapter


_ALLOWED_IMAGE_MIME_TYPES: tuple[str, ...] = (
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/tiff",
    "image/avif",
    "image/bmp",
)


class ImagePreprocessor(BaseMediaPreprocessor[PreprocessedImage]):
    """Validate image metadata, OCR text, dedupe signals, and quality."""

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        settings: ImageValidationSettings,
        modality_acceptance: ModalityAcceptanceSettings,
        pii_detector: PiiDetector,
        privacy_content_factory: ImagePrivacyContentFactory,
        embedded_metadata_adapter: EmbeddedMetadataAdapter,
        now: Callable[[], datetime],
        generate_id: Callable[[], str],
    ) -> None:
        super().__init__(
            modality="image",
            logger=logger,
            now=now,
            generate_id=generate_id,
        )
        self._settings = settings
        self._modality_acceptance = modality_acceptance
        self._pii_detector = pii_detector
        self._privacy_content_factory = privacy_content_factory
        self._embedded_metadata_adapter = embedded_metadata_adapter

    def _validate(self, *, item: PreprocessingInput) -> MediaValidationResult:
        if not self._settings.enabled:
            return accepted_media_result(signals={})

        reason, signals = validate_common_media_fields(
            item=item,
            allowed_mime_types=_ALLOWED_IMAGE_MIME_TYPES,
            min_bytes=self._settings.min_bytes,
            max_bytes=modality_preprocessing_limit(self._modality_acceptance),
        )
        media_path = resolve_media_path(item=item)
        path = resolve_path_object(media_path=media_path)
        width, height, decode_reason = _resolve_image_dimensions(
            item=item,
            path=path,
            max_decode_pixels=self._modality_acceptance.max_decode_pixels,
        )
        signals.update(
            {
                "width": width,
                "height": height,
                "decode_checked": decode_reason is None and path is not None,
            }
        )
        if reason is not None:
            return rejected_media_result(reason=reason, signals=signals)
        if decode_reason is not None:
            return rejected_media_result(reason=decode_reason, signals=signals)
        if width is None or height is None:
            return rejected_media_result(
                reason="invalid_dimensions",
                signals=signals,
            )
        if (
            width < self._settings.min_width
            or height < self._settings.min_height
        ):
            return rejected_media_result(
                reason="invalid_dimensions",
                signals=signals,
            )
        needs_semantic = (
            self._settings.require_semantic_text
            or self._settings.require_semantic_text_for_alignment
        )
        if needs_semantic and not _has_semantic_text(item=item):
            return rejected_media_result(
                reason="missing_ocr_or_caption",
                signals=signals,
            )
        return accepted_media_result(signals=signals)

    def _build_record(
        self,
        *,
        item: PreprocessingInput,
        validation: MediaValidationResult,
    ) -> PreprocessedImage | PreprocessingQuarantineRecord:
        width = as_optional_int(validation.signals.get("width"))
        height = as_optional_int(validation.signals.get("height"))
        ocr_text: str | None = None
        fields = text_payload_fields(
            item=item,
            names=(
                "caption_text",
                "alt_text",
                "figcaption",
                "surrounding_text",
                "page_title",
                "author",
                "creator",
                "description",
            ),
        )
        embedded_fields, metadata_artifact, metadata_rejection = (
            self._prepare_embedded_metadata(
                item=item,
                adapter=self._embedded_metadata_adapter,
            )
        )
        if metadata_rejection is not None:
            return PreprocessingQuarantineRecord.from_input(
                item=item,
                reason=metadata_rejection,
                finding_counts={},
                quality_signals={},
            )
        fields.update(embedded_fields)

        inspected_path = Path(
            metadata_artifact.path
            if metadata_artifact is not None
            else item.media_path or ""
        )
        content = self._privacy_content_factory.build(
            item=item,
            media_path=inspected_path,
            metadata={},
            residual=False,
        )
        if content.ocr_text:
            fields["ocr_text"] = content.ocr_text
        inspection = inspect_image(content, self._pii_detector.registry)
        (
            remediated_artifact,
            residual_inspection,
            derivative_rejection,
        ) = _prepare_image_privacy_derivative(
            item=item,
            original_source_path=Path(item.media_path or ""),
            transform_input_path=inspected_path,
            parent_artifact=metadata_artifact,
            inspection=inspection,
            content_factory=self._privacy_content_factory,
            registry=self._pii_detector.registry,
            max_decode_pixels=self._modality_acceptance.max_decode_pixels,
            now=self._now,
            generate_id=self._generate_id,
        )
        if derivative_rejection is not None:
            return PreprocessingQuarantineRecord.from_input(
                item=item,
                reason=derivative_rejection,
                finding_counts=inspection.finding_counts,
                quality_signals={},
            )

        privacy = inspect_media_privacy(
            item=item,
            object_id=self._media_id(item=item),
            detector=self._pii_detector,
            fields=fields,
            inspection=inspection,
            media_path=str(inspected_path),
            source_media_path=item.media_path,
            inspected_artifact=metadata_artifact,
            remediated_artifact=remediated_artifact,
            residual_inspection=residual_inspection,
            content_field_prefixes=("ocr_text", "ocr_preview"),
        )
        if privacy.rejection_reason is not None:
            return PreprocessingQuarantineRecord.from_input(
                item=item,
                reason=privacy.rejection_reason,
                finding_counts=inspection.finding_counts,
                quality_signals={
                    "privacy_status": privacy.clearance.status.value,
                    "privacy_reasons": list(privacy.clearance.reasons),
                },
            )
        ocr_text = privacy.fields.get("ocr_text")
        caption_text = privacy.fields.get("caption_text")
        caption_source = as_optional_text(item.payload.get("caption_source"))
        caption_quality_score = as_optional_float(
            item.payload.get("caption_quality_score")
        )
        semantic_text = ocr_text or caption_text
        quality = self._quality_for_valid_item(
            item=item,
            validation=validation,
            semantic_text=semantic_text,
            has_alignment_material=bool(semantic_text),
            extra_signals={
                "ocr_available": bool(ocr_text),
                "caption_available": bool(caption_text),
                "caption_source": caption_source,
                "caption_quality_score": caption_quality_score,
                "mime_type": item.mime_type,
                "image_width": width,
                "image_height": height,
            },
        )
        fingerprints = self._fingerprints(
            item=item,
            primary_text=semantic_text,
        )
        for canonical_name, payload_name in (
            ("image_ahash", "image_average_hash"),
            ("image_dhash", "image_difference_hash"),
            ("image_phash", "image_phash"),
        ):
            value = as_optional_text(item.payload.get(payload_name))
            if value is not None:
                fingerprints[canonical_name] = value
        return PreprocessedImage(
            media_id=self._media_id(item=item),
            source_id=item.source_id,
            source_url=item.source_url,
            normalized_url=item.normalized_url,
            domain=item.domain,
            media_path=privacy.media_path,
            mime_type=item.mime_type,
            width=width,
            height=height,
            normalized_media_path=privacy.media_path,
            ocr_text=ocr_text,
            ocr_confidence=as_optional_float(
                item.payload.get("ocr_confidence")
            ),
            ocr_language=item.resolved_language()
            or as_optional_text(item.payload.get("ocr_language")),
            ocr_quality_score=as_optional_float(
                item.payload.get("ocr_quality_score")
            ),
            quality=quality,
            dedupe_fingerprints=fingerprints,
            alignment_signals={
                "ocr_available": bool(ocr_text),
                "caption_available": bool(caption_text),
                "caption": {
                    "text": caption_text,
                    "source": caption_source,
                    "quality_score": caption_quality_score,
                },
                "caption_source": caption_source,
                "caption_quality_score": caption_quality_score,
                "dimensions_available": (
                    width is not None and height is not None
                ),
            },
            safety_status="passed",
            privacy_clearance=privacy.clearance,
            privacy_evidence={
                "analysis": privacy.analysis_evidence.to_dict(),
                "residual": (
                    privacy.residual_evidence.to_dict()
                    if privacy.residual_evidence is not None
                    else None
                ),
            },
        )


def _resolve_image_dimensions(
    *,
    item: PreprocessingInput,
    path: Path | None,
    max_decode_pixels: int,
) -> tuple[int | None, int | None, str | None]:
    width = item.width or as_optional_int(
        payload_field(
            item=item,
            name="image_width",
        )
    )
    height = item.height or as_optional_int(
        payload_field(
            item=item,
            name="image_height",
        )
    )
    if path is None or not path.exists():
        return width, height, None
    decoded = inspect_image_dimensions(
        path=path,
        max_decode_pixels=max_decode_pixels,
    )
    if decoded is None:
        return width, height, "decode_failed"
    return decoded[0], decoded[1], None


def _prepare_image_privacy_derivative(
    *,
    item: PreprocessingInput,
    original_source_path: Path,
    transform_input_path: Path,
    parent_artifact: PublishedPrivacyArtifact | None,
    inspection: InspectionResult,
    content_factory: ImagePrivacyContentFactory,
    registry: DetectorRegistry,
    max_decode_pixels: int,
    now: Callable[[], datetime],
    generate_id: Callable[[], str],
) -> tuple[
    PublishedPrivacyArtifact | None,
    InspectionResult | None,
    str | None,
]:
    """Create, residual-scan, receipt-bind, then publish a fresh mask."""

    if not transform_input_path.is_file() or not inspection.safe_to_assess:
        return None, None, None
    media_findings = tuple(
        finding
        for finding in inspection.findings
        if not finding.location.field_name.startswith("metadata:")
    )
    if not media_findings:
        return None, None, None
    if any(
        finding.finding_type.value == "identity_document"
        for finding in media_findings
    ):
        return None, None, None

    if any(
        finding.location.bounding_box is None for finding in media_findings
    ):
        return None, None, None

    try:
        with PrivacyArtifactWorkspace(
            source_path=original_source_path,
            stage="privacy-sanitized",
            run_id=generate_id(),
        ) as workspace:
            source_digest = workspace.source_snapshot.sha256
            effective_transform_input = (
                parent_artifact.path
                if parent_artifact is not None
                else workspace.source_path
            )
            expected_transform_digest = (
                parent_artifact.sha256
                if parent_artifact is not None
                else source_digest
            )
            if file_sha256(transform_input_path) != expected_transform_digest:
                return None, None, "image_privacy_transform_input_mismatch"
            transform_payload = effective_transform_input.read_bytes()
            transform_input_digest = canonical_sha256_bytes(transform_payload)
            if transform_input_digest != expected_transform_digest:
                return None, None, "image_privacy_transform_input_changed"
            sanitized = mask_sensitive_regions(
                payload=transform_payload,
                findings=media_findings,
            )
            temporary = workspace.new_bytes_temp(suffix=".png")
            write_exclusive_bytes(temporary, sanitized)
            if (
                inspect_image_dimensions(
                    path=temporary,
                    max_decode_pixels=max_decode_pixels,
                )
                is None
            ):
                return None, None, "image_privacy_derivative_decode_failed"

            output_digest = file_sha256(temporary)
            residual_content = content_factory.build(
                item=item,
                media_path=temporary,
                metadata={},
                residual=True,
            )
            residual = inspect_image(residual_content, registry)
            residual_evidence = media_analysis_evidence(
                residual,
                expected_digest=output_digest,
            )
            if (
                not residual_evidence.valid
                or residual_evidence.findings
                or not residual.coverage.required_fields.issubset(
                    residual_evidence.completed_checks
                )
                or file_sha256(temporary) != output_digest
            ):
                return (
                    None,
                    residual,
                    "image_privacy_residual_inspection_failed",
                )

            transform_path = Path(
                inspect.getsourcefile(mask_sensitive_regions) or __file__
            )
            receipt = build_receipt(
                workspace=workspace,
                source_path=workspace.original_source_path,
                source_sha256=source_digest,
                transform_input_sha256=transform_input_digest,
                output_path=temporary,
                output_sha256=output_digest,
                source_mime_type=item.mime_type,
                output_mime_type="image/png",
                transform_id="image-region-mask",
                transform_version="1.1.0",
                transform_artifact_path=transform_path,
                configuration={
                    "format": "PNG",
                    "mask": "solid-black",
                    "boxes": [
                        {
                            "x": box.x,
                            "y": box.y,
                            "width": box.width,
                            "height": box.height,
                        }
                        for finding in media_findings
                        if (box := finding.location.bounding_box) is not None
                    ],
                },
                residual_inspection_sha256=canonical_sha256(
                    residual_evidence.to_dict()
                ),
                created_at=now(),
                parent_artifact=parent_artifact,
            )
            final_name = f"{original_source_path.stem}.privacy-sanitized.png"
            artifact = workspace.publish(
                temporary_path=temporary,
                receipt=receipt,
                final_name=final_name,
            )
            return artifact, residual, None
    except (OSError, RuntimeError, ValueError, TypeError):
        return None, None, "image_privacy_derivative_failed"


def canonical_sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _has_semantic_text(*, item: PreprocessingInput) -> bool:
    for value in (
        item.ocr_text,
        item.payload.get("ocr_text"),
        item.payload.get("ocr_preview"),
        item.payload.get("caption_text"),
        item.payload.get("alt_text"),
        item.payload.get("figcaption"),
    ):
        if as_optional_text(value):
            return True
    return False
