"""Build finalized sample variants from prepared augmentation text."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING

from augmentation.outcomes.rejection_reason import AugmentationRejectionReason
from augmentation.text.text_identity import text_field_value, text_identity
from augmentation.text.text_variant_quality import (
    evaluate_text_variant_quality,
)
from logger.project_logger import ProjectLogger

_AUGMENTATION_IMPLEMENTATION_VERSION = "augmentation-v1"

if TYPE_CHECKING:
    from config.augmentation.augmentation_settings import AugmentationSettings
    from mmcrawler_datasets.schema import MultimodalSample


class TextVariantAssembler:
    """Apply length, dedupe, metadata, and identity rules for variants."""

    def __init__(
        self,
        *,
        settings: AugmentationSettings,
        logger: ProjectLogger,
    ) -> None:
        """Initialize variant assembly with settings and quality scoring."""

        self._settings = settings
        self._logger = logger
        self._logger.debug("augmentation_variant_assembler_initialized")
        self._config_hash = _settings_hash(settings=settings)

    def build(
        self,
        *,
        sample: MultimodalSample,
        augmentation_name: str,
        variant_text: str | None,
        seen_texts: set[str],
    ) -> MultimodalSample | None:
        """Build a variant sample or return ``None`` when rejected."""

        variant, _reason = self.build_with_reason(
            sample=sample,
            augmentation_name=augmentation_name,
            variant_text=variant_text,
            seen_texts=seen_texts,
        )
        return variant

    def build_with_reason(
        self,
        *,
        sample: MultimodalSample,
        augmentation_name: str,
        variant_text: str | None,
        seen_texts: set[str],
    ) -> tuple[MultimodalSample | None, AugmentationRejectionReason | None]:
        """Build a variant and include a rejection reason when skipped."""

        prepared_text, rejected_reason, truncated = (
            self._prepare_text_with_reason(variant_text=variant_text)
        )
        if prepared_text is None:
            return None, rejected_reason

        prepared_identity = text_identity(prepared_text)
        if prepared_identity in seen_texts:
            return None, AugmentationRejectionReason.DUPLICATE_TEXT_SAMPLE

        seen_texts.add(prepared_identity)
        source_text = text_field_value(sample.text)
        variant_id = self._variant_id(
            source_sample_id=sample.sample_id,
            augmentation_name=augmentation_name,
            variant_text=prepared_text,
        )

        return replace(
            sample,
            sample_id=variant_id,
            text=prepared_text,
            metadata=self._metadata(
                sample=sample,
                variant_id=variant_id,
                augmentation_name=augmentation_name,
                source_text=source_text,
                variant_text=prepared_text,
                truncated=truncated,
            ),
        ), None

    def _prepare_text_with_reason(
        self,
        *,
        variant_text: str | None,
    ) -> tuple[str | None, AugmentationRejectionReason | None, bool]:
        """Normalize and optionally truncate variant text with a reason."""

        prepared_text = text_field_value(variant_text)
        if not prepared_text:
            return None, AugmentationRejectionReason.VARIANT_TEXT_EMPTY, False

        max_length = self._settings.text.maximum_text_length
        if max_length is None or len(prepared_text) <= max_length:
            return prepared_text, None, False

        if self._settings.text.truncation_rules != "truncate":
            return (
                None,
                AugmentationRejectionReason.VARIANT_TEXT_TOO_LONG,
                False,
            )

        truncated_text = _truncate_at_boundary(
            text=prepared_text,
            max_length=max_length,
        )
        if not truncated_text:
            return (
                None,
                AugmentationRejectionReason.VARIANT_TEXT_TOO_LONG,
                False,
            )

        return truncated_text, None, True

    def _metadata(
        self,
        *,
        sample: MultimodalSample,
        variant_id: str,
        augmentation_name: str,
        source_text: str,
        variant_text: str,
        truncated: bool,
    ) -> dict[str, object]:
        """Build augmentation metadata for the generated variant."""

        metadata = dict(sample.metadata or {})
        metadata["augmentation_name"] = augmentation_name
        metadata["augmentation_source_sample_id"] = sample.sample_id
        metadata["augmentation_variant_id"] = variant_id
        metadata["augmentation_config_hash"] = self._config_hash
        metadata["augmentation_seed"] = _deterministic_seed(
            source_sample_id=sample.sample_id,
            augmentation_name=augmentation_name,
        )
        metadata["augmentation_parameters"] = {
            "name": augmentation_name,
            "config_hash": self._config_hash,
        }
        metadata["augmentation_implementation_version"] = (
            _AUGMENTATION_IMPLEMENTATION_VERSION
        )
        metadata["augmentation_type"] = "text_field"
        metadata["augmentation_modifies"] = ["text"]
        metadata["augmentation_media_transform_applied"] = False
        metadata["augmentation_scope"] = "train"
        metadata["augmentation_split_selector"] = "copy_without_augmentation"
        metadata["augmentation_language_rules"] = (
            self._settings.text.language_rules
        )
        metadata["augmentation_text_truncation_applied"] = truncated
        metadata["augmentation_quality"] = evaluate_text_variant_quality(
            source_text=source_text,
            variant_text=variant_text,
        )
        return metadata

    def _variant_id(
        self,
        *,
        source_sample_id: str,
        augmentation_name: str,
        variant_text: str,
    ) -> str:
        payload = (
            f"{source_sample_id}\0"
            f"{augmentation_name}\0"
            f"{self._config_hash}\0"
            f"{variant_text}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"sample_aug_{digest}"


def _settings_hash(*, settings: AugmentationSettings) -> str:
    payload = json.dumps(
        settings.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deterministic_seed(
    *,
    source_sample_id: str,
    augmentation_name: str,
) -> int:
    payload = f"{source_sample_id}\0{augmentation_name}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _truncate_at_boundary(*, text: str, max_length: int) -> str:
    """Truncate text near a natural boundary when possible."""

    candidate = text[:max_length].rstrip()
    if not candidate:
        return ""

    candidate = _drop_unclosed_code_fence(text=candidate)
    boundary = _preferred_boundary(text=candidate, max_length=max_length)
    if boundary is not None:
        candidate = candidate[:boundary].rstrip()
    else:
        candidate = _drop_partial_token(text=candidate).rstrip()

    return candidate


def _preferred_boundary(*, text: str, max_length: int) -> int | None:
    """Return the best boundary index for readable truncation."""

    lower_bound = max(1, min(len(text) - 1, int(max_length * 0.5)))
    markers = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ")

    for marker in markers:
        position = text.rfind(marker)
        if position >= lower_bound:
            return position + len(marker.rstrip())

    return None


def _drop_partial_token(*, text: str) -> str:
    """Remove an incomplete trailing token after hard truncation."""

    for index in range(len(text) - 1, -1, -1):
        if text[index].isspace():
            return text[:index]

    return text


def _drop_unclosed_code_fence(*, text: str) -> str:
    """Remove a trailing unclosed markdown code fence block."""

    if text.count("```") % 2 == 0:
        return text

    fence_start = text.rfind("```")
    if fence_start <= 0:
        return text

    return text[:fence_start].rstrip()
