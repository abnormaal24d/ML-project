"""Create text-only document augmentation variants."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from augmentation.document.document_variant_assembler import (
    document_variant_metadata,
    text_sha256,
)
from augmentation.generated_artifact_cache import settings_fingerprint
from augmentation.outcomes.augmentation_result import AugmentationRejection
from augmentation.outcomes.rejection_reason import AugmentationRejectionReason
from augmentation.variant_lineage import media_variant_id
from mmcrawler_datasets.schema import MultimodalSample

if TYPE_CHECKING:
    from config.augmentation.document_settings import (
        DocumentAugmentationSettings,
    )


class DocumentTextAugmenter:
    def __init__(self, *, settings: DocumentAugmentationSettings) -> None:
        self._settings = settings

    def augment(
        self,
        *,
        sample: MultimodalSample,
        root: Path,
        source_page: Path | None,
        operation: str,
    ) -> tuple[MultimodalSample | None, AugmentationRejection | None]:
        text, parameters = self._transform_text(
            sample=sample, operation=operation
        )
        if parameters is None:
            return None, _rejection(
                sample,
                AugmentationRejectionReason.UNSUPPORTED_DOCUMENT_OPERATION,
                operation,
            )
        if text == sample.text:
            return None, None
        source_sha256 = text_sha256(sample.text)
        output_sha256 = text_sha256(text)
        config_hash = settings_fingerprint(
            {
                "settings": self._settings.model_dump(mode="json"),
                "operation": operation,
            }
        )
        variant_id = media_variant_id(
            source_sample_id=sample.sample_id,
            operation=f"document_{operation}",
            source_sha256=source_sha256,
            config_hash=config_hash,
            prefix="sample_doc_aug",
        )
        metadata = document_variant_metadata(
            sample=sample,
            root=root,
            source_path=source_page,
            output_path=None,
            operation=operation,
            source_sha256=source_sha256,
            output_sha256=output_sha256,
            config_hash=config_hash,
            variant_id=variant_id,
            parameters=parameters,
            spatial_receipt=None,
            output_mime_type="text/plain",
            output_byte_size=len((text or "").encode()),
            modifies=("document_text",),
        )
        metadata["document_variant_alignment"] = {
            "variant_kind": "text_only_variant",
            "page_representation_modified": False,
            "text_representation_modified": True,
        }
        return replace(
            sample, sample_id=variant_id, text=text, metadata=metadata
        ), None

    def _transform_text(
        self,
        *,
        sample: MultimodalSample,
        operation: str,
    ) -> tuple[str | None, dict[str, object] | None]:
        if operation == "text_span":
            spans = (
                sample.metadata.get("text_spans") if sample.metadata else None
            )
            values: list[str] = []
            if isinstance(spans, list):
                values = [
                    str(
                        item.get("text") if isinstance(item, dict) else item
                    ).strip()
                    for item in spans
                ]
                values = [value for value in values if value]
            return (
                "\n\n".join(values) if values else sample.text,
                {"source": "metadata.text_spans", "span_count": len(values)},
            )
        if operation == "ocr_normalization":
            return normalize_ocr(sample.text), {
                "normalize_ligatures": True,
                "collapse_whitespace": True,
                "normalize_unicode": True,
                "normalize_hyphenation": True,
            }
        return sample.text, None


def normalize_ocr(text: str | None) -> str | None:
    if text is None:
        return None
    value = (
        unicodedata.normalize("NFKC", text)
        .replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
    )
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return value.strip()


def _rejection(
    sample: MultimodalSample,
    reason: AugmentationRejectionReason,
    variant: str,
) -> AugmentationRejection:
    return AugmentationRejection(
        reason=reason,
        sample_id=sample.sample_id,
        variant_name=variant,
        modality="document",
    )
