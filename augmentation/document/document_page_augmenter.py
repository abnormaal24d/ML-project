"""Render, validate, and assemble document page image variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from augmentation.annotations.spatial_transform import SpatialTransform
from augmentation.document.document_annotation_transformer import (
    transform_document_page_sample,
)
from augmentation.document.document_variant_assembler import (
    document_variant_metadata,
    relative_artifact_path,
)
from augmentation.generated_artifact_cache import settings_fingerprint
from augmentation.outcomes.augmentation_result import AugmentationRejection
from augmentation.outcomes.rejection_reason import AugmentationRejectionReason
from augmentation.variant_lineage import file_sha256, media_variant_id
from mmcrawler_datasets.schema import MultimodalSample

if TYPE_CHECKING:
    from config.augmentation.document_settings import (
        DocumentAugmentationSettings,
    )


class DocumentPageAugmenter:
    def __init__(self, *, settings: DocumentAugmentationSettings) -> None:
        self._settings = settings

    def augment(
        self,
        *,
        sample: MultimodalSample,
        root: Path,
        source_page: Path,
        operation: str,
    ) -> tuple[MultimodalSample | None, AugmentationRejection | None]:
        try:
            artifact = self._render(
                sample=sample,
                root=root,
                source_page=source_page,
                operation=operation,
            )
        except (ImportError, OSError, ValueError) as exc:
            return None, _rejection(
                sample,
                AugmentationRejectionReason.DOCUMENT_PAGE_IMAGE_TRANSFORM_FAILED,
                operation,
                str(exc),
            )
        rejection = validate_page_output(
            path=artifact.output_path,
            mime_type=artifact.mime_type,
            max_bytes=self._settings.output_max_bytes,
            width=artifact.transform.output_width,
            height=artifact.transform.output_height,
        )
        if rejection:
            artifact.output_path.unlink(missing_ok=True)
            return None, _rejection(sample, rejection, operation)
        if (
            operation == "page_image"
            and artifact.output_sha256 == artifact.source_sha256
        ):
            artifact.output_path.unlink(missing_ok=True)
            return None, _rejection(
                sample,
                AugmentationRejectionReason.DOCUMENT_PAGE_IMAGE_UNCHANGED,
                operation,
            )

        metadata = document_variant_metadata(
            sample=sample,
            root=root,
            source_path=source_page,
            output_path=artifact.output_path,
            operation=operation,
            source_sha256=artifact.source_sha256,
            output_sha256=artifact.output_sha256,
            config_hash=artifact.config_hash,
            variant_id=artifact.variant_id,
            parameters=artifact.parameters,
            spatial_receipt=artifact.transform.receipt(),
            output_mime_type=artifact.mime_type,
            output_byte_size=artifact.output_path.stat().st_size,
            modifies=("document_page_image", "spatial_annotations"),
        )
        metadata["document_variant_alignment"] = {
            "variant_kind": "page_only_variant",
            "page_representation_modified": True,
            "text_representation_modified": False,
        }
        if operation == "layout_preserving":
            metadata["document_layout_metadata"] = {
                "validated": True,
                "coordinate_system": "pixel",
                "source_width": artifact.transform.source_width,
                "source_height": artifact.transform.source_height,
                "output_width": artifact.transform.output_width,
                "output_height": artifact.transform.output_height,
                "page_index_preserved": True,
                "reading_order_preserved": True,
                "layout_box_count_preserved": True,
                "pixel_geometry_preserved": True,
            }
        variant = transform_document_page_sample(
            sample=sample,
            variant_id=artifact.variant_id,
            output_path=artifact.output_path,
            mime_type=artifact.mime_type,
            source_path=relative_artifact_path(root, source_page),
            source_sha256=artifact.source_sha256,
            output_sha256=artifact.output_sha256,
            operation=operation,
            parameters=artifact.parameters,
            transform=artifact.transform,
            metadata=metadata,
        )
        return variant, None

    def _render(
        self,
        *,
        sample: MultimodalSample,
        root: Path,
        source_page: Path,
        operation: str,
    ) -> RenderedDocumentPage:
        from PIL import Image, ImageOps

        source_sha256 = file_sha256(path=source_page)
        with Image.open(source_page) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
            source_width, source_height = image.size
            if source_width * source_height > self._settings.page_max_pixels:
                raise ValueError("document_page_too_many_pixels")
            if operation == "page_image":
                ratio = min(
                    self._settings.page_resize_max_width / source_width,
                    self._settings.page_resize_max_height / source_height,
                    0.98,
                )
                width = max(1, round(source_width * ratio))
                height = max(1, round(source_height * ratio))
                output = image.resize(
                    (width, height), Image.Resampling.LANCZOS
                )
                transform = SpatialTransform(
                    source_width=source_width,
                    source_height=source_height,
                    output_width=width,
                    output_height=height,
                    scale_x=width / source_width,
                    scale_y=height / source_height,
                )
                parameters = {
                    "mode": "page_render_resize",
                    "width": width,
                    "height": height,
                    "resampling": "lanczos",
                }
            elif operation == "layout_preserving":
                output = image.copy()
                transform = SpatialTransform(
                    source_width=source_width,
                    source_height=source_height,
                    output_width=source_width,
                    output_height=source_height,
                )
                parameters = {
                    "mode": "layout_preserving_rerender",
                    "pixel_geometry_preserved": True,
                }
            else:
                raise ValueError(
                    f"unsupported document page operation: {operation}"
                )
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
            output_dir = root / self._settings.output_directory
            output_dir.mkdir(parents=True, exist_ok=True)
            suffix = (
                ".webp"
                if self._settings.page_output_format == "webp"
                else ".png"
            )
            mime_type = "image/webp" if suffix == ".webp" else "image/png"
            output_path = output_dir / f"{variant_id}{suffix}"
            temporary = output_path.with_suffix(f".tmp{suffix}")
            if suffix == ".webp":
                output.save(
                    temporary,
                    format="WEBP",
                    quality=self._settings.page_webp_quality,
                    method=4,
                )
            else:
                output.save(temporary, format="PNG", optimize=True)
            temporary.replace(output_path)
        return RenderedDocumentPage(
            output_path=output_path,
            mime_type=mime_type,
            source_sha256=source_sha256,
            output_sha256=file_sha256(path=output_path),
            config_hash=config_hash,
            variant_id=variant_id,
            parameters=parameters,
            transform=transform,
        )


@dataclass(frozen=True, slots=True)
class RenderedDocumentPage:
    output_path: Path
    mime_type: str
    source_sha256: str
    output_sha256: str
    config_hash: str
    variant_id: str
    parameters: dict[str, object]
    transform: SpatialTransform


def validate_page_output(
    *, path: Path, mime_type: str, max_bytes: int, width: int, height: int
) -> AugmentationRejectionReason | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return AugmentationRejectionReason.DOCUMENT_OUTPUT_MISSING
    if path.stat().st_size > max_bytes:
        return AugmentationRejectionReason.DOCUMENT_OUTPUT_TOO_LARGE
    try:
        from PIL import Image

        with Image.open(path) as image:
            image_format = image.format
            if (
                image_format is None
                or Image.MIME.get(image_format) != mime_type
            ):
                return (
                    AugmentationRejectionReason.DOCUMENT_OUTPUT_MIME_MISMATCH
                )
            if image.size != (width, height):
                return AugmentationRejectionReason.DOCUMENT_OUTPUT_DIMENSIONS_MISMATCH
            image.verify()
    except (OSError, ValueError):
        return AugmentationRejectionReason.DOCUMENT_OUTPUT_DECODE_FAILED
    return None


def _rejection(
    sample: MultimodalSample,
    reason: AugmentationRejectionReason,
    variant: str,
    message: str | None = None,
) -> AugmentationRejection:
    return AugmentationRejection(
        reason=reason,
        sample_id=sample.sample_id,
        variant_name=variant,
        modality="document",
        message=message,
    )
