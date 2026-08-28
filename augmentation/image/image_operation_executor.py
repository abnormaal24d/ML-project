"""Execute one image augmentation operation with cache and validation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from augmentation.generated_artifact_cache import settings_fingerprint
from augmentation.image.content_aware_crop import CropWindow
from augmentation.image.image_artifact_writer import (
    atomic_save_webp,
    metadata_for_save,
    remove_artifact,
)
from augmentation.image.image_difference import image_difference
from augmentation.image.image_operations import apply_image_operation
from augmentation.image.image_variant_assembler import assemble_image_variant
from augmentation.outcomes.augmentation_result import AugmentationRejection
from augmentation.outcomes.rejection_reason import AugmentationRejectionReason
from augmentation.variant_lineage import (
    MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
    file_sha256,
    media_variant_id,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL.Image import Image

    from augmentation.generated_artifact_cache import AugmentationCache
    from augmentation.outcomes.media_validation_outcome import (
        MediaValidationOutcome,
    )
    from config.augmentation.image_settings import ImageAugmentationSettings
    from mmcrawler_datasets.schema import MultimodalSample


@dataclass(frozen=True, slots=True)
class ImageExecutionOutcome:
    variant: MultimodalSample | None = None
    rejection: AugmentationRejection | None = None


class ImageOperationExecutor:
    def __init__(
        self,
        *,
        settings: ImageAugmentationSettings,
        cache: AugmentationCache,
        validate_output: Callable[..., MediaValidationOutcome],
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._validate_output = validate_output

    def execute(
        self,
        *,
        sample: MultimodalSample,
        root: Path,
        source_path: Path,
        source_sha256: str,
        prepared: Image,
        source_info: dict[str, object],
        operation: str,
        variant_operation: str,
        crop_window: CropWindow | None,
    ) -> ImageExecutionOutcome:
        config_hash = settings_fingerprint(
            {
                "settings": self._settings.model_dump(mode="json"),
                "operation": operation,
                "variant_operation": variant_operation,
            }
        )
        variant_id = media_variant_id(
            source_sample_id=sample.sample_id,
            operation=f"image_{variant_operation}",
            source_sha256=source_sha256,
            config_hash=config_hash,
        )
        output_dir = root / self._settings.output_directory
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{variant_id}.webp"
        cache_key = self._cache.cache_key(
            source_path=source_path,
            operation=f"image_{variant_operation}",
            settings_digest=config_hash,
        )
        try:
            result = apply_image_operation(
                image=prepared.copy(),
                operation=operation,
                settings=self._settings,
                seed_key=variant_id,
                crop_window=crop_window,
            )
            restored = self._cache.restore(
                dataset_root=root,
                cache_key=cache_key,
                output_path=output_path,
                expected_metadata={
                    "source_sha256": source_sha256,
                    "config_hash": config_hash,
                    "implementation_hash": MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
                    "operation": operation,
                },
            )
            if not restored:
                atomic_save_webp(
                    image=result.image,
                    output_path=output_path,
                    save_options=result.save_options,
                    metadata=metadata_for_save(
                        source_info, self._settings.metadata_policy
                    ),
                )
            validation = self._validate_output(
                path=output_path,
                expected_mime_type="image/webp",
                expected_width=result.transform.output_width,
                expected_height=result.transform.output_height,
            )
            if not validation.accepted:
                remove_artifact(output_path)
                return ImageExecutionOutcome(
                    rejection=_rejection(
                        sample,
                        AugmentationRejectionReason(
                            validation.rejection_reason
                            or "generated_image_invalid"
                        ),
                        operation,
                    )
                )
            output_sha256 = file_sha256(path=output_path)
            if (
                image_difference(source_path, output_path)
                < self._settings.minimum_image_difference
            ):
                remove_artifact(output_path)
                return ImageExecutionOutcome(
                    rejection=_rejection(
                        sample,
                        AugmentationRejectionReason.IMAGE_VARIANT_SEMANTICALLY_UNCHANGED,
                        operation,
                    )
                )
            self._cache.store(
                dataset_root=root,
                cache_key=cache_key,
                output_path=output_path,
                cache_metadata={
                    "operation": operation,
                    "source_sha256": source_sha256,
                    "output_sha256": output_sha256,
                    "config_hash": config_hash,
                    "parameters": result.parameters,
                    "implementation_hash": MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
                },
            )
        except (OSError, ValueError) as exc:
            remove_artifact(output_path)
            return ImageExecutionOutcome(
                rejection=_rejection(
                    sample,
                    AugmentationRejectionReason.IMAGE_TRANSFORM_FAILED,
                    operation,
                    str(exc),
                )
            )
        variant = assemble_image_variant(
            sample=sample,
            root=root,
            source_path=source_path,
            output_path=output_path,
            operation=operation,
            cache_key=cache_key,
            source_sha256=source_sha256,
            output_sha256=output_sha256,
            config_hash=config_hash,
            variant_id=variant_id,
            parameters=result.parameters,
            transform=result.transform,
        )
        return ImageExecutionOutcome(variant=variant)


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
        modality="image",
        message=message,
    )
