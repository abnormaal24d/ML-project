from __future__ import annotations

from functools import partial
from pathlib import Path

from PIL import Image

from augmentation.document.document_augmenter import DocumentAugmenter
from augmentation.document.document_page_augmenter import DocumentPageAugmenter
from augmentation.document.document_text_augmenter import DocumentTextAugmenter
from augmentation.generated_artifact_cache import AugmentationCache
from augmentation.image.content_aware_crop import select_crop_windows
from augmentation.image.image_augmentation_validation import (
    validate_image_input,
    validate_image_output,
)
from augmentation.image.image_augmenter import ImageAugmenter
from augmentation.image.image_operation_executor import ImageOperationExecutor
from config.augmentation.document_settings import DocumentAugmentationSettings
from config.augmentation.image_settings import ImageAugmentationSettings
from mmcrawler_datasets.schema import (
    BoundingBox,
    LayoutBox,
    ModalityObject,
    MultimodalSample,
    ObjectBox,
)


class Logger:
    def debug(self, *args, **kwargs):
        return None


def test_crop_strategy_selects_configured_ranking_policy() -> None:
    image = Image.new("RGB", (128, 96), (80, 120, 160))
    sample = MultimodalSample(
        sample_id="crop-policy",
        object_boxes=(
            ObjectBox("object", "item", BoundingBox(0.1, 0.1, 0.2, 0.2)),
        ),
    )
    select = partial(
        select_crop_windows,
        image=image,
        sample=sample,
        width=64,
        height=64,
        candidate_count=9,
        variant_count=2,
        minimum_annotation_coverage=0.0,
        seed_key=sample.sample_id,
    )
    annotation_ranked = select(strategy="annotation_aware")
    entropy_ranked = select(strategy="entropy")

    assert annotation_ranked
    assert entropy_ranked
    assert {window.strategy for window in annotation_ranked} == {
        "annotation_aware"
    }
    assert {window.strategy for window in entropy_ranked} == {"entropy"}


def test_image_operations_are_separate_and_transform_boxes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (1000, 500), (80, 120, 160)).save(source)
    settings = ImageAugmentationSettings(
        enabled=True,
        operations=(
            "resize",
            "content_aware_crop",
            "compression",
            "color_jitter",
            "blur",
        ),
        crop_width=400,
        crop_height=400,
    )
    cache = AugmentationCache(
        enabled=False,
        cache_directory=".cache/augmentation",
        logger=Logger(),
    )
    output_validator = partial(validate_image_output, settings=settings)
    augmenter = ImageAugmenter(
        settings=settings,
        operation_executor=ImageOperationExecutor(
            settings=settings,
            cache=cache,
            validate_output=output_validator,
        ),
        validate_input=partial(
            validate_image_input,
            settings=settings,
            allowed_mime_types=frozenset({"image/png"}),
            max_input_bytes=1_000_000,
        ),
        logger=Logger(),
    )
    sample = MultimodalSample(
        sample_id="image-source",
        image=ModalityObject(path=source, mime_type="image/png"),
        object_boxes=(
            ObjectBox("object", "item", BoundingBox(0.4, 0.2, 0.2, 0.4)),
        ),
    )
    variants, rejections = augmenter.augment(
        sample=sample, dataset_root=tmp_path
    )
    assert all(
        item.reason == "image_variant_semantically_unchanged"
        for item in rejections
    )
    assert {
        (
            "content_aware_crop"
            if name.startswith("content_aware_crop_")
            else name
        )
        for name, _ in variants
    } <= set(settings.operations)
    assert len({variant.sample_id for _, variant in variants}) == len(variants)
    for name, variant in variants:
        assert variant.metadata["augmentation_operation"] == (
            "content_aware_crop"
            if name.startswith("content_aware_crop_")
            else name
        )
        assert variant.metadata["augmentation_parameters"]
        assert variant.metadata["augmentation_spatial_transform"]
        assert variant.object_boxes
        assert (
            variant.image
            and variant.image.path
            and variant.image.path.stat().st_size <= settings.output_max_bytes
        )


def test_document_page_is_real_transform_and_ocr_is_normalization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (800, 600), (240, 240, 240)).save(source)
    settings = DocumentAugmentationSettings(
        enabled=True,
        mode="document_media",
        operations=("page_image", "layout_preserving", "ocr_normalization"),
    )
    sample = MultimodalSample(
        sample_id="document-source",
        text="ofﬁce  text",
        image=ModalityObject(path=source, mime_type="image/png"),
        layout_boxes=(
            LayoutBox(text="office", box=BoundingBox(0.1, 0.1, 0.5, 0.2)),
        ),
        metadata={"modality": "document"},
    )
    variants, rejections = DocumentAugmenter(
        settings=settings,
        text_augmenter=DocumentTextAugmenter(settings=settings),
        page_augmenter=DocumentPageAugmenter(settings=settings),
        logger=Logger(),
    ).augment(sample=sample, dataset_root=tmp_path)
    assert not rejections
    by_name = dict(variants)
    page = by_name["document_page_image"]
    assert page.image and page.image.path
    assert page.image.path.read_bytes() != source.read_bytes()
    assert page.metadata["augmentation_output_mime_type"] == "image/webp"
    assert page.layout_boxes
    normalized = by_name["document_ocr_normalization"]
    assert normalized.text == "office text"
    assert normalized.metadata["augmentation_operation"] == "ocr_normalization"


def test_image_augmentation_rejects_source_outside_dataset_root(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(outside)
    settings = ImageAugmentationSettings(
        enabled=True,
        operations=("resize",),
    )
    cache = AugmentationCache(
        enabled=False,
        cache_directory=".cache/augmentation",
        logger=Logger(),
    )
    output_validator = partial(validate_image_output, settings=settings)
    augmenter = ImageAugmenter(
        settings=settings,
        operation_executor=ImageOperationExecutor(
            settings=settings,
            cache=cache,
            validate_output=output_validator,
        ),
        validate_input=partial(
            validate_image_input,
            settings=settings,
            allowed_mime_types=frozenset({"image/png"}),
            max_input_bytes=1_000_000,
        ),
        logger=Logger(),
    )
    sample = MultimodalSample(
        sample_id="outside-image",
        image=ModalityObject(path=outside, mime_type="image/png"),
    )

    variants, rejections = augmenter.augment(
        sample=sample,
        dataset_root=dataset_root,
    )

    assert variants == ()
    assert rejections
    assert rejections[0].message == "image_source_path_escapes_dataset_root"


def test_document_augmentation_does_not_read_page_outside_dataset_root(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside = tmp_path / "outside-page.png"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(outside)
    sample = MultimodalSample(
        sample_id="outside-document",
        text="document",
        image=ModalityObject(path=outside, mime_type="image/png"),
        metadata={"modality": "document"},
    )
    settings = DocumentAugmentationSettings(
        enabled=True,
        mode="document_media",
        operations=("page_image",),
    )

    variants, rejections = DocumentAugmenter(
        settings=settings,
        text_augmenter=DocumentTextAugmenter(settings=settings),
        page_augmenter=DocumentPageAugmenter(settings=settings),
        logger=Logger(),
    ).augment(sample=sample, dataset_root=dataset_root)

    assert variants == ()
    assert rejections
    assert rejections[0].reason == "document_page_image_missing"
