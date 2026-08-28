"""Contract tests for caption validation coverage keys."""

from __future__ import annotations

from types import SimpleNamespace

from crawler.curation.snapshots.dataset_assembly.curated_quality_filter import (
    build_image_coverage,
    build_validation_image_coverage,
)
from mmcrawler_datasets.curated.image import CuratedImageRecord

_CAPTION_COVERAGE_KEYS = (
    "with_html_garbage_caption",
    "with_boilerplate_caption",
)

_LEGACY_REJECTED_KEYS = (
    "rejected_html_garbage_caption",
    "rejected_boilerplate_caption",
)


def _image(caption_text: str | None) -> CuratedImageRecord:
    return CuratedImageRecord(
        schema_version="3.0",
        snapshot_id="snapshot-1",
        image_id="image-1",
        object_id="object-1",
        source_run_id="run-1",
        media_path="media/image-1.png",
        image_mime_type="image/png",
        source_url="https://example.test/image-1.png",
        parent_document_id=None,
        page_title=None,
        alt_text=None,
        figcaption=None,
        surrounding_text=None,
        caption_text=caption_text,
        caption_source="caption",
        caption_quality_score=0.9,
        context_score=0.8,
        ocr_preview=None,
        image_width=100,
        image_height=100,
        image_format="PNG",
        image_average_hash=None,
        split=None,
        allow_training=True,
        license="CC0",
        trainable=True,
        curated_media_status="trainable",
    )


def test_build_image_coverage_publishes_with_caption_keys() -> None:
    images = (
        _image(caption_text="A normal caption"),
        _image(caption_text="next"),
        _image(caption_text="<meta name='description'>"),
        _image(caption_text=None),
    )
    raw_entry = SimpleNamespace(
        record=SimpleNamespace(kind="image"),
    )
    coverage = build_image_coverage(
        images=images,
        raw_entries=(raw_entry, raw_entry, raw_entry, raw_entry),
        dropped_as_duplicate=0,
    )

    for key in _CAPTION_COVERAGE_KEYS:
        assert key in coverage
    assert coverage["with_html_garbage_caption"] == 1
    assert coverage["with_boilerplate_caption"] == 1
    for key in _LEGACY_REJECTED_KEYS:
        assert key not in coverage


def test_build_validation_image_coverage_publishes_with_caption_keys() -> None:
    images = (
        _image(caption_text="A normal caption"),
        _image(caption_text="gallery grid"),
    )
    coverage = build_validation_image_coverage(images=images)

    for key in _CAPTION_COVERAGE_KEYS:
        assert key in coverage
    assert coverage["with_html_garbage_caption"] == 0
    assert coverage["with_boilerplate_caption"] == 1
    for key in _LEGACY_REJECTED_KEYS:
        assert key not in coverage
