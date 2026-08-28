"""Canonical persisted curated image record."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from mmcrawler_datasets.curated.evidence import (
    AssetContextRecord,
    PrivacyClearanceRecord,
)
from mmcrawler_datasets.curated.strict import (
    mapping_tuple,
    optional_bool_value,
    optional_float_value,
    optional_int_value,
    optional_text_value,
    relative_path,
    require_bool_value,
    require_exact_dataclass_fields,
    require_float_value,
    require_text_value,
)


@dataclass(frozen=True, slots=True)
class CuratedImageRecord:
    """Image artifact plus textual pairing metadata for multimodal datasets."""

    schema_version: str
    snapshot_id: str
    image_id: str
    object_id: str
    source_run_id: str
    media_path: str
    image_mime_type: str | None
    source_url: str
    parent_document_id: str | None
    page_title: str | None
    alt_text: str | None
    figcaption: str | None
    surrounding_text: str | None
    caption_text: str | None
    caption_source: str | None
    caption_quality_score: float
    context_score: float
    ocr_preview: str | None
    image_width: int | None
    image_height: int | None
    image_format: str | None
    image_average_hash: str | None
    split: str | None
    allow_training: bool | None
    license: str | None = None
    license_url: str | None = None
    governance_note: str | None = None
    robots_status: str | None = None
    terms_source: str | None = None
    usage_rules: str | None = None
    ocr_text: str | None = None
    ocr_confidence: float | None = None
    ocr_language: str | None = None
    ocr_quality_score: float | None = None
    ocr_boxes: tuple[dict[str, object], ...] = ()
    ocr_lines: tuple[dict[str, object], ...] = ()
    image_quality_score: float | None = None
    image_blur_variance: float | None = None
    image_aspect_ratio: float | None = None
    image_aspect_ratio_magnitude: float | None = None
    image_orientation: str | None = None
    image_exif_orientation: int | None = None
    image_is_animated: bool | None = None
    image_frame_count: int | None = None
    image_icc_profile_sha256: str | None = None
    image_payload_bytes: int | None = None
    image_difference_hash: str | None = None
    image_phash: str | None = None
    normalized_media_path: str | None = None
    trainable: bool = False
    curated_media_status: str = "metadata_only"
    curated_rejection_reason: str | None = None
    privacy_clearance: PrivacyClearanceRecord | None = None
    asset_context: AssetContextRecord | None = None

    def __post_init__(self) -> None:
        for name in (
            "schema_version",
            "snapshot_id",
            "image_id",
            "object_id",
            "source_run_id",
            "source_url",
            "curated_media_status",
        ):
            require_text_value(getattr(self, name), field_name=name)
        for name in (
            "image_mime_type",
            "parent_document_id",
            "page_title",
            "alt_text",
            "figcaption",
            "surrounding_text",
            "caption_text",
            "caption_source",
            "ocr_preview",
            "image_format",
            "image_average_hash",
            "split",
            "license",
            "license_url",
            "governance_note",
            "robots_status",
            "terms_source",
            "usage_rules",
            "ocr_text",
            "ocr_language",
            "image_orientation",
            "image_icc_profile_sha256",
            "image_difference_hash",
            "image_phash",
            "curated_rejection_reason",
        ):
            optional_text_value(getattr(self, name), field_name=name)
        for name in (
            "caption_quality_score",
            "context_score",
        ):
            require_float_value(getattr(self, name), field_name=name)
        for name in (
            "ocr_confidence",
            "ocr_quality_score",
            "image_quality_score",
            "image_blur_variance",
            "image_aspect_ratio",
            "image_aspect_ratio_magnitude",
        ):
            optional_float_value(getattr(self, name), field_name=name)
        for name in (
            "image_width",
            "image_height",
            "image_exif_orientation",
            "image_frame_count",
            "image_payload_bytes",
        ):
            optional_int_value(
                getattr(self, name),
                field_name=name,
                minimum=0,
            )
        optional_bool_value(
            self.allow_training,
            field_name="allow_training",
        )
        optional_bool_value(
            self.image_is_animated,
            field_name="image_is_animated",
        )
        require_bool_value(self.trainable, field_name="trainable")
        if not isinstance(self.ocr_boxes, tuple):
            raise TypeError("ocr_boxes must be a tuple")
        if not isinstance(self.ocr_lines, tuple):
            raise TypeError("ocr_lines must be a tuple")
        mapping_tuple(self.ocr_boxes, field_name="ocr_boxes")
        mapping_tuple(self.ocr_lines, field_name="ocr_lines")
        if self.privacy_clearance is not None and not isinstance(
            self.privacy_clearance,
            PrivacyClearanceRecord,
        ):
            raise TypeError("privacy_clearance must be typed")
        if self.asset_context is not None and not isinstance(
            self.asset_context,
            AssetContextRecord,
        ):
            raise TypeError("asset_context must be typed")
        object.__setattr__(
            self,
            "media_path",
            relative_path(self.media_path),
        )
        if self.normalized_media_path is not None:
            object.__setattr__(
                self,
                "normalized_media_path",
                relative_path(self.normalized_media_path),
            )

    @classmethod
    def from_dict(
        cls,
        row: Mapping[str, object],
    ) -> CuratedImageRecord:
        """Decode one exact current-schema curated image row."""

        require_exact_dataclass_fields(
            row,
            record_type=cls,
            label="CuratedImage",
        )
        values: dict[str, Any] = dict(row)
        values["ocr_boxes"] = _array_tuple(
            values["ocr_boxes"],
            field_name="ocr_boxes",
        )
        values["ocr_lines"] = _array_tuple(
            values["ocr_lines"],
            field_name="ocr_lines",
        )
        values["privacy_clearance"] = _privacy_clearance(
            values["privacy_clearance"]
        )
        values["asset_context"] = _asset_context(values["asset_context"])
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping."""

        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in {"privacy_clearance", "asset_context"}
        }
        payload["privacy_clearance"] = (
            self.privacy_clearance.to_dict()
            if self.privacy_clearance is not None
            else None
        )
        context = (
            self.asset_context.to_dict()
            if self.asset_context is not None
            else None
        )
        payload["asset_context"] = (
            {key: value for key, value in context.items() if value is not None}
            if context is not None
            else None
        )
        return payload


def _array_tuple(value: object, *, field_name: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be an array")
    return tuple(value)


def _privacy_clearance(value: object) -> PrivacyClearanceRecord | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("privacy_clearance must be an object")
    return PrivacyClearanceRecord.model_validate(dict(value))


def _asset_context(value: object) -> AssetContextRecord | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("asset_context must be an object")
    return AssetContextRecord.from_mapping(value)


__all__ = ["CuratedImageRecord"]
