"""Canonical strict wire contracts for curated document records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from mmcrawler_datasets.curated.evidence import PrivacyClearanceRecord
from mmcrawler_datasets.curated.strict import (
    optional_bool_value,
    optional_float_value,
    optional_text_value,
    relative_path,
    require_bool_value,
    require_float_value,
    require_int_value,
    require_text_value,
    string_tuple,
)


@dataclass(frozen=True, slots=True)
class CuratedDocumentRecord:
    """Normalized persisted document record."""

    schema_version: str
    snapshot_id: str
    document_id: str
    source_run_id: str
    source_fetch_record_id: str
    object_id: str
    requested_url: str
    final_url: str
    normalized_url: str
    domain: str
    path: str
    modality: str
    language: str | None
    title: str | None
    text_path: str
    markdown_path: str | None
    raw_storage_path: str
    raw_byte_size: int
    extracted_char_count: int
    extracted_token_count_estimate: int
    boilerplate_ratio: float
    code_block_count: int
    quality_score: float
    quality_bucket: str
    rejection_reason: str | None
    content_role: str
    discovery_useful: bool
    exact_duplicate_key: str
    near_duplicate_cluster_id: str | None
    is_near_duplicate: bool
    license: str | None
    license_url: str | None
    allow_training: bool | None
    created_at: str
    governance_note: str | None = None
    language_confidence: float | None = None
    language_script: str | None = None
    robots_status: str | None = None
    terms_source: str | None = None
    usage_rules: str | None = None
    privacy_clearance: PrivacyClearanceRecord | None = None

    def __post_init__(self) -> None:
        for name in (
            "schema_version",
            "snapshot_id",
            "document_id",
            "source_run_id",
            "source_fetch_record_id",
            "object_id",
            "requested_url",
            "final_url",
            "normalized_url",
            "domain",
            "path",
            "modality",
            "raw_storage_path",
            "quality_bucket",
            "content_role",
            "exact_duplicate_key",
            "created_at",
        ):
            require_text_value(getattr(self, name), field_name=name)
        for name in (
            "language",
            "title",
            "markdown_path",
            "rejection_reason",
            "near_duplicate_cluster_id",
            "license",
            "license_url",
            "governance_note",
            "language_script",
            "robots_status",
            "terms_source",
            "usage_rules",
        ):
            optional_text_value(getattr(self, name), field_name=name)
        for name in (
            "raw_byte_size",
            "extracted_char_count",
            "extracted_token_count_estimate",
            "code_block_count",
        ):
            require_int_value(
                getattr(self, name),
                field_name=name,
                minimum=0,
            )
        require_float_value(
            self.boilerplate_ratio,
            field_name="boilerplate_ratio",
        )
        require_float_value(
            self.quality_score,
            field_name="quality_score",
        )
        optional_float_value(
            self.language_confidence,
            field_name="language_confidence",
        )
        require_bool_value(
            self.discovery_useful,
            field_name="discovery_useful",
        )
        require_bool_value(
            self.is_near_duplicate,
            field_name="is_near_duplicate",
        )
        optional_bool_value(
            self.allow_training,
            field_name="allow_training",
        )
        object.__setattr__(
            self,
            "text_path",
            relative_path(self.text_path),
        )
        if self.privacy_clearance is not None and not isinstance(
            self.privacy_clearance,
            PrivacyClearanceRecord,
        ):
            raise TypeError("privacy_clearance must be typed")

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> CuratedDocumentRecord:
        """Decode one exact current-schema curated document row."""

        values: dict[str, Any] = dict(row)
        clearance = values.get("privacy_clearance")
        if clearance is not None:
            if not isinstance(clearance, Mapping):
                raise ValueError("privacy_clearance must be an object")
            values["privacy_clearance"] = (
                PrivacyClearanceRecord.model_validate(clearance)
            )
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("invalid document record") from exc

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping."""

        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "privacy_clearance"
        }
        payload["privacy_clearance"] = (
            self.privacy_clearance.to_dict()
            if self.privacy_clearance is not None
            else None
        )
        return payload


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """Train-ready text chunk with lineage back to curated documents."""

    schema_version: str
    snapshot_id: str
    chunk_id: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count_estimate: int
    text: str
    language: str | None
    title: str | None
    section_path: tuple[str, ...]
    quality_score: float
    exact_duplicate_key: str
    near_duplicate_cluster_id: str | None
    split: str | None

    def __post_init__(self) -> None:
        for name in (
            "schema_version",
            "snapshot_id",
            "chunk_id",
            "document_id",
            "text",
            "exact_duplicate_key",
        ):
            require_text_value(getattr(self, name), field_name=name)
        for name in (
            "language",
            "title",
            "near_duplicate_cluster_id",
            "split",
        ):
            optional_text_value(getattr(self, name), field_name=name)
        for name in (
            "chunk_index",
            "start_char",
            "end_char",
            "token_count_estimate",
        ):
            require_int_value(
                getattr(self, name),
                field_name=name,
                minimum=0,
            )
        require_float_value(self.quality_score, field_name="quality_score")
        if not isinstance(self.section_path, tuple):
            raise TypeError("section_path must be a tuple")
        string_tuple(self.section_path, field_name="section_path")
        if self.end_char < self.start_char:
            raise ValueError("end_char must not precede start_char")

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> ChunkRecord:
        """Decode one exact current-schema curated text chunk row."""

        values: dict[str, Any] = dict(row)
        section_path = values.get("section_path")
        if isinstance(section_path, list):
            values["section_path"] = tuple(section_path)
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("invalid chunk record") from exc

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping."""

        return asdict(self)


__all__ = ["ChunkRecord", "CuratedDocumentRecord"]
