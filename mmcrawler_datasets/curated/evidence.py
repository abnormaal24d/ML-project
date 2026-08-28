"""Canonical persisted evidence shared by curated record types."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Literal

from pydantic import field_validator, model_validator

from mmcrawler_datasets.curated.strict import (
    NonEmptyText,
    Sha256Text,
    StrictContractModel,
)


class ApprovedTextFieldRecord(StrictContractModel):
    """Wire-safe evidence for one approved text field."""

    name: NonEmptyText
    value: str
    input_digest: Sha256Text
    output_digest: Sha256Text

    @model_validator(mode="after")
    def _validate_output_digest(self) -> ApprovedTextFieldRecord:
        actual = hashlib.sha256(self.value.encode("utf-8")).hexdigest()
        if actual != self.output_digest:
            raise ValueError(
                "approved text output digest does not match value"
            )
        return self


ApprovedObjectRole = Literal[
    "primary_media",
    "keyframe",
    "visual_proxy",
    "edit_source",
    "edit_mask",
]


class ApprovedObjectRecord(StrictContractModel):
    """Wire-safe evidence for one approved binary object."""

    object_id: NonEmptyText
    role: ApprovedObjectRole
    output_digest: Sha256Text
    derived_from_digest: Sha256Text | None


PrivacyStatus = Literal[
    "approved",
    "remediated",
    "review_required",
    "rejected",
    "incomplete",
]


class PrivacyClearanceRecord(StrictContractModel):
    """Neutral persisted privacy evidence used by crawler and training."""

    status: PrivacyStatus
    input_digest: Sha256Text
    output_digest: Sha256Text | None
    checked_fields: tuple[NonEmptyText, ...]
    required_fields: tuple[NonEmptyText, ...]
    approved_text_fields: tuple[ApprovedTextFieldRecord, ...]
    approved_objects: tuple[ApprovedObjectRecord, ...]
    inspection_digest: Sha256Text
    assessment_digest: Sha256Text
    remediation_verified: bool
    derivation_digest: Sha256Text | None = None
    reasons: tuple[str, ...]

    @field_validator(
        "checked_fields",
        "required_fields",
        "approved_text_fields",
        "approved_objects",
        "reasons",
        mode="before",
    )
    @classmethod
    def _array_to_tuple(cls, value: object) -> object:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        raise TypeError("privacy clearance collection must be an array")

    @model_validator(mode="after")
    def _validate_evidence(self) -> PrivacyClearanceRecord:
        if len(self.checked_fields) != len(set(self.checked_fields)):
            raise ValueError("checked privacy fields must be unique")
        if len(self.required_fields) != len(set(self.required_fields)):
            raise ValueError("required privacy fields must be unique")
        if tuple(sorted(self.checked_fields)) != self.checked_fields:
            raise ValueError("checked privacy fields must be sorted")
        if tuple(sorted(self.required_fields)) != self.required_fields:
            raise ValueError("required privacy fields must be sorted")
        text_names = tuple(field.name for field in self.approved_text_fields)
        if len(text_names) != len(set(text_names)):
            raise ValueError("approved text field names must be unique")
        object_keys = tuple(
            (approved.object_id, approved.role)
            for approved in self.approved_objects
        )
        if len(object_keys) != len(set(object_keys)):
            raise ValueError("approved object identities must be unique")
        if (
            self.status in {"approved", "remediated"}
            and self.output_digest is None
        ):
            raise ValueError(
                "approved privacy clearance requires output_digest"
            )
        if self.status == "remediated" and not self.remediation_verified:
            raise ValueError(
                "remediated privacy clearance requires verified remediation"
            )
        if self.status != "remediated" and self.derivation_digest is not None:
            raise ValueError(
                "non-remediated clearance cannot carry derivation digest"
            )
        if (
            self.status == "remediated"
            and any(
                value.derived_from_digest is not None
                for value in self.approved_objects
            )
            and self.derivation_digest is None
        ):
            raise ValueError("binary remediation requires derivation digest")
        return self

    @property
    def complete(self) -> bool:
        return set(self.required_fields).issubset(self.checked_fields)

    @property
    def permits_training(self) -> bool:
        return (
            self.complete
            and self.output_digest is not None
            and self.status in {"approved", "remediated"}
            and (self.status == "approved" or self.remediation_verified)
            and bool(self.inspection_digest)
            and bool(self.assessment_digest)
        )

    def approved_text(self, name: str) -> str | None:
        for field in self.approved_text_fields:
            if field.name == name:
                return field.value
        return None

    def bind_training_text(
        self,
        value: str,
        *,
        source_name: str | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> PrivacyClearanceRecord:
        """Bind exact emitted training text to approved source evidence."""

        if not self.permits_training:
            raise ValueError("privacy clearance does not permit training")
        source = self._approved_source(
            value=value,
            source_name=source_name,
            start=start,
            end=end,
        )
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        training_field = ApprovedTextFieldRecord(
            name="training_text",
            value=value,
            input_digest=source.output_digest,
            output_digest=digest,
        )
        retained = tuple(
            field
            for field in self.approved_text_fields
            if field.name != training_field.name
        )
        payload = self.model_dump(mode="python")
        payload["approved_text_fields"] = (*retained, training_field)
        return PrivacyClearanceRecord.model_validate(payload)

    def _approved_source(
        self,
        *,
        value: str,
        source_name: str | None,
        start: int | None,
        end: int | None,
    ) -> ApprovedTextFieldRecord:
        candidates = (
            tuple(
                field
                for field in self.approved_text_fields
                if field.name == source_name
            )
            if source_name is not None
            else self.approved_text_fields
        )
        for field in candidates:
            if start is not None or end is not None:
                if start is None or end is None:
                    raise ValueError("both text slice bounds are required")
                if start < 0 or end < start:
                    raise ValueError("invalid approved text slice")
                if field.value[start:end] == value:
                    return field
                continue
            if field.value == value:
                return field
        raise ValueError("training text is not bound to approved source text")


class AssetContextRecord(StrictContractModel):
    """Neutral persisted non-content lineage for one media artifact."""

    safety_status: NonEmptyText
    fetch_record_id: NonEmptyText | None
    parent_fetch_record_id: NonEmptyText | None
    parent_stable_url_id: NonEmptyText | None
    media_identity: NonEmptyText | None
    fetch_mode: NonEmptyText | None
    asset_fetch_mode: NonEmptyText | None
    source_page_url: NonEmptyText | None
    embed_host: NonEmptyText | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AssetContextRecord:
        """Build the exact wire shape from a sparse trusted domain mapping."""

        unknown = set(value).difference(cls.model_fields)
        if unknown:
            raise ValueError(
                f"asset context contains unknown fields: {sorted(unknown)}"
            )
        return cls.model_validate(
            {name: value.get(name) for name in cls.model_fields}
        )


__all__ = [
    "ApprovedObjectRecord",
    "ApprovedObjectRole",
    "ApprovedTextFieldRecord",
    "AssetContextRecord",
    "PrivacyClearanceRecord",
    "PrivacyStatus",
]
