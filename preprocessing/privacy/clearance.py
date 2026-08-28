"""Privacy clearance bound to exact inspected and emitted content."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum


class PrivacyClearanceStatus(StrEnum):
    APPROVED = "approved"
    REMEDIATED = "remediated"
    REVIEW_REQUIRED = "review_required"
    REJECTED = "rejected"
    INCOMPLETE = "incomplete"


class ApprovedObjectRole(StrEnum):
    PRIMARY_MEDIA = "primary_media"
    KEYFRAME = "keyframe"
    VISUAL_PROXY = "visual_proxy"
    EDIT_SOURCE = "edit_source"
    EDIT_MASK = "edit_mask"


_RELEASE_SEVERITY = {
    PrivacyClearanceStatus.APPROVED: 0,
    PrivacyClearanceStatus.REMEDIATED: 1,
    PrivacyClearanceStatus.INCOMPLETE: 2,
    PrivacyClearanceStatus.REVIEW_REQUIRED: 3,
    PrivacyClearanceStatus.REJECTED: 4,
}


@dataclass(frozen=True, slots=True)
class ApprovedTextField:
    """One inspected text field that downstream code may safely consume."""

    name: str
    value: str
    input_digest: str
    output_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("approved text field name must be non-empty")
        if not isinstance(self.value, str):
            raise TypeError("approved text field value must be text")
        _require_sha256(self.input_digest, field_name="input_digest")
        _require_sha256(self.output_digest, field_name="output_digest")
        actual_output_digest = hashlib.sha256(
            self.value.encode("utf-8")
        ).hexdigest()
        if self.output_digest != actual_output_digest:
            raise ValueError(
                "approved text output digest does not match value"
            )


@dataclass(frozen=True, slots=True)
class ApprovedObject:
    """One approved primary or derived binary object."""

    object_id: str
    role: ApprovedObjectRole
    output_digest: str
    derived_from_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("approved object_id must be non-empty")
        if not isinstance(self.role, ApprovedObjectRole):
            raise TypeError("approved object role must be typed")
        _require_sha256(self.output_digest, field_name="output_digest")
        if self.derived_from_digest is not None:
            _require_sha256(
                self.derived_from_digest,
                field_name="derived_from_digest",
            )


@dataclass(frozen=True, slots=True)
class PrivacyClearance:
    """Fail-closed evidence for an exact preprocessing output."""

    status: PrivacyClearanceStatus
    input_digest: str
    output_digest: str | None
    checked_fields: frozenset[str]
    required_fields: frozenset[str]
    approved_text_fields: tuple[ApprovedTextField, ...] = ()
    approved_objects: tuple[ApprovedObject, ...] = ()
    inspection_digest: str = ""
    assessment_digest: str = ""
    remediation_verified: bool = False
    derivation_digest: str | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, PrivacyClearanceStatus):
            raise TypeError("privacy clearance status must be typed")
        if not isinstance(self.checked_fields, frozenset) or not all(
            isinstance(value, str) for value in self.checked_fields
        ):
            raise TypeError("checked_fields must be a frozenset of strings")
        if not isinstance(self.required_fields, frozenset) or not all(
            isinstance(value, str) for value in self.required_fields
        ):
            raise TypeError("required_fields must be a frozenset of strings")
        if not isinstance(self.approved_text_fields, tuple) or not all(
            isinstance(value, ApprovedTextField)
            for value in self.approved_text_fields
        ):
            raise TypeError("approved_text_fields must contain typed values")
        if not isinstance(self.approved_objects, tuple) or not all(
            isinstance(value, ApprovedObject)
            for value in self.approved_objects
        ):
            raise TypeError("approved_objects must contain typed values")
        if not isinstance(self.reasons, tuple) or not all(
            isinstance(value, str) for value in self.reasons
        ):
            raise TypeError("privacy reasons must be a tuple of strings")
        if not isinstance(self.remediation_verified, bool):
            raise TypeError("remediation_verified must be boolean")
        _require_sha256(self.input_digest, field_name="input_digest")
        _require_sha256(
            self.inspection_digest,
            field_name="inspection_digest",
        )
        _require_sha256(
            self.assessment_digest,
            field_name="assessment_digest",
        )
        if self.output_digest is not None:
            _require_sha256(self.output_digest, field_name="output_digest")
        if self.derivation_digest is not None:
            _require_sha256(
                self.derivation_digest,
                field_name="derivation_digest",
            )
        if (
            self.status
            in {
                PrivacyClearanceStatus.APPROVED,
                PrivacyClearanceStatus.REMEDIATED,
            }
            and self.output_digest is None
        ):
            raise ValueError(
                "approved privacy clearance requires output_digest"
            )
        if self.status is PrivacyClearanceStatus.REMEDIATED:
            if not self.remediation_verified:
                raise ValueError(
                    "remediated privacy clearance requires verified remediation"
                )
            if (
                any(
                    value.derived_from_digest is not None
                    for value in self.approved_objects
                )
                and self.derivation_digest is None
            ):
                raise ValueError(
                    "binary remediation requires derivation digest"
                )
        elif self.derivation_digest is not None:
            raise ValueError(
                "non-remediated clearance cannot carry derivation digest"
            )
        text_names = tuple(field.name for field in self.approved_text_fields)
        if len(text_names) != len(set(text_names)):
            raise ValueError("approved text field names must be unique")
        object_keys = tuple(
            (approved.object_id, approved.role)
            for approved in self.approved_objects
        )
        if len(object_keys) != len(set(object_keys)):
            raise ValueError("approved object identities must be unique")

    @property
    def complete(self) -> bool:
        return self.required_fields.issubset(self.checked_fields)

    @property
    def permits_training(self) -> bool:
        return (
            self.complete
            and self.output_digest is not None
            and self.status
            in {
                PrivacyClearanceStatus.APPROVED,
                PrivacyClearanceStatus.REMEDIATED,
            }
            and (
                self.status is PrivacyClearanceStatus.APPROVED
                or self.remediation_verified
            )
            and bool(self.inspection_digest)
            and bool(self.assessment_digest)
        )

    def approved_text(self, name: str) -> str | None:
        for field in self.approved_text_fields:
            if field.name == name:
                return field.value
        return None

    def approved_object(
        self,
        *,
        object_id: str,
        role: ApprovedObjectRole,
    ) -> ApprovedObject | None:
        """Return the exact object-level approval for an object and role."""

        for approved in self.approved_objects:
            if approved.object_id == object_id and approved.role == role:
                return approved
        return None

    def block(
        self,
        *,
        status: PrivacyClearanceStatus,
        reason: str,
    ) -> PrivacyClearance:
        """Block release with a typed status and one recorded reason.

        A block never downgrades an already-recorded severity; the reason
        is still recorded so downstream evidence stays complete.
        """

        recorded = tuple(dict.fromkeys((*self.reasons, reason)))
        if _RELEASE_SEVERITY[status] < _RELEASE_SEVERITY[self.status]:
            return replace(self, reasons=recorded)
        return replace(
            self,
            status=status,
            output_digest=None,
            approved_objects=(),
            remediation_verified=False,
            derivation_digest=None,
            reasons=recorded,
        )

    def mark_incomplete(self, *, reason: str) -> PrivacyClearance:
        """Record an incomplete inspection without releasing anything."""

        return self.block(
            status=PrivacyClearanceStatus.INCOMPLETE,
            reason=reason,
        )

    def require_review(self, *, reason: str) -> PrivacyClearance:
        """Block release until a human confirms the recorded reason."""

        return self.block(
            status=PrivacyClearanceStatus.REVIEW_REQUIRED,
            reason=reason,
        )

    def reject(self, *, reason: str) -> PrivacyClearance:
        """Permanently reject release for the recorded reason."""

        return self.block(
            status=PrivacyClearanceStatus.REJECTED, reason=reason
        )

    def bind_output(self, *, digest: str | None) -> PrivacyClearance:
        """Bind the exact emitted output digest when training is permitted."""

        if not self.permits_training:
            return self
        return replace(self, output_digest=digest)

    def bind_verified_remediation(
        self,
        *,
        input_digest: str,
        output_digest: str,
        derivation_digest: str | None,
    ) -> PrivacyClearance:
        """Promote clearance with stable verified derivation provenance."""

        if self.status in {
            PrivacyClearanceStatus.REJECTED,
            PrivacyClearanceStatus.INCOMPLETE,
            PrivacyClearanceStatus.REVIEW_REQUIRED,
        }:
            return self
        return replace(
            self,
            status=PrivacyClearanceStatus.REMEDIATED,
            input_digest=input_digest,
            output_digest=output_digest,
            remediation_verified=True,
            derivation_digest=derivation_digest,
        )

    def approve_object(
        self,
        *,
        object_id: str,
        role: ApprovedObjectRole,
        output_digest: str,
        derived_from_digest: str | None = None,
    ) -> PrivacyClearance:
        """Approve one emitted object for training, deduplicated by identity."""

        if not self.permits_training:
            raise ValueError("privacy clearance does not permit training")
        retained = tuple(
            approved
            for approved in self.approved_objects
            if not (approved.object_id == object_id and approved.role == role)
        )
        return replace(
            self,
            approved_objects=(
                *retained,
                ApprovedObject(
                    object_id=object_id,
                    role=role,
                    output_digest=output_digest,
                    derived_from_digest=derived_from_digest,
                ),
            ),
        )

    def bind_training_text(
        self,
        value: str,
        *,
        source_name: str | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> PrivacyClearance:
        """Bind one exact emitted training string to approved source text."""

        if not self.permits_training:
            raise ValueError("privacy clearance does not permit training")
        source = self._approved_source(
            value=value,
            source_name=source_name,
            start=start,
            end=end,
        )
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        training_field = ApprovedTextField(
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
        return replace(
            self,
            approved_text_fields=(*retained, training_field),
        )

    def _approved_source(
        self,
        *,
        value: str,
        source_name: str | None,
        start: int | None,
        end: int | None,
    ) -> ApprovedTextField:
        if source_name is not None:
            candidates = tuple(
                field
                for field in self.approved_text_fields
                if field.name == source_name
            )
        else:
            candidates = self.approved_text_fields
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

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["checked_fields"] = sorted(self.checked_fields)
        payload["required_fields"] = sorted(self.required_fields)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PrivacyClearance":
        expected_keys = {
            "status",
            "input_digest",
            "output_digest",
            "checked_fields",
            "required_fields",
            "approved_text_fields",
            "approved_objects",
            "inspection_digest",
            "assessment_digest",
            "remediation_verified",
            "derivation_digest",
            "reasons",
        }
        if set(payload) != expected_keys:
            raise ValueError("privacy clearance fields are incomplete")
        raw_fields = _mapping_sequence(
            payload,
            "approved_text_fields",
        )
        for item in raw_fields:
            if set(item) != {
                "name",
                "value",
                "input_digest",
                "output_digest",
            }:
                raise ValueError("approved text field schema is invalid")
        approved_fields = tuple(
            ApprovedTextField(
                name=_required_string(item, "name"),
                value=_required_string(item, "value", allow_empty=True),
                input_digest=_required_string(item, "input_digest"),
                output_digest=_required_string(item, "output_digest"),
            )
            for item in raw_fields
        )
        raw_objects = _mapping_sequence(payload, "approved_objects")
        for item in raw_objects:
            if set(item) != {
                "object_id",
                "role",
                "output_digest",
                "derived_from_digest",
            }:
                raise ValueError("approved object schema is invalid")
        approved_objects = tuple(
            ApprovedObject(
                object_id=_required_string(item, "object_id"),
                role=ApprovedObjectRole(_required_string(item, "role")),
                output_digest=_required_string(item, "output_digest"),
                derived_from_digest=_optional_string(
                    item,
                    "derived_from_digest",
                ),
            )
            for item in raw_objects
        )
        status = _required_string(payload, "status")
        output_digest = _optional_string(payload, "output_digest")
        remediation_verified = payload.get("remediation_verified")
        if not isinstance(remediation_verified, bool):
            raise ValueError("remediation_verified must be boolean")
        derivation_digest = _optional_string(payload, "derivation_digest")
        return cls(
            status=PrivacyClearanceStatus(status),
            input_digest=_required_string(payload, "input_digest"),
            output_digest=output_digest,
            checked_fields=frozenset(
                _string_sequence(payload, "checked_fields")
            ),
            required_fields=frozenset(
                _string_sequence(payload, "required_fields")
            ),
            approved_text_fields=approved_fields,
            approved_objects=approved_objects,
            inspection_digest=_required_string(
                payload,
                "inspection_digest",
            ),
            assessment_digest=_required_string(
                payload,
                "assessment_digest",
            ),
            remediation_verified=remediation_verified,
            derivation_digest=derivation_digest,
            reasons=_string_sequence(payload, "reasons"),
        )


def _mapping_sequence(
    payload: Mapping[str, object],
    key: str,
) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"privacy clearance {key} must be an array")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"privacy clearance {key} contains invalid values")
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_sequence(
    payload: Mapping[str, object],
    key: str,
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"privacy clearance {key} must contain strings")
    return tuple(value)


def _required_string(
    payload: Mapping[str, object],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"privacy clearance {key} must be text")
    return value


def _optional_string(
    payload: Mapping[str, object],
    key: str,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"privacy clearance {key} must be non-empty text")
    return value


def _require_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a SHA-256 digest") from exc


__all__ = [
    "ApprovedObject",
    "ApprovedObjectRole",
    "ApprovedTextField",
    "PrivacyClearance",
    "PrivacyClearanceStatus",
]
