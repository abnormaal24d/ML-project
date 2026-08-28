"""Inspect and, when safe, remediate independent text fields once."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from preprocessing.preprocessing_input import PreprocessingInput
from preprocessing.privacy.clearance import (
    ApprovedTextField,
    PrivacyClearance,
    PrivacyClearanceStatus,
)
from preprocessing.privacy.public_provenance import public_source_url
from preprocessing.privacy.text_privacy import PiiDetectionResult, PiiDetector


@dataclass(frozen=True, slots=True)
class InspectedFields:
    values: dict[str, str]
    clearance: PrivacyClearance
    finding_counts: dict[str, int]
    spans: tuple[dict[str, object], ...]

    def has_findings_for_prefixes(self, prefixes: tuple[str, ...]) -> bool:
        """Return whether any finding belongs to one of the field prefixes."""

        return any(
            str(span.get("field") or "").startswith(prefixes)
            for span in self.spans
        )


def inspect_text_fields_for_release(
    *,
    fields: dict[str, str | None],
    detector: PiiDetector,
    required_fields: frozenset[str] | None = None,
    evidence_fields: frozenset[str] = frozenset(),
    input_digest: str | None = None,
    output_digest: str | None = None,
    released_fields: frozenset[str] | None = None,
) -> InspectedFields:
    """Return only fields that passed inspection or verified remediation."""

    normalized = {
        name: value.strip()
        for name, value in fields.items()
        if isinstance(value, str) and value.strip()
    }
    required = (
        frozenset(normalized) if required_fields is None else required_fields
    )
    released = (
        frozenset(normalized) if released_fields is None else released_fields
    )
    checked = set(evidence_fields)
    approved: list[ApprovedTextField] = []
    values: dict[str, str] = {}
    all_counts: dict[str, int] = {}
    all_spans: list[dict[str, object]] = []
    reasons: list[str] = []
    remediated = False
    rejected = False
    review_required = False
    incomplete = False

    for name, value in normalized.items():
        result = detector.detect(text=value)
        checked.add(name)
        _merge_counts(all_counts, result)
        all_spans.extend({**span, "field": name} for span in result.spans)
        if not result.inspection_complete:
            incomplete = True
            reasons.append(f"inspection_incomplete:{name}")
            continue
        if result.has_secret_findings or result.assessment_outcome in {
            "reject",
            "escalate",
        }:
            rejected = True
            reasons.append(f"restricted_finding:{name}")
            continue
        if result.assessment_outcome in {"review", "quarantine"}:
            review_required = True
            reasons.append(f"review_required:{name}")
            continue

        released_value = value
        if result.assessment_outcome == "remediate" or result.has_findings:
            released_value = detector.redact(text=value, result=result)
            residual = detector.detect(text=released_value)
            if not residual.inspection_complete:
                incomplete = True
                reasons.append(f"residual_inspection_incomplete:{name}")
                continue
            if residual.has_findings:
                review_required = True
                reasons.append(f"residual_findings:{name}")
                continue
            remediated = remediated or released_value != value

        if name in released:
            values[name] = released_value
            approved.append(
                ApprovedTextField(
                    name=name,
                    value=released_value,
                    input_digest=_text_digest(value),
                    output_digest=_text_digest(released_value),
                )
            )

    missing = required - checked
    if missing:
        incomplete = True
        reasons.extend(f"unchecked:{name}" for name in sorted(missing))

    if rejected:
        status = PrivacyClearanceStatus.REJECTED
    elif review_required:
        status = PrivacyClearanceStatus.REVIEW_REQUIRED
    elif incomplete:
        status = PrivacyClearanceStatus.INCOMPLETE
    elif remediated:
        status = PrivacyClearanceStatus.REMEDIATED
    else:
        status = PrivacyClearanceStatus.APPROVED

    canonical_input = input_digest or _fields_digest(normalized)
    canonical_output = output_digest or _fields_digest(values)
    inspection_digest = _inspection_digest(
        checked_fields=frozenset(checked),
        required_fields=required,
        finding_counts=all_counts,
        reasons=tuple(reasons),
    )
    assessment_digest = _assessment_digest(
        status=status,
        inspection_digest=inspection_digest,
        remediated=remediated,
    )
    clearance = PrivacyClearance(
        status=(
            PrivacyClearanceStatus.REMEDIATED
            if remediated
            else PrivacyClearanceStatus.APPROVED
        ),
        input_digest=canonical_input,
        output_digest=canonical_output,
        checked_fields=frozenset(checked),
        required_fields=required,
        approved_text_fields=tuple(
            sorted(approved, key=lambda item: item.name)
        ),
        inspection_digest=inspection_digest,
        assessment_digest=assessment_digest,
        remediation_verified=remediated,
    )
    for reason in reasons:
        if reason.startswith("restricted_finding:"):
            clearance = clearance.reject(reason=reason)
        elif reason.startswith(("review_required:", "residual_findings:")):
            clearance = clearance.require_review(reason=reason)
        else:
            clearance = clearance.mark_incomplete(reason=reason)
    return InspectedFields(
        values=values,
        clearance=clearance,
        finding_counts=all_counts,
        spans=tuple(all_spans),
    )


def inspect_media_fields_for_release(
    *,
    item: PreprocessingInput,
    fields: Mapping[str, str | None],
    detector: PiiDetector,
    evidence_fields: frozenset[str],
    required_evidence_fields: frozenset[str],
    input_digest: str | None,
    output_digest: str | None,
) -> InspectedFields:
    """Collect and inspect every releasable text field of one media item."""

    raw_source_url = item.normalized_url or item.source_url
    all_fields = {
        **metadata_text_fields(item=item),
        **canonical_payload_fields(item=item),
        **fields,
        "source_url": public_source_url(raw_source_url),
        "source_url_raw": raw_source_url,
        "path": item.path,
        "title": item.title,
    }
    present_fields = frozenset(
        name
        for name, value in all_fields.items()
        if isinstance(value, str) and value.strip()
    )
    required_fields = present_fields | required_evidence_fields
    return inspect_text_fields_for_release(
        fields=all_fields,
        detector=detector,
        required_fields=required_fields,
        evidence_fields=evidence_fields,
        input_digest=input_digest or _EMPTY_SHA256,
        output_digest=output_digest or _EMPTY_SHA256,
        released_fields=frozenset(
            name
            for name in present_fields
            if name != "source_url_raw" and not name.startswith("metadata:")
        ),
    )


def text_payload_fields(
    *,
    item: PreprocessingInput,
    names: Iterable[str],
) -> dict[str, str | None]:
    """Read content text only; these values never count as inspection proof."""

    fields: dict[str, str | None] = {}
    for name in names:
        value = item.payload.get(name)
        if isinstance(value, str):
            fields[name] = value
    return fields


def metadata_text_fields(*, item: PreprocessingInput) -> dict[str, str]:
    """Flatten metadata values without exposing attacker-controlled keys."""

    fields: dict[str, str] = {}
    _flatten_metadata(
        value=item.payload,
        prefix="metadata",
        target=fields,
        remaining=[256],
        counter=[0],
    )
    return fields


def canonical_payload_fields(
    *, item: PreprocessingInput
) -> dict[str, str | None]:
    names = (
        "source_page_url",
        "embed_host",
        "license",
        "license_url",
        "governance_note",
        "robots_status",
        "terms_source",
        "usage_rules",
        "training_reason",
    )
    return {
        name: value
        for name in names
        if isinstance((value := item.payload.get(name)), str) and value.strip()
    }


def _flatten_metadata(
    *,
    value: object,
    prefix: str,
    target: dict[str, str],
    remaining: list[int],
    counter: list[int],
) -> None:
    if remaining[0] <= 0:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).startswith("privacy_"):
                continue
            _flatten_metadata(
                value=item,
                prefix=prefix,
                target=target,
                remaining=remaining,
                counter=counter,
            )
        return
    if isinstance(value, (list, tuple)):
        for item in value[:32]:
            _flatten_metadata(
                value=item,
                prefix=prefix,
                target=target,
                remaining=remaining,
                counter=counter,
            )
        return
    if isinstance(value, str) and value.strip():
        name = f"{prefix}:{counter[0]}"
        counter[0] += 1
        target[name] = value.strip()[:4096]
        remaining[0] -= 1


_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _merge_counts(target: dict[str, int], result: PiiDetectionResult) -> None:
    for name, count in result.finding_counts.items():
        target[name] = target.get(name, 0) + count


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fields_digest(fields: dict[str, str]) -> str:
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inspection_digest(
    *,
    checked_fields: frozenset[str],
    required_fields: frozenset[str],
    finding_counts: dict[str, int],
    reasons: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "checked_fields": sorted(checked_fields),
            "required_fields": sorted(required_fields),
            "finding_counts": finding_counts,
            "reasons": list(reasons),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assessment_digest(
    *,
    status: PrivacyClearanceStatus,
    inspection_digest: str,
    remediated: bool,
) -> str:
    payload = f"{status.value}\n{inspection_digest}\n{int(remediated)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "InspectedFields",
    "canonical_payload_fields",
    "inspect_media_fields_for_release",
    "inspect_text_fields_for_release",
    "metadata_text_fields",
    "text_payload_fields",
]
