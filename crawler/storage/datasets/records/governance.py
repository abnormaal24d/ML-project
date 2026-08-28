"""Strict, fail-closed governance schema and data-record assembly.

Governance evidence is validated at creation time, and training eligibility
is derived from the full fail-closed permission engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

from crawler.governance.training_permission import (
    resolve_training_permission,
)

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.fetching.results.result import FetchResult
    from crawler.governance.domains.domain_governance_registry import (
        DomainGovernanceRegistry,
    )
    from crawler.governance.processing_activity import (
        ProcessingActivityRegistry,
    )

Decision = Literal["allow", "deny"]
Check = Literal["pass", "fail"]


class _GovernanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceEvidence(_GovernanceModel):
    id: str
    name: str
    rules_version: str
    registry_version: str


class CollectionDecision(_GovernanceModel):
    allowed: bool
    reason: str


class RobotsAccess(_GovernanceModel):
    checked: bool
    decision: Literal["allow", "deny", "unavailable", "unreachable"]
    reason: str
    robots_url: str
    user_agent: str
    fetched_at: str
    cache_expires_at: str | None
    crawl_delay_seconds: float | None


class RightsEvidence(_GovernanceModel):
    """Explicit rights evidence used for training permission decisions."""

    checked: bool
    decision: Decision
    expression: str
    evidence_url: str
    evidence_kind: Literal["source_registry", "robots", "terms_page"]
    reason: str
    rights_reserved: bool
    tdm_allowed: bool
    commercial_use_allowed: bool
    attribution_required: bool
    review_expires_at: str | None
    rules_version: str

    @field_validator("review_expires_at")
    @classmethod
    def validate_review_expiry(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _parse_utc_timestamp(value)
        return value

    @model_validator(mode="after")
    def validate_evidence_consistency(self) -> "RightsEvidence":
        if not self.checked and self.decision == "allow":
            raise ValueError("unchecked rights evidence cannot allow training")
        if self.decision == "allow":
            if self.rights_reserved:
                raise ValueError("rights_reserved blocks training")
            if not self.tdm_allowed:
                raise ValueError("tdm_allowed=false blocks training")
            if not self.expression.strip() or not self.evidence_url.strip():
                raise ValueError(
                    "allowed rights require expression and evidence_url"
                )
        return self

    def is_current(self, *, now: datetime) -> bool:
        if self.review_expires_at is None:
            return True
        return _parse_utc_timestamp(self.review_expires_at) > now


class PrivacyCheck(_GovernanceModel):
    checked: bool
    result: Check
    action: Literal["none", "redacted", "rejected"]
    reason: str


class DedupeCheck(_GovernanceModel):
    checked: bool
    result: Check
    content_hash: str
    duplicate_of: str | None


class QualityCheck(_GovernanceModel):
    checked: bool
    result: Check
    reason: str


class FetchLineage(_GovernanceModel):
    complete: bool
    requested_url: str
    final_url: str
    origin: str
    fetched_at: str
    run_id: str


class TrainingEligibility(_GovernanceModel):
    allowed: bool
    reason: str
    rules_version: str


class RecordGovernance(_GovernanceModel):
    source: SourceEvidence
    collection: CollectionDecision
    access: RobotsAccess
    rights: RightsEvidence
    privacy: PrivacyCheck
    dedupe: DedupeCheck
    quality: QualityCheck
    lineage: FetchLineage
    training: TrainingEligibility

    @model_validator(mode="after")
    def validate_training_eligibility(self) -> "RecordGovernance":
        if not self.training.allowed:
            return self
        decision = resolve_training_permission(
            collection=self.collection,
            access=self.access,
            rights=self.rights,
            privacy=self.privacy,
            dedupe=self.dedupe,
            quality=self.quality,
            lineage=self.lineage,
            processing_activity_allowed=True,
            dpia_approved=True,
            now=_parse_utc_timestamp(self.access.fetched_at),
        )
        if not decision.allowed:
            raise ValueError(
                "training.allowed=true conflicts with governance evidence: "
                + ",".join(decision.violations)
            )
        return self


def create_record_governance(
    *,
    registry: DomainGovernanceRegistry | None,
    task: CrawlTask,
    result: FetchResult,
    content_hash: str,
    domain: str,
    asset_context: Mapping[str, object],
    enrichment: Mapping[str, object],
    run_id: str,
    now: datetime,
    processing_activity_registry: ProcessingActivityRegistry | None = None,
    processing_activity_id: str | None = None,
) -> RecordGovernance:
    """Build strict governance for one successful dataset record.

    All parts are required; ``training.allowed`` implies full green lights.
    Uses DomainGovernance (mapped from SourceRulesSettings) for source
    decisions. Processing-activity permission is resolved fail-closed.
    """
    rules = None
    if registry is not None:
        rules = registry.get(domain=domain)

    src_id = (
        getattr(rules, "source_id", None)
        or getattr(rules, "domain", None)
        or domain
    )
    src_name = getattr(rules, "source_name", None) or src_id
    pol_ver = getattr(rules, "rules_version", None) or "1"
    reg_ver = getattr(rules, "registry_version", None) or "1"

    source = SourceEvidence(
        id=str(src_id),
        name=str(src_name),
        rules_version=str(pol_ver),
        registry_version=str(reg_ver),
    )

    coll_allowed = bool(
        rules is not None and getattr(rules, "allow_collection", False)
    )
    coll_reason = (
        "rules_collection_allowed"
        if coll_allowed
        else getattr(rules, "governance_note", None) or "no_collection_rules"
    )

    collection_decision = CollectionDecision(
        allowed=coll_allowed,
        reason=coll_reason,
    )

    raw_robots_decision = str(
        asset_context.get("robots_status") or "unavailable"
    )
    robots_decision: Literal["allow", "deny", "unavailable", "unreachable"]
    if raw_robots_decision == "allow":
        robots_decision = "allow"
    elif raw_robots_decision == "deny":
        robots_decision = "deny"
    elif raw_robots_decision == "unreachable":
        robots_decision = "unreachable"
    else:
        robots_decision = "unavailable"
    access_model = RobotsAccess(
        checked=robots_decision != "unavailable",
        decision=robots_decision,
        reason="from_robots_or_context",
        robots_url=f"https://{domain}/robots.txt",
        user_agent=str(
            asset_context.get("user_agent") or "MultimodalCrawler/1.0"
        ),
        fetched_at=result.fetched_at,
        cache_expires_at=None,
        crawl_delay_seconds=None,
    )

    license_expr = (
        _text(asset_context.get("license"))
        or getattr(rules, "license", None)
        or "unknown"
    )
    license_evidence = (
        _text(asset_context.get("license_url"))
        or getattr(rules, "license_url", None)
        or ""
    )
    license_checked = bool(
        rules is not None
        or _text(asset_context.get("license"))
        or _text(asset_context.get("license_url"))
    )
    license_allowed = (
        license_checked
        and license_expr not in ("unknown", "", None)
        and _is_http_evidence_url(license_evidence)
    )
    raw_license_kind = str(
        asset_context.get("license_evidence_kind")
        or getattr(rules, "terms_source", None)
        or "source_registry"
    ).replace("-", "_")
    license_kind: Literal["source_registry", "robots", "terms_page"]
    if raw_license_kind == "robots":
        license_kind = "robots"
    elif raw_license_kind == "terms_page":
        license_kind = "terms_page"
    else:
        license_kind = "source_registry"

    rights = RightsEvidence(
        checked=license_checked,
        decision="allow" if license_allowed else "deny",
        expression=str(license_expr or ""),
        evidence_url=str(license_evidence or ""),
        evidence_kind=license_kind,
        reason=(
            "rights_evidence_verified"
            if license_allowed
            else "rights_evidence_missing_or_invalid"
        ),
        rights_reserved=bool(asset_context.get("rights_reserved", False)),
        tdm_allowed=bool(asset_context.get("tdm_allowed", license_allowed)),
        commercial_use_allowed=bool(
            asset_context.get("commercial_use_allowed", False)
        ),
        attribution_required=bool(
            asset_context.get("attribution_required", False)
        ),
        review_expires_at=_text(asset_context.get("rights_review_expires_at")),
        rules_version=str(pol_ver),
    )

    governance_checks = _gather_governance_checks(
        asset_context=asset_context,
        enrichment=enrichment,
    )
    privacy = _privacy_from_check(governance_checks.get("privacy"))
    dedupe = _dedupe_from_check(
        check=governance_checks.get("dedupe"),
        content_hash=content_hash,
    )
    quality = _quality_from_check(governance_checks.get("quality"))

    activity_allowed, dpia_approved = processing_activity_permission(
        registry=processing_activity_registry,
        activity_id=processing_activity_id,
        now=now,
    )
    permission_reference = datetime.fromisoformat(
        access_model.fetched_at.strip().replace("Z", "+00:00")
    )
    if permission_reference.tzinfo is None:
        raise ValueError("fetched_at must include timezone")

    lineage = FetchLineage(
        complete=True,
        requested_url=task.url,
        final_url=result.final_url,
        origin=domain,
        fetched_at=result.fetched_at,
        run_id=run_id,
    )

    permission = resolve_training_permission(
        collection=collection_decision,
        access=access_model,
        rights=rights,
        privacy=privacy,
        dedupe=dedupe,
        quality=quality,
        lineage=lineage,
        processing_activity_allowed=activity_allowed,
        dpia_approved=dpia_approved,
        now=permission_reference.astimezone(timezone.utc),
    )
    if rules is None:
        training_allowed = False
        training_reason = "no_training_rules"
    elif not getattr(rules, "allow_training", False):
        training_allowed = False
        training_reason = "rules_training_not_allowed"
    elif permission.allowed:
        training_allowed = True
        training_reason = getattr(rules, "governance_note", None) or "rules"
    else:
        training_allowed = False
        training_reason = permission.violations[0]

    training = TrainingEligibility(
        allowed=training_allowed,
        reason=training_reason,
        rules_version=str(pol_ver),
    )

    return RecordGovernance(
        source=source,
        collection=collection_decision,
        access=access_model,
        rights=rights,
        privacy=privacy,
        dedupe=dedupe,
        quality=quality,
        lineage=lineage,
        training=training,
    )


def _text(value: object) -> str | None:
    """Return stripped text or None for empty/missing values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_utc_timestamp(value: str) -> datetime:
    """Parse a timestamp and normalize it to UTC."""

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, AttributeError) as exc:
        raise ValueError("timestamp must be a valid ISO-8601 value") from exc


def _is_http_evidence_url(value: object) -> bool:
    text = _text(value)
    if text is None:
        return False
    try:
        parsed = urlparse(text)
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)


def processing_activity_permission(
    *,
    registry: ProcessingActivityRegistry | None,
    activity_id: str | None,
    now: datetime,
) -> tuple[bool, bool]:
    """Resolve processing-activity permission fail-closed from the registry."""
    if registry is None or not activity_id:
        return False, False
    try:
        activity = registry.require(activity_id=activity_id)
    except KeyError:
        return False, False
    return (
        activity.permits_training(now=now),
        activity.dpia_status == "approved",
    )


def _gather_governance_checks(
    *,
    asset_context: Mapping[str, object],
    enrichment: Mapping[str, object],
) -> dict[str, object]:
    checks: dict[str, object] = {}
    for value in (
        enrichment.get("governance_checks"),
        asset_context.get("governance_checks"),
    ):
        if isinstance(value, Mapping):
            checks.update({str(name): check for name, check in value.items()})
    return checks


def _check_values(
    check: object,
) -> tuple[Mapping[str, object], bool, bool, str]:
    payload = check if isinstance(check, Mapping) else {}
    checked = payload.get("checked") is True
    passed = checked and str(payload.get("result") or "").lower() == "pass"
    reason = _text(payload.get("reason")) or (
        "explicit_governance_check" if checked else "not_evaluated"
    )
    return payload, checked, passed, reason


def _privacy_from_check(check: object) -> PrivacyCheck:
    payload, checked, passed, reason = _check_values(check)
    raw_action = str(payload.get("action") or "").lower()
    action: Literal["none", "redacted", "rejected"]
    if raw_action == "none":
        action = "none"
    elif raw_action == "redacted":
        action = "redacted"
    elif raw_action == "rejected":
        action = "rejected"
    else:
        action = "none" if passed else "rejected"
    return PrivacyCheck(
        checked=checked,
        result="pass" if passed else "fail",
        action=action,
        reason=reason,
    )


def _dedupe_from_check(*, check: object, content_hash: str) -> DedupeCheck:
    payload, checked, passed, _ = _check_values(check)
    return DedupeCheck(
        checked=checked,
        result="pass" if passed else "fail",
        content_hash=content_hash,
        duplicate_of=_text(payload.get("duplicate_of")),
    )


def _quality_from_check(check: object) -> QualityCheck:
    _, checked, passed, reason = _check_values(check)
    return QualityCheck(
        checked=checked,
        result="pass" if passed else "fail",
        reason=reason,
    )
