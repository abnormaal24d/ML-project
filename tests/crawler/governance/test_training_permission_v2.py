from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from crawler.governance.domains.domain_governance import DomainGovernance
from crawler.governance.processing_activity import ProcessingActivity
from crawler.governance.training_permission import (
    TrainingPermissionDecision,
    resolve_training_permission,
)
from crawler.storage.datasets.records.governance import (
    CollectionDecision,
    DedupeCheck,
    FetchLineage,
    PrivacyCheck,
    QualityCheck,
    RightsEvidence,
    RobotsAccess,
    create_record_governance,
)


def _evidence():
    return dict(
        collection=CollectionDecision(allowed=True, reason="policy"),
        access=RobotsAccess(
            checked=True,
            decision="allow",
            reason="robots",
            robots_url="https://example/robots.txt",
            user_agent="crawler",
            fetched_at="2026-07-23T00:00:00Z",
            cache_expires_at=None,
            crawl_delay_seconds=None,
        ),
        rights=RightsEvidence(
            checked=True,
            decision="allow",
            expression="CC-BY-4.0",
            evidence_url="https://example/rights",
            evidence_kind="terms_page",
            reason="verified",
            rights_reserved=False,
            tdm_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
            review_expires_at="2027-07-23T00:00:00Z",
            rules_version="2",
        ),
        privacy=PrivacyCheck(
            checked=True, result="pass", action="none", reason="clear"
        ),
        dedupe=DedupeCheck(
            checked=True,
            result="pass",
            content_hash="a" * 64,
            duplicate_of=None,
        ),
        quality=QualityCheck(checked=True, result="pass", reason="ok"),
        lineage=FetchLineage(
            complete=True,
            requested_url="https://example/a",
            final_url="https://example/a",
            origin="example",
            fetched_at="2026-07-23T00:00:00Z",
            run_id="r1",
        ),
    )


def test_permission_allows_only_complete_evidence():
    decision = resolve_training_permission(
        **_evidence(),
        processing_activity_allowed=True,
        dpia_approved=True,
        now=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )
    assert decision.allowed
    assert decision.violations == ()


def test_permission_fails_closed_for_missing_dpia():
    decision = resolve_training_permission(
        **_evidence(),
        processing_activity_allowed=True,
        dpia_approved=False,
        now=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )
    assert not decision.allowed
    assert "dpia_not_approved" in decision.violations


def test_rights_reserved_cannot_allow():
    payload = _evidence()["rights"].model_dump()
    payload["rights_reserved"] = True
    with pytest.raises(ValueError, match="rights_reserved"):
        RightsEvidence.model_validate(payload)


def test_rights_review_expiry_must_be_valid_iso_8601() -> None:
    payload = _evidence()["rights"].model_dump()
    payload["review_expires_at"] = "not-a-timestamp"

    with pytest.raises(ValueError, match="valid ISO-8601"):
        RightsEvidence.model_validate(payload)


def test_processing_activity_expiry_blocks_training():
    activity = ProcessingActivity(
        activity_id="build",
        purpose="training",
        personal_data_allowed=False,
        dpia_status="approved",
        dpia_review_expires_at="2026-07-22T00:00:00Z",
        retention_days=30,
        rules_version="1",
        enabled=True,
    )
    assert not activity.permits_training(
        now=datetime(2026, 7, 23, tzinfo=timezone.utc)
    )


def test_record_assembly_uses_canonical_training_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = DomainGovernance(
        domain="example.test",
        license="CC-BY-4.0",
        license_url="https://example.test/rights",
        allow_training=True,
        governance_note="source_rules",
        allow_collection=True,
    )
    observed: dict[str, object] = {}

    def deny_by_canonical_engine(
        **kwargs: object,
    ) -> TrainingPermissionDecision:
        observed.update(kwargs)
        return TrainingPermissionDecision(
            allowed=False,
            violations=("canonical_policy_denial",),
        )

    monkeypatch.setattr(
        "crawler.storage.datasets.records.governance.resolve_training_permission",
        deny_by_canonical_engine,
    )
    governance = create_record_governance(
        registry=SimpleNamespace(get=lambda *, domain: rules),
        task=SimpleNamespace(url="https://example.test/item"),
        result=SimpleNamespace(
            final_url="https://example.test/item",
            fetched_at="2026-07-23T00:00:00Z",
        ),
        content_hash="a" * 64,
        domain="example.test",
        asset_context={"robots_status": "allow"},
        enrichment={
            "governance_checks": {
                "privacy": {"checked": True, "result": "pass"},
                "dedupe": {"checked": True, "result": "pass"},
                "quality": {"checked": True, "result": "pass"},
            }
        },
        run_id="run-1",
        now=datetime(2026, 7, 23, tzinfo=timezone.utc),
        processing_activity_registry=None,
        processing_activity_id=None,
    )

    assert observed["processing_activity_allowed"] is False
    assert observed["dpia_approved"] is False
    assert not governance.training.allowed
    assert governance.training.reason == "canonical_policy_denial"


def test_record_assembly_fails_closed_without_processing_activity() -> None:
    rules = DomainGovernance(
        domain="example.test",
        license="CC-BY-4.0",
        license_url="https://example.test/rights",
        allow_training=True,
        governance_note="source_rules",
        allow_collection=True,
    )

    governance = create_record_governance(
        registry=SimpleNamespace(get=lambda *, domain: rules),
        task=SimpleNamespace(url="https://example.test/item"),
        result=SimpleNamespace(
            final_url="https://example.test/item",
            fetched_at="2026-07-23T00:00:00Z",
        ),
        content_hash="a" * 64,
        domain="example.test",
        asset_context={"robots_status": "allow", "tdm_allowed": True},
        enrichment={
            "governance_checks": {
                "privacy": {"checked": True, "result": "pass"},
                "dedupe": {"checked": True, "result": "pass"},
                "quality": {"checked": True, "result": "pass"},
            }
        },
        run_id="run-1",
        now=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    assert not governance.training.allowed
    assert governance.training.reason == "processing_activity_not_allowed"
