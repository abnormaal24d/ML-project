"""Architecture tests for CrawlerGovernance ownership invariants.

Enforces that CrawlerGovernance exports only governance-owned capabilities
and never re-exports infrastructure-owned services.
"""

from __future__ import annotations

import dataclasses

from orchestration.composition.runtime.crawler_governance import (
    CrawlerGovernance,
)
from orchestration.composition.runtime.crawler_infrastructure import (
    CrawlerInfrastructure,
)

# Required governance capabilities that MUST be present
REQUIRED_GOVERNANCE_FIELDS = frozenset(
    {
        "url_validator",
        "url_filter",
        "robots_request_gate",
        "redirector",
        "source_scope_registry",
        "blacklist_repository",
    }
)

# Infrastructure-owned capabilities that CrawlerGovernance MUST NEVER export
FORBIDDEN_INFRASTRUCTURE_FIELDS = frozenset(
    {
        "network_access_guard",
        "rate_limiter",
        "metrics",
        "clock",
        "host_normalizer",
        "host_extractor",
        "session_manager",
        "prometheus_exporter",
        "coverage_tracker",
        "host_budget_tracker",
        "host_media_byte_budget",
        "priority_resolver",
        "host_suppression_store",
        "conditional_representation_cache",
        # Removed duplicate alias from CrawlerInfrastructure;
        # forbidden here to prevent re-introduction as governance-owned.
        "host_suppression_reader",
    }
)


def test_crawler_governance_contains_required_capabilities() -> None:
    """CrawlerGovernance must expose all required governance-owned capabilities."""
    actual_fields = frozenset(
        f.name for f in dataclasses.fields(CrawlerGovernance)
    )
    missing = REQUIRED_GOVERNANCE_FIELDS - actual_fields
    assert not missing, (
        f"CrawlerGovernance missing required governance capabilities: {sorted(missing)}"
    )


def test_crawler_governance_forbids_infrastructure_fields() -> None:
    """CrawlerGovernance must not re-export any infrastructure-owned capability."""
    actual_fields = frozenset(
        f.name for f in dataclasses.fields(CrawlerGovernance)
    )
    forbidden_present = actual_fields & FORBIDDEN_INFRASTRUCTURE_FIELDS
    assert not forbidden_present, (
        f"CrawlerGovernance illegally re-exports infrastructure fields: {sorted(forbidden_present)}. "
        f"These must remain owned by CrawlerInfrastructure."
    )


def test_crawler_governance_no_overlap_with_infrastructure() -> None:
    """CrawlerGovernance and CrawlerInfrastructure must have disjoint field sets."""
    governance_fields = frozenset(
        f.name for f in dataclasses.fields(CrawlerGovernance)
    )
    infrastructure_fields = frozenset(
        f.name for f in dataclasses.fields(CrawlerInfrastructure)
    )
    overlap = governance_fields & infrastructure_fields
    assert not overlap, (
        f"CrawlerGovernance and CrawlerInfrastructure share fields: {sorted(overlap)}. "
        f"Ownership must be exclusive: governance owns governance, infrastructure owns infrastructure."
    )
