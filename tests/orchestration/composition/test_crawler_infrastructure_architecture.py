"""Architecture tests for CrawlerInfrastructure contract invariants.

Enforces that the infrastructure aggregate has no untyped service fields
and no duplicate capability aliases.
"""

from __future__ import annotations

import dataclasses

from orchestration.composition.runtime.crawler_infrastructure import (
    CrawlerInfrastructure,
)


def test_crawler_infrastructure_has_no_untyped_service_fields() -> None:
    """Every service capability in CrawlerInfrastructure must have a concrete type annotation."""
    untyped = []
    for field in dataclasses.fields(CrawlerInfrastructure):
        raw = field.type if isinstance(field.type, str) else str(field.type)
        if raw == "object":
            untyped.append(field.name)

    assert not untyped, (
        f"CrawlerInfrastructure has fields typed as 'object': {sorted(untyped)}. "
        f"All service capabilities must use concrete types."
    )


def test_crawler_infrastructure_has_no_duplicate_suppression_reader() -> None:
    """host_suppression_reader was a duplicate alias for host_suppression_store."""
    fields = {
        field.name
        for field in dataclasses.fields(CrawlerInfrastructure)
    }

    assert "host_suppression_store" in fields
    assert "host_suppression_reader" not in fields, (
        "host_suppression_reader was removed as a duplicate alias. "
        "Consumers should use infrastructure.host_suppression_store directly."
    )
