"""Governance decisions for one canonical source domain."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DomainGovernance:
    """Resolved governance metadata for a crawled domain."""

    domain: str
    license: str | None
    license_url: str | None
    allow_training: bool | None
    governance_note: str | None
    allow_collection: bool | None = None
    robots_status: str | None = None
    terms_source: str | None = None
    usage_rules: str | None = None
    allow_boilerplate_image_caption: bool = False
