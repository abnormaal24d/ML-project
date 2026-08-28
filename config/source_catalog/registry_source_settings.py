"""Typed governance validation for source registry entries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel
from config.environment.source_selection import PRODUCTION_ENVIRONMENTS

SourceStatus = Literal[
    "approved",
    "approved_training",
    "approved_collect_only",
    "proposed",
    "paused",
    "revoked",
]
PiiRisk = Literal["low", "medium", "high"]

_PROD_ALLOWED_STATUSES = frozenset({"approved", "approved_training"})
_MAX_RETENTION_DAYS = 3650


@dataclass(frozen=True, slots=True)
class RegistryGovernanceValidationResult:
    """Outcome of validating one or more registry sources."""

    source_names: tuple[str, ...]
    environment: str


class SourceGovernanceEntry(SettingsModel):
    """Per-domain governance rules using one strict field schema."""

    model_config = {"extra": "forbid"}

    domain: str
    allow_collection: bool = True
    allow_training: bool = False
    license: str = ""
    license_evidence_url: str = ""
    license_evidence_kind: Literal["source_registry", "terms_page"] = (
        "source_registry"
    )
    allow_boilerplate_image_caption: bool = False
    terms_source: str = ""

    @model_validator(mode="after")
    def _validate_collection_and_training(self) -> SourceGovernanceEntry:
        if self.allow_training and not self.allow_collection:
            raise ValueError(
                "allow_training=true requires allow_collection=true"
            )
        return self


class RegistrySourceEntry(SettingsModel):
    """One approved source family in the registry."""

    description: str
    owner: str
    status: SourceStatus
    reviewed_at: str | None = None
    last_verified_at: str | None = None
    review_expires_at: str
    allowed_hosts: tuple[str, ...]
    seed_hosts: tuple[str, ...] = ()
    seed_urls: tuple[str, ...] = ()
    allowed_asset_hosts: tuple[str, ...] = ()
    allowed_redirect_hosts: tuple[str, ...] = ()
    allow_subdomains: bool = False
    default_training_allowed: bool = False
    training_allowed_domains: tuple[str, ...] = ()
    disallowed_patterns: tuple[str, ...] = ()
    pii_risk: PiiRisk
    retention_days: int = Field(gt=0, le=_MAX_RETENTION_DAYS)
    contact_or_rules_url: str
    robots_rules_expected: str | None = None
    copyright_basis: str | None = None
    feed_alternate_urls: dict[str, tuple[str, ...]] = Field(
        default_factory=dict
    )
    # P source catalog media: explicit entries so audio/video not solely from general page discovery
    explicit_media_seeds: tuple[str, ...] = ()
    media_feeds: tuple[
        str, ...
    ] = ()  # direct RSS/Atom media, podcast, video indexes
    media_api_endpoints: tuple[str, ...] = ()
    governance: tuple[SourceGovernanceEntry, ...] = ()

    @field_validator(
        "allowed_hosts",
        "seed_hosts",
        "seed_urls",
        "allowed_asset_hosts",
        "allowed_redirect_hosts",
        "training_allowed_domains",
        "disallowed_patterns",
        "explicit_media_seeds",
        "media_feeds",
        "media_api_endpoints",
        mode="before",
    )
    @classmethod
    def _normalize_string_lists(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("expected a list or tuple of strings")
        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return tuple(normalized)

    @field_validator("feed_alternate_urls", mode="before")
    @classmethod
    def _normalize_feed_alternates(
        cls,
        value: object,
    ) -> dict[str, tuple[str, ...]]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("feed_alternate_urls must be an object")
        normalized: dict[str, tuple[str, ...]] = {}
        for primary, alternates in value.items():
            if not isinstance(alternates, list):
                raise ValueError("feed_alternate_urls values must be lists")
            normalized[str(primary)] = tuple(str(item) for item in alternates)
        return normalized

    @field_validator("governance", mode="before")
    @classmethod
    def _normalize_governance(
        cls,
        value: object,
    ) -> tuple[SourceGovernanceEntry, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise ValueError("governance must be a list")
        return tuple(
            SourceGovernanceEntry.model_validate(item) for item in value
        )

    @field_validator("contact_or_rules_url", mode="after")
    @classmethod
    def _validate_rules_url(cls, value: str) -> str:
        parsed = urlparse(value.strip())
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.netloc
            or not parsed.hostname
        ):
            raise ValueError(
                "contact_or_rules_url must be an absolute http/https URL"
            )
        return value.strip()

    @model_validator(mode="after")
    def _validate_host_relationships(self) -> RegistrySourceEntry:
        allowed = {host.lower() for host in self.allowed_hosts}
        if not allowed:
            raise ValueError("allowed_hosts must not be empty")

        if self.default_training_allowed:
            raise ValueError("default_training_allowed must be false")

        for host in self.seed_hosts:
            if host.lower() not in allowed:
                raise ValueError(
                    f"seed_hosts entry {host!r} must be listed in allowed_hosts"
                )

        for host in self.allowed_asset_hosts:
            if host.lower() not in allowed:
                raise ValueError(
                    "allowed_asset_hosts entry "
                    f"{host!r} must be listed in allowed_hosts"
                )

        for host in self.allowed_redirect_hosts:
            if host.lower() not in allowed:
                raise ValueError(
                    "allowed_redirect_hosts entry "
                    f"{host!r} must be listed in allowed_hosts"
                )

        for url in self.seed_urls:
            hostname = urlparse(url).hostname
            if hostname is None or hostname.lower() not in allowed:
                raise ValueError(
                    f"seed_urls entry {url!r} must use a host from allowed_hosts"
                )

        # validate explicit media seeds use allowed hosts
        for url in (
            self.explicit_media_seeds
            + self.media_feeds
            + self.media_api_endpoints
        ):
            hostname = urlparse(url).hostname
            if hostname is None or hostname.lower() not in allowed:
                raise ValueError(
                    f"media entry {url!r} must use a host from allowed_hosts"
                )

        governance_domains = {
            entry.domain.lower() for entry in self.governance
        }
        for domain in self.training_allowed_domains:
            if domain.lower() not in governance_domains:
                raise ValueError(
                    "training_allowed_domains entry "
                    f"{domain!r} requires a matching governance.domain entry"
                )
            governance_entry = next(
                entry
                for entry in self.governance
                if entry.domain.lower() == domain.lower()
            )
            if not governance_entry.allow_training:
                raise ValueError(
                    "training_allowed_domains entry "
                    f"{domain!r} requires governance.allow_training=true"
                )

        return self


def _parse_review_date(value: str) -> date:
    text = value.strip()
    if not text:
        raise ValueError("review expiry date must not be empty")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(f"review expiry date must be ISO-8601: {value!r}")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"review expiry date must be ISO-8601: {value!r}"
        ) from exc


def validate_registry_source_entry(
    *,
    source_name: str,
    payload: dict[str, Any],
    environment: str,
    reference_date: date | None = None,
) -> RegistrySourceEntry:
    """Validate one registry source entry for the active environment."""

    entry = RegistrySourceEntry.model_validate(payload)
    if environment not in PRODUCTION_ENVIRONMENTS:
        return entry

    if entry.status not in _PROD_ALLOWED_STATUSES:
        raise ValueError(
            f"sources.{source_name}.status must be one of "
            f"{sorted(_PROD_ALLOWED_STATUSES)} for prod, got {entry.status!r}"
        )

    expiry = _parse_review_date(entry.review_expires_at)
    today = reference_date or datetime.now(tz=None).date()
    if expiry < today:
        raise ValueError(
            f"sources.{source_name}.review_expires_at={entry.review_expires_at!r} "
            "is expired for prod"
        )

    return entry


def validate_registry_sources(
    *,
    registry: dict[str, Any],
    source_names: tuple[str, ...],
    environment: str,
    reference_date: date | None = None,
) -> RegistryGovernanceValidationResult:
    """Validate selected registry sources for one environment."""

    definitions = registry.get("sources")
    if not isinstance(definitions, dict):
        raise ValueError("source registry sources must be an object")

    validated: list[str] = []
    for source_name in source_names:
        payload = definitions.get(source_name)
        if not isinstance(payload, dict):
            raise ValueError(f"unknown source registry entry: {source_name!r}")
        validate_registry_source_entry(
            source_name=source_name,
            payload=payload,
            environment=environment,
            reference_date=reference_date,
        )
        validated.append(source_name)

    return RegistryGovernanceValidationResult(
        source_names=tuple(validated),
        environment=environment,
    )
