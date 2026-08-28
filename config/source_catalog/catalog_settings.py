"""Source and governance settings for configured crawl targets."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel
from config.collection.value_normalizers import normalize_string_tuple


class RulesDecisionSettings(SettingsModel):
    allowed: bool = False
    reason: str = "rules"


class TrainingRulesSettings(SettingsModel):
    """Training decision for a source domain."""

    allowed: bool = False
    reason: str = "rules"


class RightsRulesSettings(SettingsModel):
    """License info for source."""

    expression: str
    evidence_url: str
    evidence_kind: Literal["source_registry", "terms_page"]
    reason: str


class SourceRulesSettings(SettingsModel):
    """Structured source rules replacing loose governance entries."""

    domain: str
    source_id: str
    source_name: str
    rules_version: str
    registry_version: str
    collection: RulesDecisionSettings
    license: RightsRulesSettings
    training: TrainingRulesSettings
    allow_boilerplate_image_caption: bool = False

    @field_validator("domain", mode="before")
    @classmethod
    def normalize_domain(cls, value: object) -> str:
        return str(value).strip().lower()

    @model_validator(mode="after")
    def validate_training_requires_collection(self) -> SourceRulesSettings:
        if self.training.allowed and not self.collection.allowed:
            raise ValueError(
                "training.allowed=true requires collection.allowed=true"
            )
        return self


class SourceSeedSettings(SettingsModel):
    """One crawl seed bound to its immutable registry source."""

    source_name: str
    url: str

    @field_validator("source_name", mode="before")
    @classmethod
    def normalize_source_name(cls, value: object) -> str:
        source_name = str(value).strip().lower()
        if not source_name:
            raise ValueError("source seed requires source_name")
        return source_name

    @field_validator("url", mode="before")
    @classmethod
    def normalize_url(cls, value: object) -> str:
        url = str(value).strip()
        if not url:
            raise ValueError("source seed requires url")
        return url

    @model_validator(mode="after")
    def validate_url(self) -> SourceSeedSettings:
        parsed = urlparse(self.url)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError(
                f"source seed URL {self.url!r} must use http or https"
            )
        if not parsed.hostname:
            raise ValueError(f"invalid source seed URL: {self.url!r}")
        return self


class SourceScopeSettings(SettingsModel):
    """Source-bound page, asset, and redirect host scope."""

    source_name: str
    page_hosts: tuple[str, ...] = ()
    asset_hosts: tuple[str, ...] = ()
    redirect_hosts: tuple[str, ...] = ()
    allow_subdomains: bool = False

    @field_validator("source_name", mode="before")
    @classmethod
    def normalize_source_name(cls, value: object) -> str:
        source_name = str(value).strip().lower()
        if not source_name:
            raise ValueError("source scope requires source_name")
        return source_name

    @field_validator(
        "page_hosts",
        "asset_hosts",
        "redirect_hosts",
        mode="before",
    )
    @classmethod
    def normalize_hosts(cls, value: object) -> tuple[str, ...]:
        return normalize_string_tuple(value, lowercase=True)

    @model_validator(mode="after")
    def validate_hosts(self) -> SourceScopeSettings:
        if not self.page_hosts and not self.asset_hosts:
            raise ValueError(
                f"source scope {self.source_name!r} has no approved hosts"
            )
        return self


class SourceProfileSettings(SettingsModel):
    """Crawl source profile containing seeds, hosts, and governance rules."""

    require_seed_urls: bool = True
    seed_urls: tuple[str, ...] = ()
    seed_entries: tuple[SourceSeedSettings, ...] = ()
    source_scopes: tuple[SourceScopeSettings, ...] = ()
    seed_hosts: tuple[str, ...] = ()
    allowed_asset_hosts: tuple[str, ...] = ()
    training_allowed_domains: tuple[str, ...] = ()
    feed_alternate_urls: dict[str, tuple[str, ...]] = Field(
        default_factory=dict,
    )
    # Explicit media entries for direct audio/video (not only via page discovery)
    explicit_media_seeds: tuple[str, ...] = ()
    media_feeds: tuple[str, ...] = ()
    media_api_endpoints: tuple[str, ...] = ()
    governance: tuple[SourceRulesSettings, ...] = ()

    @field_validator("seed_urls", mode="before")
    @classmethod
    def normalize_seed_urls(cls, value: object) -> tuple[str, ...]:
        """Normalize configured seed URLs while preserving case."""

        return normalize_string_tuple(value, lowercase=False)

    @field_validator(
        "seed_hosts",
        "allowed_asset_hosts",
        "training_allowed_domains",
        mode="before",
    )
    @classmethod
    def normalize_domains(cls, value: object) -> tuple[str, ...]:
        """Normalize configured host/domain lists to lowercase strings."""

        return normalize_string_tuple(value, lowercase=True)

    @field_validator(
        "explicit_media_seeds",
        "media_feeds",
        "media_api_endpoints",
        mode="before",
    )
    @classmethod
    def normalize_direct_media_urls(cls, value: object) -> tuple[str, ...]:
        return normalize_string_tuple(value, lowercase=False)

    @model_validator(mode="after")
    def validate_profile(self) -> SourceProfileSettings:
        """
        Validate source profile seed, asset-host, and governance coverage.
        """

        if self.require_seed_urls and not self.seed_urls:
            raise ValueError("sources profile seed_urls must not be empty")
        if self.require_seed_urls and not self.seed_entries:
            raise ValueError("sources profile seed_entries must not be empty")

        entry_urls = tuple(entry.url for entry in self.seed_entries)
        if entry_urls != self.seed_urls:
            raise ValueError(
                "sources profile seed_entries must map one-to-one, in order, "
                "to seed_urls"
            )

        scope_by_name: dict[str, SourceScopeSettings] = {}
        for scope in self.source_scopes:
            if scope.source_name in scope_by_name:
                raise ValueError(
                    f"duplicate source scope: {scope.source_name!r}"
                )
            scope_by_name[scope.source_name] = scope

        seed_source_names = {entry.source_name for entry in self.seed_entries}
        missing_source_scopes = seed_source_names - set(scope_by_name)
        if missing_source_scopes:
            raise ValueError(
                "seed entries reference sources without source scopes: "
                f"{sorted(missing_source_scopes)}"
            )

        for entry in self.seed_entries:
            host = (urlparse(entry.url).hostname or "").lower()
            scope = scope_by_name[entry.source_name]
            approved_hosts = set(scope.page_hosts) | set(scope.asset_hosts)
            if host not in approved_hosts:
                raise ValueError(
                    f"seed URL host {host!r} is outside source scope "
                    f"{entry.source_name!r}"
                )

        seed_hosts = set(self.seed_hosts)
        for seed_url in self.seed_urls:
            parsed_seed_url = urlparse(seed_url)
            scheme = parsed_seed_url.scheme.lower()
            if scheme not in {"http", "https"}:
                raise ValueError(
                    f"seed URL {seed_url!r} must use http or https"
                )
            host = (parsed_seed_url.hostname or "").lower()
            if not host:
                raise ValueError(f"invalid seed URL: {seed_url!r}")
            if host not in seed_hosts:
                raise ValueError(
                    f"seed URL host {host!r} is not listed in seed_hosts"
                )

        governance_by_domain = {
            entry.domain: entry for entry in self.governance
        }
        training_domains = set(self.training_allowed_domains)
        missing_governance = training_domains - set(governance_by_domain)
        if missing_governance:
            raise ValueError(
                "training_allowed_domains missing governance entries: "
                f"{sorted(missing_governance)}"
            )

        governed_training_domains = {
            domain
            for domain, entry in governance_by_domain.items()
            if entry.training.allowed
        }
        missing_training_domain = governed_training_domains - training_domains
        if missing_training_domain:
            raise ValueError(
                "training governance entries missing from "
                "training_allowed_domains: "
                f"{sorted(missing_training_domain)}"
            )

        asset_hosts = set(self.allowed_asset_hosts)
        missing_asset_governance = asset_hosts - set(governance_by_domain)
        if missing_asset_governance:
            raise ValueError(
                "allowed_asset_hosts missing governance entries: "
                f"{sorted(missing_asset_governance)}"
            )
        permitted_hosts = seed_hosts | asset_hosts
        for media_url in (
            *self.explicit_media_seeds,
            *self.media_feeds,
            *self.media_api_endpoints,
        ):
            parsed_media_url = urlparse(media_url)
            host = (parsed_media_url.hostname or "").lower()
            if (
                parsed_media_url.scheme.lower() not in {"http", "https"}
                or not host
            ):
                raise ValueError(f"invalid direct media URL: {media_url!r}")
            if host not in permitted_hosts:
                raise ValueError(
                    "direct media URL host is neither a seed nor an allowed "
                    f"asset host: {host!r}"
                )
        return self


class SourceCatalogSettings(SettingsModel):
    """Single active source definition for the selected environment."""

    active_profile: str = "public_science"
    selected_registry_sources: tuple[str | dict[str, object], ...] = ()
    registry_sources_provenance: dict[str, object] = Field(
        default_factory=dict
    )
    retention_days_by_domain: dict[str, int] = Field(default_factory=dict)
    require_seed_urls: bool = True
    seed_urls: tuple[str, ...] = ()
    seed_entries: tuple[SourceSeedSettings, ...] = ()
    source_scopes: tuple[SourceScopeSettings, ...] = ()
    seed_hosts: tuple[str, ...] = ()
    allowed_asset_hosts: tuple[str, ...] = ()
    training_allowed_domains: tuple[str, ...] = ()
    feed_alternate_urls: dict[str, tuple[str, ...]] = Field(
        default_factory=dict,
    )
    # Explicit media for direct audio/video support (P source catalog)
    explicit_media_seeds: tuple[str, ...] = ()
    media_feeds: tuple[str, ...] = ()
    media_api_endpoints: tuple[str, ...] = ()
    governance: tuple[SourceRulesSettings, ...] = ()

    @field_validator("selected_registry_sources", mode="before")
    @classmethod
    def normalize_selected_registry_sources(
        cls,
        value: object,
    ) -> tuple[str | dict[str, object], ...]:
        if value is None:
            return ()
        if isinstance(value, tuple):
            return value
        if not isinstance(value, list):
            raise ValueError("selected_registry_sources must be a list")
        normalized: list[str | dict[str, object]] = []
        for entry in value:
            if isinstance(entry, str):
                text = entry.strip()
                if text:
                    normalized.append(text)
                continue
            if isinstance(entry, dict):
                normalized.append(entry)
                continue
            raise ValueError(
                "selected_registry_sources entries must be strings or objects"
            )
        return tuple(normalized)

    @field_validator("seed_urls", mode="before")
    @classmethod
    def normalize_seed_urls(cls, value: object) -> tuple[str, ...]:
        """Normalize configured seed URLs while preserving case."""

        return normalize_string_tuple(value, lowercase=False)

    @field_validator("retention_days_by_domain", mode="before")
    @classmethod
    def normalize_retention_days(
        cls,
        value: object,
    ) -> dict[str, int]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("retention_days_by_domain must be an object")
        normalized: dict[str, int] = {}
        for raw_domain, raw_days in value.items():
            domain = str(raw_domain).strip().lower()
            if not domain:
                raise ValueError("retention domain must not be empty")
            if isinstance(raw_days, bool) or not isinstance(raw_days, int):
                raise ValueError("retention days must be integers")
            if not 1 <= raw_days <= 3650:
                raise ValueError("retention days must be between 1 and 3650")
            normalized[domain] = raw_days
        return normalized

    @field_validator(
        "seed_hosts",
        "allowed_asset_hosts",
        "training_allowed_domains",
        mode="before",
    )
    @classmethod
    def normalize_domains(cls, value: object) -> tuple[str, ...]:
        """Normalize configured host/domain lists to lowercase strings."""

        return normalize_string_tuple(value, lowercase=True)

    @field_validator(
        "explicit_media_seeds",
        "media_feeds",
        "media_api_endpoints",
        mode="before",
    )
    @classmethod
    def normalize_media_urls(cls, value: object) -> tuple[str, ...]:
        """Keep direct media URLs case-preserving and deduplicated."""

        return normalize_string_tuple(value, lowercase=False)

    @model_validator(mode="after")
    def validate_active_source(self) -> SourceCatalogSettings:
        """Validate the active source definition for this environment."""

        _ = self.active
        return self

    @property
    def active(self) -> SourceProfileSettings:
        """Return the active source profile view used by runtime services."""

        return SourceProfileSettings(
            require_seed_urls=self.require_seed_urls,
            seed_urls=self.seed_urls,
            seed_entries=self.seed_entries,
            source_scopes=self.source_scopes,
            seed_hosts=self.seed_hosts,
            allowed_asset_hosts=self.allowed_asset_hosts,
            training_allowed_domains=self.training_allowed_domains,
            feed_alternate_urls=self.feed_alternate_urls,
            explicit_media_seeds=self.explicit_media_seeds,
            media_feeds=self.media_feeds,
            media_api_endpoints=self.media_api_endpoints,
            governance=self.governance,
        )
