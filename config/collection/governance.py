"""Public models for config.collection.governance.

Exports: StaticAssetFilterSettings, UrlFilterSettings,
    BlacklistManagerSettings, RobotsSettings.
"""

from __future__ import annotations

import ipaddress
from typing import Literal

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel
from config.collection.value_normalizers import normalize_string_tuple


class StaticAssetFilterSettings(SettingsModel):
    """Settings for low-value embedded static asset suppression."""

    enabled: bool = True
    apply_to_embedded_assets: bool = True
    blockable_static_asset_kinds: tuple[str, ...] = ("document", "other")
    trainable_media_kinds: tuple[str, ...] = ("image", "audio", "video")
    document_kinds: tuple[str, ...] = ("document", "other")
    blocked_extensions: tuple[str, ...] = (
        ".css",
        ".js",
        ".mjs",
        ".map",
        ".svg",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".ico",
    )

    @field_validator(
        "blockable_static_asset_kinds",
        "trainable_media_kinds",
        "document_kinds",
        mode="before",
    )
    @classmethod
    def normalize_document_kinds(cls, value: object) -> tuple[str, ...]:
        return normalize_string_tuple(value, lowercase=True)

    @field_validator("blocked_extensions", mode="before")
    @classmethod
    def normalize_blocked_extensions(cls, value: object) -> tuple[str, ...]:
        return normalize_string_tuple(
            value,
            lowercase=True,
            require_prefix=".",
        )

    @model_validator(mode="after")
    def validate_static_asset_filter(self) -> StaticAssetFilterSettings:
        blockable_kinds = (
            self.blockable_static_asset_kinds or self.document_kinds
        )
        if self.enabled and not blockable_kinds:
            raise ValueError(
                "blockable_static_asset_kinds must not be empty when enabled is true"
            )

        if self.enabled and not self.blocked_extensions:
            raise ValueError(
                "blocked_extensions must not be empty when enabled is true"
            )

        return self


class UrlFilterSettings(SettingsModel):
    """Settings for crawl scope and URL allow/deny rules."""

    restrict_to_seed_hosts: bool = True
    allow_subdomains_of_seed_hosts: bool = False
    blocked_hosts: tuple[str, ...] = ()
    blocked_ip_literals: tuple[str, ...] = ()
    max_pagination_page_number: int = Field(default=200, ge=1)
    pagination_query_keys: tuple[str, ...] = (
        "page",
        "paged",
        "p",
        "start",
    )
    intelligent_domain_expansion_enabled: bool = True
    max_expanded_hosts: int = Field(default=20, ge=0)
    expanded_host_min_quality: float = Field(default=0.55, ge=0.0, le=1.0)
    expanded_host_topic_keywords: tuple[str, ...] = (
        "python",
        "api",
        "docs",
        "research",
        "dataset",
        "machine-learning",
        "ai",
        "science",
    )
    static_assets: StaticAssetFilterSettings = Field(
        default_factory=StaticAssetFilterSettings,
    )

    # Rules data moved from UrlSyntaxRules (per governance cleanup)
    blocked_path_fragments: tuple[str, ...] = (
        "/social-media/",
        "/app/",
        "/apps/",
        "/event-type/",
        "/exit.html",
        "/exit/",
        "/a-to-z-topics-listing/",
        "/nasa-brand-center/",
    )
    blocked_query_keys: tuple[str, ...] = (
        "ical",
        "icalendar",
        "mc_cid",
        "mc_eid",
        "msclkid",
        "print",
        "printable",
        "replytocom",
        "share",
        "shared",
        "sharing",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
        "igshid",
    )
    blocked_query_value_patterns: dict[str, tuple[str, ...]] = Field(
        default_factory=lambda: {
            "feed": ("rss", "atom", "xml"),
            "format": ("rss", "atom", "xml", "print"),
            "output": ("rss", "atom", "xml", "print"),
            "view": ("calendar", "ical", "print"),
        }
    )
    tracking_query_tokens: tuple[str, ...] = (
        "analytics",
        "beacon",
        "collect",
        "impression",
        "pixel",
        "tracker",
        "tracking",
    )
    low_value_image_path_fragments: tuple[str, ...] = (
        "/icons/",
        "/icon/",
        "/social/",
        "/usa-icons/",
    )
    low_value_image_filenames: tuple[str, ...] = (
        "facebook.svg",
        "favicon.svg",
        "icon.svg",
        "instagram.svg",
        "linkedin.svg",
        "logo.jpg",
        "logo.jpeg",
        "logo.png",
        "logo.svg",
        "logo.webp",
        "pinterest.svg",
        "share.svg",
        "twitter.svg",
        "x.svg",
        "youtube.svg",
    )
    social_icon_tokens: tuple[str, ...] = (
        "facebook",
        "instagram",
        "linkedin",
        "pinterest",
        "share",
        "social",
        "twitter",
        "whatsapp",
    )

    @field_validator("expanded_host_topic_keywords", mode="before")
    @classmethod
    def normalize_expanded_host_topic_keywords(
        cls,
        value: object,
    ) -> tuple[str, ...]:
        return normalize_string_tuple(value, lowercase=True)

    @field_validator(
        "blocked_hosts",
        "blocked_ip_literals",
        "pagination_query_keys",
        "blocked_path_fragments",
        "blocked_query_keys",
        "tracking_query_tokens",
        "low_value_image_path_fragments",
        "low_value_image_filenames",
        "social_icon_tokens",
        mode="before",
    )
    @classmethod
    def normalize_blocked_hosts(
        cls,
        value: object,
    ) -> tuple[str, ...]:
        return normalize_string_tuple(value, lowercase=True)

    @model_validator(mode="after")
    def validate_domain_expansion(self) -> UrlFilterSettings:
        for ip_literal in self.blocked_ip_literals:
            try:
                ipaddress.ip_address(ip_literal.strip("[]"))
            except ValueError as exc:
                raise ValueError(
                    "blocked_ip_literals must contain valid IP addresses"
                ) from exc

        if self.intelligent_domain_expansion_enabled:
            if self.max_expanded_hosts < 1:
                raise ValueError(
                    "max_expanded_hosts must be at least 1 when "
                    "intelligent_domain_expansion_enabled is true"
                )

            if not self.expanded_host_topic_keywords:
                raise ValueError(
                    "expanded_host_topic_keywords must not be empty when "
                    "intelligent_domain_expansion_enabled is true"
                )

        return self

    @field_validator("blocked_query_value_patterns", mode="before")
    @classmethod
    def normalize_blocked_query_value_patterns(
        cls, value: object
    ) -> dict[str, tuple[str, ...]]:
        if value is None:
            return {}
        if isinstance(value, dict):
            result: dict[str, tuple[str, ...]] = {}
            for k, v in value.items():
                key = str(k).strip().lower()
                raw_vals = v if isinstance(v, (list, tuple, set)) else [v]
                vals = tuple(
                    str(x).strip().lower() for x in raw_vals if str(x).strip()
                )
                if key and vals:
                    result[key] = vals
            return result
        return {}


class BlacklistManagerSettings(SettingsModel):
    """Blacklist persistence configuration only."""

    blacklist_database_path: str = Field(
        default="runtime/state/blacklist/blacklist.sqlite3",
        min_length=1,
    )
    blacklist_table_name: str = Field(default="blacklist_urls", min_length=1)
    blacklist_auto_initialize: bool = True
    blacklist_busy_timeout_ms: int = Field(default=5_000, ge=0)


class RobotsSettings(SettingsModel):
    """Robots.txt governance applied at request time by the robots gate.

    - ``mode``: disabled (no checks), observe (record would-block metrics
      without changing behavior), or enforce (apply allow/block/defer).
    - 404/410 on robots.txt -> authoritative allow; a 2xx document that
      disallows the target -> block; unknown multimodal follow the
      ``on_*_unknown`` policies (weak/hostile default block, transient
      default defer).
    - crawl-delay is forwarded to pacing; decisions use identity.name UA.
    """

    mode: Literal["disabled", "observe", "enforce"] = "enforce"
    on_weak_unknown: Literal["allow", "block", "defer"] = "block"
    on_transient_unknown: Literal["allow", "block", "defer"] = "defer"
    on_hostile_unknown: Literal["allow", "block", "defer"] = "block"
