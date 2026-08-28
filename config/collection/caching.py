"""Public models and helpers for config.collection.caching.

Exports: CacheNamespaceSettings, CollectionCacheSettings,
    MetricsSettings.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from config.base.settings_model import SettingsModel


class CacheNamespaceSettings(SettingsModel):
    """Limits and TTL rules for one named in-process cache."""

    enabled: bool = True
    max_entries: int = Field(default=1_024, ge=1)
    ttl_seconds: float | None = Field(default=None, ge=0.0)
    stale_ttl_seconds: float | None = Field(default=None, ge=0.0)


class CollectionCacheSettings(SettingsModel):
    """Settings for all centrally managed collection caches."""

    host_profile: CacheNamespaceSettings = Field(
        default_factory=lambda: CacheNamespaceSettings(
            enabled=True,
            max_entries=10_000,
            ttl_seconds=3_600.0,
        ),
    )

    conditional_representation_cache: CacheNamespaceSettings = Field(
        default_factory=lambda: CacheNamespaceSettings(
            enabled=True,
            max_entries=10_000,
            ttl_seconds=3_600.0,
        ),
    )

    mime_type_normalization: CacheNamespaceSettings = Field(
        default_factory=lambda: CacheNamespaceSettings(
            enabled=True,
            max_entries=512,
            ttl_seconds=None,
        ),
    )

    url_normalization: CacheNamespaceSettings = Field(
        default_factory=lambda: CacheNamespaceSettings(
            enabled=True,
            max_entries=10_000,
            ttl_seconds=None,
        ),
    )

    robots_parser: CacheNamespaceSettings = Field(
        default_factory=lambda: CacheNamespaceSettings(
            enabled=True,
            max_entries=10_000,
            ttl_seconds=3_600.0,
            stale_ttl_seconds=86_400.0,
        ),
    )

    robots_error: CacheNamespaceSettings = Field(
        default_factory=lambda: CacheNamespaceSettings(
            enabled=True,
            max_entries=10_000,
            ttl_seconds=300.0,
        ),
    )

    scheduler_seen_url: CacheNamespaceSettings = Field(
        default_factory=lambda: CacheNamespaceSettings(
            enabled=False,
            max_entries=250_000,
            ttl_seconds=None,
        ),
    )

    robots_host_rules_advice: CacheNamespaceSettings = Field(
        default_factory=lambda: CacheNamespaceSettings(
            enabled=False,
            max_entries=100_000,
            ttl_seconds=None,
        ),
    )

    @model_validator(mode="after")
    def validate_conditional_representation_cache(
        self,
    ) -> CollectionCacheSettings:
        """Require expiring entries when the representation cache is enabled."""

        cache = self.conditional_representation_cache
        if cache.enabled and (
            cache.ttl_seconds is None or cache.ttl_seconds <= 0.0
        ):
            raise ValueError(
                "conditional_representation_cache.ttl_seconds "
                "must be positive when enabled"
            )

        return self


class MetricsSettings(SettingsModel):
    """Settings for in-process crawl metrics aggregation and export."""

    enabled: bool = True
    prometheus_enabled: bool = False
    prometheus_port: int = Field(default=9765, ge=1, le=65_535)
    emit_snapshot_log_interval_seconds: float = Field(default=30.0, gt=0.0)

    @model_validator(mode="after")
    def validate_metrics_export(self) -> MetricsSettings:
        """Ensure Prometheus export is enabled only with metrics enabled."""

        if self.prometheus_enabled and not self.enabled:
            raise ValueError(
                "metrics.enabled must be true when prometheus_enabled is true"
            )

        return self
