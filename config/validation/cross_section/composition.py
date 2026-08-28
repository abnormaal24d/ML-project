"""Composition-layer config validation rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings.root import Settings


def validate_composition_config(settings: Settings) -> None:
    """Validate relational config required before composition begins."""

    _validate_host_profile_cache(settings)
    _validate_namespace_ttl(settings)
    _validate_state_settings(settings)


def _validate_host_profile_cache(settings: Settings) -> None:
    """Validate host profile cache settings."""

    host_profile = settings.collection.cache.host_profile
    if not host_profile.enabled:
        raise ValueError("collection.cache.host_profile must be enabled")
    if host_profile.ttl_seconds is None or host_profile.ttl_seconds <= 0:
        raise ValueError(
            "collection.cache.host_profile.ttl_seconds must be positive"
        )


def _validate_namespace_ttl(settings: Settings) -> None:
    """Validate governance namespace TTL settings."""

    cache_settings = settings.collection.cache
    namespaces = (
        ("blacklist_manager", cache_settings.blacklist_manager),
        ("source_scope_registry", cache_settings.source_scope_registry),
    )
    for namespace, namespace_settings in namespaces:
        if not namespace_settings.enabled:
            raise ValueError(f"collection.cache.{namespace} must be enabled")
        if (
            namespace_settings.ttl_seconds is None
            or namespace_settings.ttl_seconds <= 0
        ):
            raise ValueError(
                f"collection.cache.{namespace}.ttl_seconds must be positive"
            )


def _validate_state_settings(settings: Settings) -> None:
    """Validate crawler state settings when enabled."""

    state_settings = settings.crawler.state
    if state_settings.enabled:
        if state_settings.checkpoint_filename is None:
            raise ValueError(
                "crawler.state.checkpoint_filename must be configured when state is enabled"
            )
        if (
            state_settings.dead_letter_enabled
            and state_settings.dead_letter_filename is None
        ):
            raise ValueError(
                "crawler.state.dead_letter_filename must be configured when dead_letter is enabled"
            )
