"""Composition-layer config validation rules.

These validations ensure that config required by the composition layer
is well-formed. They run after settings are loaded but before composition
begins, providing fail-fast feedback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings.root import Settings


def validate_composition_config(settings: Settings) -> None:
    """Validate config required by the composition layer.

    These checks ensure that configuration required by the composition
    layer is well-formed before any construction begins.
    """

    _validate_project_root(settings)
    _validate_host_profile_cache(settings)
    _validate_namespace_ttl(settings)
    _validate_state_settings(settings)


def _validate_project_root(settings: Settings) -> None:
    """Validate that project root is configured."""
    if settings.paths.root is None:
        raise ValueError(
            "paths.root must be configured; "
            "set datasets.paths.root or ensure config root resolves a workspace"
        )


def _validate_host_profile_cache(settings: Settings) -> None:
    """Validate host profile cache settings."""
    host_profile = settings.collection.cache.host_profile
    if not host_profile.enabled:
        raise ValueError("collection.cache.host_profile must be enabled")
    ttl_seconds = host_profile.ttl_seconds
    if ttl_seconds is None or ttl_seconds <= 0:
        raise ValueError(
            "collection.cache.host_profile.ttl_seconds must be positive"
        )


def _validate_namespace_ttl(settings: Settings) -> None:
    """Validate governance namespace TTL settings."""
    cache_settings = settings.collection.cache
    for namespace in ("blacklist_manager", "source_scope_registry"):
        ns_settings = getattr(cache_settings, namespace, None)
        if ns_settings is None:
            continue
        if not ns_settings.enabled:
            raise ValueError(f"collection.cache.{namespace} must be enabled")
        ttl_seconds = ns_settings.ttl_seconds
        if ttl_seconds is None or ttl_seconds <= 0:
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
