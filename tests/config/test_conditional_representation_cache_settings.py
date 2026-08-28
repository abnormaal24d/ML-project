"""Configuration ownership for the conditional representation cache."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.collection.caching import CollectionCacheSettings
from config.collection.fetching import FetcherSettings


def test_conditional_representation_cache_is_owned_by_collection_cache() -> (
    None
):
    settings = CollectionCacheSettings()

    cache = settings.conditional_representation_cache
    assert cache.enabled is True
    assert cache.max_entries == 10_000
    assert cache.ttl_seconds == 3_600.0
    assert all("cache" not in field for field in FetcherSettings.model_fields)


@pytest.mark.parametrize("ttl_seconds", (None, 0.0))
def test_enabled_conditional_representation_cache_requires_positive_ttl(
    ttl_seconds: float | None,
) -> None:
    with pytest.raises(
        ValidationError,
        match="conditional_representation_cache.ttl_seconds",
    ):
        CollectionCacheSettings(
            conditional_representation_cache={
                "enabled": True,
                "max_entries": 1,
                "ttl_seconds": ttl_seconds,
            }
        )


def test_disabled_conditional_representation_cache_allows_no_ttl() -> None:
    settings = CollectionCacheSettings(
        conditional_representation_cache={
            "enabled": False,
            "max_entries": 1,
            "ttl_seconds": None,
        }
    )

    assert settings.conditional_representation_cache.ttl_seconds is None
