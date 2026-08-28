"""Direct behavior tests for conditional representation cache states."""

from __future__ import annotations

from crawler.fetching.response.cache import ConditionalRepresentationCache


def test_disabled_cache_accepts_no_ttl_and_never_enriches_headers() -> None:
    cache = ConditionalRepresentationCache(
        enabled=False,
        max_entries=1,
        ttl_seconds=None,
        clock=lambda: 0.0,
    )
    headers = {"Accept": "image/*"}

    assert cache.get_representation("https://example.test/image") is None
    assert (
        cache.enrich_headers(
            url="https://example.test/image",
            base_headers=headers,
        )
        is headers
    )

    cache.invalidate("https://example.test/image")

    assert cache.size == 0
