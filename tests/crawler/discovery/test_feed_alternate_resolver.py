"""Canonical feed-alternate identity and seed expansion invariants."""

from __future__ import annotations

import pytest

from config.collection.extraction import UrlNormalizerSettings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.discovery.feed_alternate_resolver import (
    FeedAlternateResolver,
    expand_seed_tasks_with_feed_alternates,
)
from crawler.extraction.urls.normalizer import UrlNormalizer
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.source_scope.source_scope_registry import (
    SourceScope,
    SourceScopeRegistry,
)
from tests.support.logging import TEST_LOGGER


class _SequentialIdGenerator:
    def __init__(self) -> None:
        self._next_id = 1

    def generate(self) -> str:
        generated = f"alternate-{self._next_id}"
        self._next_id += 1
        return generated


def _normalizer(**settings: object) -> UrlNormalizer:
    return UrlNormalizer(
        settings=UrlNormalizerSettings.model_validate(settings),
        logger=TEST_LOGGER,
        host_normalizer=HostNormalizer(),
    )


def _scope_registry(*source_names: str) -> SourceScopeRegistry:
    return SourceScopeRegistry(
        tuple(
            SourceScope(
                source_name=source_name,
                page_hosts=frozenset({"example.com"}),
                asset_hosts=frozenset(),
                redirect_hosts=frozenset(),
            )
            for source_name in source_names
        )
    )


def test_fragment_never_participates_in_feed_identity() -> None:
    normalizer = _normalizer(enabled=False, strip_fragments=False)
    resolver = FeedAlternateResolver(
        alternates_by_primary={
            "HTTPS://EXAMPLE.COM/feed#configured": (
                "https://EXAMPLE.com/alternate/./rss#stored",
            )
        },
        url_normalizer=normalizer,
        host_normalizer=HostNormalizer(),
    )

    assert resolver.alternates_for(
        url="https://example.com/feed#top",
        status_code=503,
    ) == ("https://example.com/alternate/rss",)
    assert (
        resolver.alternate_for(
            url="https://example.com/feed#another",
            error_type="transport_timeout",
        )
        == "https://example.com/alternate/rss"
    )
    assert (
        resolver.alternates_for(
            url="https://example.com/feed#top",
            status_code=404,
        )
        == ()
    )
    assert (
        resolver.alternates_for(
            url="../feed#top",
            status_code=503,
        )
        == ()
    )


def test_canonical_primary_collisions_merge_alternates_stably() -> None:
    resolver = FeedAlternateResolver(
        alternates_by_primary={
            "HTTPS://EXAMPLE.COM/feed#one": (
                "https://EXAMPLE.com/a#first",
                "https://example.com/shared#first",
                "https://example.com/feed#self",
            ),
            "https://example.com/feed#two": (
                "https://example.com/shared#second",
                "https://example.com/b#second",
            ),
        },
        url_normalizer=_normalizer(enabled=False),
        host_normalizer=HostNormalizer(),
    )

    assert resolver.alternates_for(
        url="https://example.com/feed#lookup",
        status_code=504,
    ) == (
        "https://example.com/a",
        "https://example.com/shared",
        "https://example.com/b",
    )


def test_signed_feed_queries_remain_byte_exact() -> None:
    primary_query = "X-Amz-Signature=AbC%2fD&Token=A+B%2fc&z=2&a=1"
    alternate_query = "X-Amz-Signature=EfG%2fH&Token=C+D%2fe&b=2&a=1"
    resolver = FeedAlternateResolver(
        alternates_by_primary={
            f"HTTPS://EXAMPLE.COM/feed?{primary_query}#configured": (
                f"https://EXAMPLE.com/rss?{alternate_query}#stored",
            )
        },
        url_normalizer=_normalizer(),
        host_normalizer=HostNormalizer(),
    )

    assert resolver.alternates_for(
        url=f"https://example.com/feed?{primary_query}#lookup",
        status_code=503,
    ) == (f"https://example.com/rss?{alternate_query}",)


def test_seed_expansion_publishes_only_canonical_feed_urls() -> None:
    normalizer = _normalizer()
    original = CrawlTask(
        url="HTTPS://EXAMPLE.COM:443/a/../feed/#top",
        source_name="source-a",
        task_id="primary-id",
        kind=MediaKind.PAGE,
        depth=2,
        source_type="registry",
        priority=7,
    )
    duplicate_alternate = CrawlTask(
        url=("https://example.com/rss/main?b=2&a=1&utm_source=noise#seed"),
        source_name="source-a",
        task_id="duplicate-id",
    )

    expanded = expand_seed_tasks_with_feed_alternates(
        seed_tasks=(original, duplicate_alternate),
        alternates_by_primary={
            "https://example.com/feed#configured": (
                "HTTPS://EXAMPLE.COM:443/rss/./main"
                "?utm_source=noise&b=2&a=1#alternate",
                "https://example.com/feed#self",
            )
        },
        url_normalizer=normalizer,
        host_normalizer=HostNormalizer(),
        source_scope_registry=_scope_registry("source-a"),
        id_generator=_SequentialIdGenerator(),
    )

    assert [task.url for task in expanded] == [
        "https://example.com/feed",
        "https://example.com/rss/main?a=1&b=2",
    ]
    primary, alternate = expanded
    assert primary.task_id == "primary-id"
    assert primary.kind is MediaKind.FEED
    assert alternate.task_id == "alternate-1"
    assert alternate.kind is MediaKind.FEED
    assert alternate.parent_url == primary.url
    assert alternate.depth == primary.depth
    assert alternate.source_type == primary.source_type
    assert alternate.priority == primary.priority
    assert alternate.context is not None
    assert alternate.context.source_page_url == primary.url
    assert alternate.context.discovery_reason == "feed_alternate_seed"


def test_same_canonical_seed_url_is_retained_once_per_source() -> None:
    seeds = tuple(
        CrawlTask(
            url=f"https://EXAMPLE.com/feed#{source_name}",
            source_name=source_name,
            task_id=f"{source_name}-id",
        )
        for source_name in ("source-a", "source-b")
    )

    expanded = expand_seed_tasks_with_feed_alternates(
        seed_tasks=seeds,
        alternates_by_primary={},
        url_normalizer=_normalizer(enabled=False),
        host_normalizer=HostNormalizer(),
        source_scope_registry=_scope_registry("source-a", "source-b"),
        id_generator=_SequentialIdGenerator(),
    )

    assert [(task.source_name, task.url) for task in expanded] == [
        ("source-a", "https://example.com/feed"),
        ("source-b", "https://example.com/feed"),
    ]


@pytest.mark.parametrize(
    ("alternates", "message"),
    [
        ({"../feed": ("https://example.com/rss",)}, "primary"),
        (
            {"https://example.com/feed": ("mailto:feed@example.com",)},
            "alternate",
        ),
        ({"https://example.com/feed": ("",)}, "alternate"),
    ],
)
def test_invalid_configured_feed_urls_fail_closed(
    alternates: dict[str, tuple[str, ...]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FeedAlternateResolver(
            alternates_by_primary=alternates,
            url_normalizer=_normalizer(),
            host_normalizer=HostNormalizer(),
        )


def test_alternate_outside_source_scope_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the configured source"):
        expand_seed_tasks_with_feed_alternates(
            seed_tasks=(
                CrawlTask(
                    url="https://example.com/feed#seed",
                    source_name="source-a",
                    task_id="primary-id",
                ),
            ),
            alternates_by_primary={
                "https://example.com/feed": (
                    "https://outside.example/rss#alternate",
                )
            },
            url_normalizer=_normalizer(enabled=False),
            host_normalizer=HostNormalizer(),
            source_scope_registry=_scope_registry("source-a"),
            id_generator=_SequentialIdGenerator(),
        )
