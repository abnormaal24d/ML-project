"""Regression coverage for fetch construction and request governance."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.settings.root import Settings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.fetching.errors.exceptions import IgnoredFetchError
from crawler.fetching.request.context import FetchRequestContext
from crawler.fetching.request.context_builder import FetchRequestContextBuilder
from crawler.fetching.response.cache import ConditionalRepresentationCache
from crawler.fetching.response.snapshot import FetchResponseSnapshot
from crawler.fetching.response.validator import FetchResponseValidator
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.host_suppression import HostSuppressionStore
from crawler.governance.redirect.redirect_rules_validator import (
    RedirectRulesValidator,
)
from orchestration.composition.runtime.fetch import build_fetcher


class _Logger:
    def debug(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def info(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def error(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _LoggerFactory:
    def get_logger_for(self, *args: object, **kwargs: object) -> _Logger:
        del args, kwargs
        return _Logger()


class _Clock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class _RecordingRedirector:
    def __init__(self) -> None:
        self.hops: list[dict[str, object]] = []

    def validate_hop(self, **kwargs: object) -> None:
        self.hops.append(kwargs)


class _UrlValidator:
    def is_allowed(self, url: str) -> bool:
        return bool(url)


class _HostExtractor:
    def __init__(self) -> None:
        self.called = False

    def extract(self, url: str) -> str:
        self.called = True
        return url.split("://", maxsplit=1)[-1].split("/", maxsplit=1)[0]


class _InitialUrlGuard:
    def __init__(self, reason: str | None) -> None:
        self.reason = reason
        self.urls: list[str] = []

    def rejection_reason_for_url(self, url: str) -> str | None:
        self.urls.append(url)
        return self.reason


def _context() -> FetchRequestContext:
    return FetchRequestContext(
        url="https://origin.example.test/image.jpg",
        host="origin.example.test",
        source_name="source",
        requested_kind=MediaKind.IMAGE,
        acceptance_mode="strict",
        acceptance=SimpleNamespace(
            allows_content_type=lambda _content_type: True,
            max_bytes_for_content_type=lambda _content_type: 1_000_000,
        ),
    )


@pytest.mark.parametrize(
    "snapshot",
    (
        FetchResponseSnapshot(
            status=200,
            url="https://cdn.example.test/image.jpg",
            headers={},
        ),
        FetchResponseSnapshot(
            status=200,
            url="https://cdn.example.test/image.jpg",
            headers={},
            redirect_chain=("https://cdn.example.test/image.jpg",),
        ),
    ),
)
def test_redirect_is_validated_for_all_redirect_shapes(
    snapshot: FetchResponseSnapshot,
) -> None:
    redirector = _RecordingRedirector()
    validator = FetchResponseValidator(
        redirector=redirector,  # type: ignore[arg-type]
        logger=_Logger(),  # type: ignore[arg-type]
    )

    validator.validate(context=_context(), response=snapshot)

    assert redirector.hops


def test_network_guard_and_blacklist_always_block() -> None:
    class _Guard:
        def rejection_reason_for_url(self, target_url: str) -> str | None:
            return "private_ip_blocked" if "127.0.0.1" in target_url else None

    class _Blacklist:
        def __init__(self, blocked: bool) -> None:
            self._blocked = blocked

        def contains(self, *, url: str) -> bool:
            del url
            return self._blocked

    class _HostExtractorForRedirect:
        def extract(self, url: str) -> str:
            return url.split("://", maxsplit=1)[-1].split("/", maxsplit=1)[0]

    settings = SimpleNamespace(
        max_redirects=3,
        cross_host_mode="deny",
        block_https_downgrade=False,
    )
    common = {
        "settings": settings,
        "host_extractor": _HostExtractorForRedirect(),
        "url_validator": _UrlValidator(),
        "host_normalizer": HostNormalizer(),
        "logger": _Logger(),
    }
    guarded = RedirectRulesValidator(
        **common,
        blacklist_repository=_Blacklist(blocked=False),  # type: ignore[arg-type]
        network_access_guard=_Guard(),  # type: ignore[arg-type]
    )
    with pytest.raises(IgnoredFetchError, match="redirect_private_ip_blocked"):
        guarded.validate_hop(
            current_url="https://origin.example.test/media",
            target_url="https://127.0.0.1/private",
            redirect_count=1,
            source_name="source",
        )

    blacklisted = RedirectRulesValidator(
        **common,
        blacklist_repository=_Blacklist(blocked=True),  # type: ignore[arg-type]
        network_access_guard=_Guard(),  # type: ignore[arg-type]
    )
    with pytest.raises(IgnoredFetchError, match="redirect_blacklisted"):
        blacklisted.validate_hop(
            current_url="https://origin.example.test/media",
            target_url="https://cdn.example.test/media",
            redirect_count=1,
            source_name="source",
        )


def test_context_builder_blocks_initial_network_target_before_host_lookup() -> (
    None
):
    extractor = _HostExtractor()
    guard = _InitialUrlGuard("url_credentials_blocked")
    builder = FetchRequestContextBuilder(
        url_validator=_UrlValidator(),  # type: ignore[arg-type]
        host_extractor=extractor,  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        acceptance_resolver=object(),  # type: ignore[arg-type]
        logger=_Logger(),  # type: ignore[arg-type]
        network_access_guard=guard,  # type: ignore[arg-type]
    )
    task = CrawlTask(
        url="https://user:pass@example.test/private",
        source_name="source",
    )

    with pytest.raises(
        IgnoredFetchError,
        match="blocked_network_target:url_credentials_blocked",
    ):
        builder.build(task=task)

    assert guard.urls == [task.url]
    assert extractor.called is False


def test_build_fetcher_smoke_constructs_the_composed_graph(
    tmp_path: Path,
) -> None:
    settings = Settings(profile="test")
    logger_factory = _LoggerFactory()
    host_normalizer = HostNormalizer()
    clock = _Clock()
    conditional_cache_settings = (
        settings.collection.cache.conditional_representation_cache
    )
    host_suppression = HostSuppressionStore(
        ttl_seconds=settings.collection.cache.host_profile.ttl_seconds,
        max_size=settings.collection.cache.host_profile.max_entries,
        suppress_after_forbidden_responses=(
            settings.collection.fetcher.host_profile_forbidden_host_threshold
        ),
        forbidden_host_cooldown_seconds=(
            settings.collection.fetcher.host_profile_forbidden_host_cooldown_seconds
        ),
        host_normalizer=host_normalizer,
        monotonic_seconds=lambda: 0.0,
        logger=_Logger(),  # type: ignore[arg-type]
    )

    fetcher = build_fetcher(
        project_root=tmp_path,
        settings=settings,
        logger_factory=logger_factory,  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        rate_limiter=object(),  # type: ignore[arg-type]
        metrics=object(),  # type: ignore[arg-type]
        host_normalizer=host_normalizer,
        clock=clock,
        url_validator=object(),  # type: ignore[arg-type]
        host_extractor=object(),  # type: ignore[arg-type]
        blacklist_repository=object(),  # type: ignore[arg-type]
        redirector=object(),  # type: ignore[arg-type]
        robots_request_gate=object(),  # type: ignore[arg-type]
        network_access_guard=object(),  # type: ignore[arg-type]
        host_suppression_store=host_suppression,
        source_scope_registry=object(),  # type: ignore[arg-type]
        conditional_representation_cache=ConditionalRepresentationCache(
            enabled=conditional_cache_settings.enabled,
            max_entries=conditional_cache_settings.max_entries,
            ttl_seconds=conditional_cache_settings.ttl_seconds,
            clock=lambda: 0.0,
        ),
    )

    assert type(fetcher).__name__ == "FetchOrchestrator"
