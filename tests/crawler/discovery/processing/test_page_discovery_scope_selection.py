"""Regression tests: crawl-scope preflight and canonicalize-once selection.

Invariants (scheduler crawl-scope governance):

- Candidates whose host is outside the current crawl scope must not consume
  selection slots: they are counted as scope_blocked, never as rejected or
  truncated, and never appear in the rejected-assets output.
- URL canonicalization happens exactly once at the start of selection; URL
  filtering, deduplication, scope preflight, and ranking all see the same
  canonical URL.
- Scope preflight is keyed by full discovery identity (url|kind|source_type),
  not by URL alone, to distinguish same-URL different-task candidates.
"""

from __future__ import annotations

from types import SimpleNamespace

from config.collection.extraction import UrlNormalizerSettings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.discovery.processing.page_discovery_selection import (
    PageDiscoverySelectionRequest,
    discovery_dedupe_key,
    select_page_discovery_tasks,
)
from crawler.discovery.task_identity import discovered_task_identity
from crawler.extraction.urls.normalizer import UrlNormalizer
from crawler.governance.domains.host_normalizer import HostNormalizer


def _logger() -> SimpleNamespace:
    return SimpleNamespace(
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )


def _ranking() -> SimpleNamespace:
    return SimpleNamespace(
        kind_weights={"page": 1.0, "image": 1.0},
        default_kind_weight=1.0,
        discovered_link_bonus=0.5,
        embedded_media_asset_penalty=0.5,
        embedded_asset_penalty=1.0,
        non_link_source_penalty=1.0,
        media_page_bonus=2.0,
        page_bonus=0.5,
        page_bonus_tokens=(),
        page_penalty_tokens=(),
        page_penalty=0.5,
        asset_path_penalty=0.5,
        asset_penalty_tokens=(),
        query_penalty=0.25,
    )


def _task(*, url: str, kind: MediaKind = MediaKind.PAGE) -> CrawlTask:
    return CrawlTask(
        url=url,
        source_name="test",
        kind=kind,
        source_type="discovered_link",
    )


def _identity(task: CrawlTask) -> str:
    """Return the full discovery identity for a task."""
    return discovered_task_identity(task=task, normalized_url=task.url)


def _select(
    *,
    tasks: list[CrawlTask],
    scope_eligibility: dict[str, bool] | None = None,
    url_normalizer: UrlNormalizer | None = None,
    max_total: int = 10,
) -> object:
    return select_page_discovery_tasks(
        request=PageDiscoverySelectionRequest(
            task_stream=tasks,
            max_total=max_total,
            max_pages=max_total,
            max_embedded_assets=max_total,
            max_non_page_media=max_total,
            ranking=_ranking(),
            active_focus_kinds=(),
            focus_asset_boost=0.0,
            host_normalizer=HostNormalizer(),
            url_filter=None,
            url_normalizer=url_normalizer,
            scope_eligibility=scope_eligibility,
        )
    )


def test_scope_blocked_candidates_do_not_consume_selection_slots() -> None:
    allowed = [_task(url=f"https://a{i}.example/page") for i in range(14)]
    blocked = [_task(url=f"https://b{i}.example/page") for i in range(6)]
    scope_eligibility = {_identity(t): False for t in blocked}

    selection = _select(
        tasks=allowed + blocked,
        scope_eligibility=scope_eligibility,
    )

    assert len(selection.tasks) == 10
    assert selection.scope_blocked_count == 6
    assert selection.truncated_count == 4
    assert selection.filtered_count == 0
    assert all("b" not in task.url[8] for task in selection.tasks)
    assert len(selection.scope_blocked_tasks) == 6
    assert all("b" in task.url[8] for task in selection.scope_blocked_tasks)


def test_scope_blocked_recorded_separately_in_metrics() -> None:
    blocked = [_task(url=f"https://b{i}.example/page") for i in range(3)]

    selection = _select(
        tasks=blocked,
        scope_eligibility={_identity(t): False for t in blocked},
    )

    payload = selection.metrics
    assert payload["scope_blocked_by_kind"] == {"page": 3}
    assert payload["scope_blocked_by_reason"] == {
        "page:crawl_scope_blocked": 3
    }
    assert payload["filtered_by_kind"] == {}
    assert payload["truncated_by_kind"] == {}
    assert selection.tasks == ()


def test_scope_preflight_absent_passes_everything_through() -> None:
    candidates = [_task(url=f"https://a{i}.example/page") for i in range(4)]

    selection = _select(tasks=candidates)

    assert len(selection.tasks) == 4
    assert selection.scope_blocked_count == 0


def _canonical_identity(task: CrawlTask, normalizer: UrlNormalizer) -> str:
    """Return the discovery identity after URL canonicalization."""
    from crawler.discovery.processing.discovered_url_normalization import (
        dedupe_url_key,
    )

    normalized_url = normalizer.normalize(task.url)
    if not normalized_url:
        normalized_url = dedupe_url_key(task.url)
    return f"{normalized_url}|{task.kind.value}|{task.source_type}"


def test_scope_preflight_key_matches_canonicalized_candidate_url() -> None:
    normalizer = UrlNormalizer(
        settings=UrlNormalizerSettings(),
        logger=_logger(),
        host_normalizer=HostNormalizer(),
    )
    blocked = _task(url="HTTPS://BLOCKED.example:443/page?utm_source=x#frag")
    allowed = _task(url="https://allowed.example/page")

    selection = _select(
        tasks=[blocked, allowed],
        scope_eligibility={
            _canonical_identity(blocked, normalizer): False,
        },
        url_normalizer=normalizer,
    )

    assert len(selection.tasks) == 1
    assert selection.tasks[0].url == "https://allowed.example/page"
    assert selection.scope_blocked_count == 1
    assert selection.scope_blocked_tasks[0].url == (
        "https://blocked.example/page"
    )


def test_canonicalization_happens_once_before_dedupe_and_selection() -> None:
    normalizer = UrlNormalizer(
        settings=UrlNormalizerSettings(upgrade_http_to_https=True),
        logger=_logger(),
        host_normalizer=HostNormalizer(),
    )
    raw_page = _task(url="HTTP://EXAMPLE.com:80/article?utm_source=news#frag")
    case_variant = _task(url="https://example.com/article?utm_source=other")
    image = _task(
        url="https://example.com/photo.jpg?w=100&utm_source=news",
        kind=MediaKind.IMAGE,
    )

    selection = _select(
        tasks=[raw_page, case_variant, image],
        url_normalizer=normalizer,
    )

    selected_urls = [task.url for task in selection.tasks]
    assert "https://example.com/article" in selected_urls
    assert "https://example.com/photo.jpg" in selected_urls
    assert "EXAMPLE" not in " ".join(selected_urls)
    assert "utm_source" not in " ".join(selected_urls)
    assert selection.duplicate_count == 1
    assert len(selection.tasks) == 2


def test_canonicalization_preserves_distinct_schemes_without_upgrade() -> None:
    normalizer = UrlNormalizer(
        settings=UrlNormalizerSettings(),
        logger=_logger(),
        host_normalizer=HostNormalizer(),
    )

    selection = _select(
        tasks=[
            _task(url="http://example.com/article"),
            _task(url="https://example.com/article"),
        ],
        url_normalizer=normalizer,
    )

    assert [task.url for task in selection.tasks] == [
        "http://example.com/article",
        "https://example.com/article",
    ]
    assert selection.duplicate_count == 0


def test_discovery_dedupe_key_uses_canonical_task_url_only() -> None:
    canonical = _task(url="https://example.com/article")

    key = discovery_dedupe_key(task=canonical)

    assert key == f"{canonical.url}|page|discovered_link"
    assert key == canonical.url + "|page|discovered_link"
