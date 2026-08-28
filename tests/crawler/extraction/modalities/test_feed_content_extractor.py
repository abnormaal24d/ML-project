"""Tests for FeedContentExtractor structural feed parsing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from crawler.classification.media_kind import MediaKind
from crawler.extraction.modalities.feed_content_extractor import (
    FeedContentExtractor,
)
from crawler.fetching.results.payload import FetchedPayload
from crawler.fetching.results.result import FetchResult
from crawler.processing.processors.processor_failure_handler import (
    ProcessorFailureHandler,
)
from tests.support.logging import TEST_LOGGER

_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Example Feed</title>
    <link>https://example.test/</link>
    <item>
      <title>Post One</title>
      <link>https://example.test/posts/1</link>
      <enclosure url="https://cdn.example.test/a.mp3" type="audio/mpeg" />
    </item>
    <item>
      <title>Post Two</title>
      <link>/posts/2</link>
    </item>
  </channel>
</rss>
"""


class _FakeFeedParser:
    def parse(self, body: bytes | str):
        # Minimal feedparser-like mapping for unit tests without network.
        text = body.decode("utf-8") if isinstance(body, bytes) else str(body)
        if "Example Feed" not in text:
            return {"feed": {}, "entries": []}
        return {
            "feed": {"title": "Example Feed"},
            "entries": [
                {
                    "link": "https://example.test/posts/1",
                    "enclosures": [
                        {
                            "url": "https://cdn.example.test/a.mp3",
                            "type": "audio/mpeg",
                        }
                    ],
                },
                {"link": "/posts/2"},
            ],
        }


def _fetch_result(tmp_path: Path, body: bytes) -> FetchResult:
    path = tmp_path / "feed.xml"
    path.write_bytes(body)
    return FetchResult(
        url="https://example.test/feed.xml",
        final_url="https://example.test/feed.xml",
        status_code=200,
        headers={"content-type": "application/rss+xml"},
        fetched_at="2024-01-01T00:00:00Z",
        content_type="application/rss+xml",
        mime_type="application/rss+xml",
        encoding="utf-8",
        language=None,
        kind=MediaKind.FEED,
        payload=FetchedPayload(
            temp_path=path,
            byte_size=len(body),
            sha256_hex="a" * 64,
            sniff_bytes=body[:64],
            chunk_count=1,
        ),
        body_sha256="a" * 64,
    )


def test_extract_entry_links_and_enclosures(tmp_path: Path) -> None:
    extractor = FeedContentExtractor(
        parser=_FakeFeedParser(),
        max_entries=10,
        logger=TEST_LOGGER,
    )
    result = extractor.extract(fetch_result=_fetch_result(tmp_path, _RSS))
    assert result.title == "Example Feed"
    assert result.entry_links == (
        "https://example.test/posts/1",
        "https://example.test/posts/2",
    )
    assert result.media_enclosure_links == ("https://cdn.example.test/a.mp3",)
    assert result.media_enclosures[0]["kind"] == "audio"


def test_extract_respects_max_entries(tmp_path: Path) -> None:
    extractor = FeedContentExtractor(
        parser=_FakeFeedParser(),
        max_entries=1,
        logger=TEST_LOGGER,
    )
    result = extractor.extract(fetch_result=_fetch_result(tmp_path, _RSS))
    assert result.entry_links == ("https://example.test/posts/1",)
    assert result.media_enclosure_links == ("https://cdn.example.test/a.mp3",)


def test_extract_handles_parse_failure(tmp_path: Path) -> None:
    class _Broken:
        def parse(self, body: object) -> object:
            raise ValueError("boom")

    warnings: list[str] = []

    class _Logger:
        def warning(self, event: str, **kwargs: object) -> None:
            warnings.append(event)

    extractor = FeedContentExtractor(
        parser=_Broken(),
        max_entries=5,
        logger=_Logger(),
    )
    result = extractor.extract(
        fetch_result=_fetch_result(tmp_path, b"not-xml")
    )
    assert result.entry_links == ()
    assert result.title is None
    assert warnings == ["feed_parse_failed"]


def test_max_entries_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        FeedContentExtractor(
            parser=SimpleNamespace(), max_entries=0, logger=TEST_LOGGER
        )


def test_feed_handler_uses_content_extractor_directly(
    tmp_path: Path,
) -> None:
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from crawler.processing.handlers.feed_handler import FeedHandler

    extractor = FeedContentExtractor(
        parser=_FakeFeedParser(),
        max_entries=10,
        logger=TEST_LOGGER,
    )
    handler = FeedHandler(
        settings=SimpleNamespace(
            schedule_entry_links=False,
            max_feed_entries=10,
        ),
        url_filter=None,
        url_normalizer=None,
        scheduler=MagicMock(),
        dataset_writer=MagicMock(),
        logger=MagicMock(),
        failure_handler=ProcessorFailureHandler(
            default_retry_wait_seconds=5.0
        ),
        feed_content_extractor=extractor,
        id_generator=SimpleNamespace(generate=lambda: "task-001"),
        host_normalizer=MagicMock(),
    )
    fetch = _fetch_result(tmp_path, _RSS)
    analysis = asyncio.run(
        handler.prepare_analysis(
            task=SimpleNamespace(source_name="s", depth=0),
            result=fetch,
        )
    )
    assert analysis.title == "Example Feed"
    assert len(analysis.entry_links) == 2
    ok, reason, fields = asyncio.run(
        handler.validate_result(
            task=SimpleNamespace(),
            result=fetch,
            analysis=analysis,
        )
    )
    assert ok is True
    assert reason is None
    assert fields["feed_entry_count"] == 2


def test_feed_handler_discovery_depth_semantics(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from crawler.processing.handlers.feed_handler import FeedHandler

    handler = FeedHandler(
        settings=SimpleNamespace(
            schedule_entry_links=True,
            max_feed_items_discovered=10,
            max_discovered_links_per_host=10,
            max_audio_links=10,
            discovery_queue_high_watermark=0,
            discovery_queue_critical_watermark=0,
            max_feed_items_discovered_critical=10,
            max_feed_items_discovered_under_pressure=10,
        ),
        url_filter=None,
        url_normalizer=None,
        scheduler=MagicMock(),
        dataset_writer=MagicMock(),
        logger=MagicMock(),
        failure_handler=ProcessorFailureHandler(
            default_retry_wait_seconds=5.0
        ),
        feed_content_extractor=MagicMock(),
        id_generator=SimpleNamespace(generate=lambda: "task-001"),
        host_normalizer=SimpleNamespace(
            normalize=lambda _host: "example.test"
        ),
    )
    selection = handler._select_discovered_tasks(
        task=SimpleNamespace(source_name="s", depth=4),
        result=_fetch_result(tmp_path, _RSS),
        entry_links=("https://example.test/posts/1",),
        media_enclosures=(
            {
                "url": "https://cdn.example.test/a.mp3",
                "kind": "audio",
                "mime_type": "audio/mpeg",
            },
        ),
    )

    assert [task.kind for task in selection.tasks] == [
        MediaKind.PAGE,
        MediaKind.AUDIO,
    ]
    assert [task.depth for task in selection.tasks] == [5, 4]
    assert selection.tasks[1].context.source_page_depth == 4
