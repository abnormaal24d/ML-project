"""Feed content extraction: parse once, structural fields only.

Owns feed parsing, feed metadata, entry links, and enclosure dispatch.
Does not schedule discovery, dedupe URLs, or persist records. Media kind
hints on enclosures come from shared enclosure helpers; candidate
resolution remains with the discovery/collector layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

from crawler.extraction.assets.structured_data.feed_enclosure_extractor import (
    extract_feed_enclosures,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.fetching.results.result import FetchResult


@dataclass(frozen=True, slots=True)
class FeedContentExtractionResult:
    """DOM-free structural output of one feed payload."""

    entry_links: tuple[str, ...]
    media_enclosure_links: tuple[str, ...]
    title: str | None
    media_enclosures: tuple[dict[str, object], ...] = ()


class FeedContentExtractor:
    """Extract feed structure from an already-fetched payload."""

    def __init__(
        self,
        *,
        parser: Any,
        max_entries: int,
        logger: ProjectLogger,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._parser = parser
        self._max_entries = int(max_entries)
        self._logger = logger

    def extract(
        self,
        *,
        fetch_result: FetchResult,
    ) -> FeedContentExtractionResult:
        """Parse the feed body once and return structural fields only."""

        try:
            body = fetch_result.read_body_required()
        except (OSError, ValueError, RuntimeError):
            return FeedContentExtractionResult(
                entry_links=(),
                media_enclosure_links=(),
                title=None,
            )

        try:
            parsed = self._parser.parse(body)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            self._logger.warning("feed_parse_failed")
            return FeedContentExtractionResult(
                entry_links=(),
                media_enclosure_links=(),
                title=None,
            )

        entry_links, media_enclosures = _extract_entry_fields(
            parsed=parsed,
            base_url=fetch_result.final_url,
            max_entries=self._max_entries,
        )
        return FeedContentExtractionResult(
            entry_links=entry_links,
            media_enclosure_links=tuple(
                str(item["url"])
                for item in media_enclosures
                if item.get("url")
            ),
            title=_extract_feed_title(parsed=parsed),
            media_enclosures=media_enclosures,
        )


def _extract_entry_fields(
    *,
    parsed: Any,
    base_url: str,
    max_entries: int,
) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
    if not isinstance(parsed, dict):
        return (), ()

    entries = parsed.get("entries", ()) or ()
    if not isinstance(entries, (list, tuple)):
        return (), ()

    links: list[str] = []
    enclosures: list[dict[str, object]] = []

    for entry in entries[:max_entries]:
        if not isinstance(entry, dict):
            continue

        link = entry.get("link")
        if link:
            links.append(urljoin(base_url, str(link).strip()))

        for enc in extract_feed_enclosures(feed_entry=entry):
            url = enc.get("url")
            if url:
                enclosures.append(
                    {
                        **enc,
                        "url": urljoin(base_url, str(url).strip()),
                    }
                )

    return tuple(links), tuple(enclosures)


def _extract_feed_title(*, parsed: Any) -> str | None:
    if not isinstance(parsed, dict):
        return None
    feed = parsed.get("feed")
    if not isinstance(feed, dict):
        return None
    raw_title = feed.get("title")
    if not raw_title:
        return None
    return str(raw_title).strip() or None
