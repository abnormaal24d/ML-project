"""RSS/Atom feed enclosure extraction for media assets."""

from __future__ import annotations

from collections.abc import Mapping

from crawler.classification.media_kind_registry import (
    match_extension,
)


def extract_feed_enclosures(
    *, feed_entry: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    """Return media enclosure payloads from a parsed feed entry."""

    results: list[dict[str, object]] = []
    for enclosure in _iter_enclosure_mappings(feed_entry=feed_entry):
        url = enclosure.get("href") or enclosure.get("url")
        mime_type = enclosure.get("type")

        if not isinstance(url, str) or not url.strip():
            continue

        kind = _kind_from_mime(mime_type) or match_extension(url)
        if kind is None:
            continue

        results.append(
            {
                "url": url.strip(),
                "kind": kind,
                "mime_type": mime_type,
                "source_type": "feed_enclosure",
            }
        )

    return tuple(results)


def _iter_enclosure_mappings(
    *,
    feed_entry: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    candidates: list[Mapping[str, object]] = []
    for raw_collection in (
        feed_entry.get("enclosures"),
        feed_entry.get("links"),
    ):
        if not isinstance(raw_collection, (list, tuple)):
            continue
        for item in raw_collection:
            if not isinstance(item, Mapping):
                continue
            if item in candidates:
                continue
            if raw_collection is feed_entry.get("links"):
                rel = str(item.get("rel") or "").lower()
                if rel != "enclosure":
                    continue
            candidates.append(item)
    return tuple(candidates)


def _kind_from_mime(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered.startswith("audio/"):
        return "audio"
    if lowered.startswith("video/"):
        return "video"
    if lowered in {"application/pdf"}:
        return "document"
    return None
