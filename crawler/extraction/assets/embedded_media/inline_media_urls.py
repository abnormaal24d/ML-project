"""Shared inline media URL scanning helpers for modality extractors."""

from __future__ import annotations

import re

from crawler.extraction.assets.candidate.asset_extraction_records import (
    clean_string,
)

_INLINE_MEDIA_URL_PATTERN = re.compile(
    r"""(?P<quote>["'])"""
    r"""(?P<url>(?:(?:https?:)?//|/|[A-Za-z0-9_.-]+/)"""
    r"""[^"'<>\\\s]+?\."""
    r"""(?:mp4|webm|mov|m4v|m3u8|mp3|m4a|wav|aac|flac|ogg|opus|"""
    r"""vtt|srt|ttml|pdf|docx?|pptx?|xlsx?|csv|tsv)"""
    r"""(?:\?[^"'<>\\\s]*)?)"""
    r"""(?P=quote)""",
    re.IGNORECASE,
)


def looks_like_inline_media_config(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            ".m3u8",
            ".mp4",
            ".webm",
            ".mov",
            ".m4v",
            ".mp3",
            ".m4a",
            ".wav",
            ".aac",
            ".flac",
            ".ogg",
            ".opus",
            ".pdf",
            ".doc",
            ".docx",
            ".ppt",
            ".pptx",
            ".csv",
            ".vtt",
            ".srt",
            "contenturl",
            "embedurl",
            "player",
            "transcript",
            "download",
            "enclosure",
        )
    )


def iter_inline_media_urls(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    urls: list[str] = []

    for match in _INLINE_MEDIA_URL_PATTERN.finditer(text):
        url = clean_string(match.group("url"))
        if url is None or url in seen:
            continue
        seen.add(url)
        urls.append(url)

    return tuple(urls)
