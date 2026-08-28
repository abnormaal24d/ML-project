"""Extract image URLs from inline CSS declarations."""

from __future__ import annotations

import re

_BACKGROUND_URL_RE = re.compile(
    r"background(?:-image)?\s*:\s*url\("
    r"(?P<quote>['\"]?)(?P<url>.*?)(?P=quote)\)",
    re.IGNORECASE,
)


def extract_background_image_urls(*, style: object) -> tuple[str, ...]:
    """Return background-image URL references from an inline style value."""

    if not isinstance(style, str):
        return ()

    return tuple(
        url
        for match in _BACKGROUND_URL_RE.finditer(style)
        for url in (match.group("url").strip(),)
        if url
    )
