"""Page-level metadata extracted from a typed page element index.

Parity target: structural fields previously produced by
``PageAnalyzer`` (title, canonical, robots, meta-refresh). Reads only index
buckets — never calls ``find_all`` on the full document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from crawler.extraction.html.html_parser import (
    element_attribute,
    element_visible_text,
)
from crawler.extraction.modalities.page_element_index import PageElementIndex


@dataclass(frozen=True, slots=True)
class PageMetadata:
    """DOM-free page metadata for discovery and downstream consumers."""

    title: str | None = None
    canonical_url: str | None = None
    meta_robots: tuple[str, ...] = ()
    meta_refresh_url: str | None = None


class PageMetadataExtractor:
    """Extract page metadata from an indexed document structure."""

    def extract(self, *, index: PageElementIndex) -> PageMetadata:
        """Return title, canonical, robots, and meta-refresh from the index."""

        return PageMetadata(
            title=_extract_title(index=index),
            canonical_url=_extract_canonical_url(index=index),
            meta_robots=_extract_meta_robots(index=index),
            meta_refresh_url=_extract_meta_refresh_url(index=index),
        )


def _extract_title(*, index: PageElementIndex) -> str | None:
    for element in index.title_elements:
        text = _clean_string(element_visible_text(element=element))
        if text:
            return text

    for element in index.metadata_elements:
        key = _clean_string(
            element_attribute(element=element, name="property")
        ).lower()
        if key not in {"og:title", "twitter:title"}:
            key = _clean_string(
                element_attribute(element=element, name="name")
            ).lower()
        if key not in {"og:title", "twitter:title"}:
            continue
        text = _clean_string(
            element_attribute(element=element, name="content")
        )
        if text:
            return text
    return None


def _extract_canonical_url(*, index: PageElementIndex) -> str | None:
    for element in index.resource_link_elements:
        rel = _tokens(element_attribute(element=element, name="rel"))
        if "canonical" not in rel:
            continue
        href = _clean_string(element_attribute(element=element, name="href"))
        if href:
            return href
    return None


def _extract_meta_robots(*, index: PageElementIndex) -> tuple[str, ...]:
    directives: list[str] = []
    seen: set[str] = set()
    for element in index.metadata_elements:
        name = _clean_string(
            element_attribute(element=element, name="name")
        ).lower()
        if name not in {"robots", "googlebot", "bingbot"}:
            continue
        for token in _tokens(
            element_attribute(element=element, name="content")
        ):
            if token in seen:
                continue
            seen.add(token)
            directives.append(token)
    return tuple(directives)


def _extract_meta_refresh_url(*, index: PageElementIndex) -> str | None:
    for element in index.metadata_elements:
        http_equiv = _clean_string(
            element_attribute(element=element, name="http-equiv")
        ).lower()
        if http_equiv != "refresh":
            continue
        content = _clean_string(
            element_attribute(element=element, name="content")
        )
        match = re.search(
            r"(?:^|;)\s*url\s*=\s*(?P<url>[^;]+)",
            content,
            re.I,
        )
        if match is None:
            continue
        return match.group("url").strip(" '\"")
    return None


def _tokens(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (tuple, list, set)):
        raw_parts = [str(part) for part in value]
    else:
        raw_parts = re.split(r"[,\s]+", str(value))
    return tuple(
        token.strip().lower() for token in raw_parts if token and token.strip()
    )


def _clean_string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
