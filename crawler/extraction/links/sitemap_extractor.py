"""Extract page and nested-sitemap candidates from parsed XML sitemaps."""

from __future__ import annotations

import re
from typing import Any

from crawler.crawl_tasks.crawl_task_context import CrawlTaskContext
from crawler.extraction.candidates.url_candidate_resolution import (
    ExtractionCandidate,
)

_SITEMAP_ROOT_NAMES = frozenset(
    {
        "urlset",
        "sitemapindex",
    }
)

_NON_SITEMAP_ROOT_NAMES = frozenset(
    {
        "html",
        "rss",
        "feed",
    }
)

_SITEMAP_ROOT_PATTERN = "|".join(
    re.escape(name) for name in sorted(_SITEMAP_ROOT_NAMES)
)

_SITEMAP_MARKUP_ROOT_RE = re.compile(
    rf"""
    \A
    \s*
    \ufeff?
    (?:<\?xml\b.*?\?>\s*)?
    (?:<!--.*?-->\s*)*
    (?:<!DOCTYPE[^>]*>\s*)*
    <
    (?:[A-Za-z_][A-Za-z0-9_.-]*:)?
    (?:{_SITEMAP_ROOT_PATTERN})
    (?:\s|/?>)
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def _local_name(value: Any) -> str:
    """Return a case-folded XML local name without a namespace prefix."""

    return (
        str(value or "")
        .strip()
        .rsplit("}", maxsplit=1)[-1]
        .rsplit(":", maxsplit=1)[-1]
        .casefold()
    )


class SitemapUrlExtractor:
    """Interpret URL records from sitemap and sitemap-index XML."""

    def extract(
        self,
        *,
        document: Any,
        max_candidates: int,
    ) -> tuple[ExtractionCandidate, ...]:
        """Return raw page and nested-sitemap candidates.

        URLs remain unresolved. URL normalization, scheme validation and
        deduplication belong to the shared candidate-resolution pipeline.
        """

        if document is None or max_candidates <= 0:
            return ()

        find_all = getattr(
            document,
            "find_all",
            None,
        )

        if not callable(find_all):
            return ()

        try:
            elements = find_all()
        except (AttributeError, TypeError, ValueError):
            return ()

        candidates: list[ExtractionCandidate] = []

        for element in elements:
            element_name = _local_name(
                getattr(
                    element,
                    "name",
                    "",
                )
            )

            if element_name != "loc":
                continue

            parent = getattr(
                element,
                "parent",
                None,
            )
            parent_name = _local_name(
                getattr(
                    parent,
                    "name",
                    "",
                )
            )

            if parent_name == "url":
                source_type = "sitemap_page"
                mime_hint = None
            elif parent_name == "sitemap":
                source_type = "sitemap_reference"
                mime_hint = "application/xml"
            else:
                # Exclude extension elements such as image:loc and video:loc.
                continue

            get_text = getattr(
                element,
                "get_text",
                None,
            )

            if callable(get_text):
                try:
                    candidate_url = str(
                        get_text(
                            separator=" ",
                            strip=True,
                        )
                    ).strip()
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                ):
                    candidate_url = ""
            else:
                candidate_url = str(
                    getattr(
                        element,
                        "string",
                        "",
                    )
                    or ""
                ).strip()

            candidate_url = " ".join(candidate_url.split())

            if not candidate_url:
                continue

            metadata: list[str] = []

            if parent is not None:
                parent_children = getattr(
                    parent,
                    "children",
                    (),
                )

                try:
                    children = tuple(parent_children)
                except TypeError:
                    children = ()

                for metadata_name in (
                    "lastmod",
                    "changefreq",
                    "priority",
                ):
                    metadata_value = ""

                    for child in children:
                        child_name = _local_name(
                            getattr(
                                child,
                                "name",
                                "",
                            )
                        )

                        if child_name != metadata_name:
                            continue

                        child_get_text = getattr(
                            child,
                            "get_text",
                            None,
                        )

                        if callable(child_get_text):
                            try:
                                metadata_value = str(
                                    child_get_text(
                                        separator=" ",
                                        strip=True,
                                    )
                                ).strip()
                            except (
                                AttributeError,
                                TypeError,
                                ValueError,
                            ):
                                metadata_value = ""
                        else:
                            metadata_value = str(
                                getattr(
                                    child,
                                    "string",
                                    "",
                                )
                                or ""
                            ).strip()

                        metadata_value = " ".join(metadata_value.split())[:120]

                        break

                    if metadata_value:
                        metadata.append(f"{metadata_name}={metadata_value}")

            candidates.append(
                ExtractionCandidate(
                    url=candidate_url,
                    kind="page",
                    context=CrawlTaskContext(
                        tag_name="loc",
                        source_tag=parent_name,
                        source_attribute="loc",
                        text_hint=candidate_url[:280],
                        surrounding_text=(" ".join(metadata)[:320] or None),
                        mime_hint=mime_hint,
                    ),
                    source_type=source_type,
                )
            )

            if len(candidates) >= max_candidates:
                break

        return tuple(candidates)


def is_sitemap_markup(
    body: bytes,
    *,
    max_bytes: int,
) -> bool:
    """Return whether raw markup starts with a sitemap XML root."""

    if not body or max_bytes <= 0:
        return False

    sample = body[:max_bytes].decode(
        "utf-8",
        errors="ignore",
    )
    return _SITEMAP_MARKUP_ROOT_RE.search(sample) is not None


def is_sitemap_document(
    document: Any,
) -> bool:
    """Return whether the parsed document has a sitemap XML root."""

    if document is None:
        return False

    document_name = _local_name(
        getattr(
            document,
            "name",
            "",
        )
    )

    if document_name in _SITEMAP_ROOT_NAMES:
        return True

    if document_name in _NON_SITEMAP_ROOT_NAMES:
        return False

    find_all = getattr(
        document,
        "find_all",
        None,
    )

    if not callable(find_all):
        return False

    try:
        elements = find_all()
    except (AttributeError, TypeError, ValueError):
        return False

    for element in elements:
        element_name = _local_name(
            getattr(
                element,
                "name",
                "",
            )
        )

        if element_name in _SITEMAP_ROOT_NAMES:
            return True

        if element_name in _NON_SITEMAP_ROOT_NAMES:
            return False

    return False


__all__ = [
    "SitemapUrlExtractor",
    "is_sitemap_document",
    "is_sitemap_markup",
]
