"""Extract normalized links from HTML pages and XML sitemaps."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup, Tag

from crawler.crawl_tasks.crawl_task_context import CrawlTaskContext
from crawler.crawl_tasks.crawl_task_context_builder import (
    coerce_crawl_task_context,
)
from crawler.extraction.candidates.url_candidate_resolution import (
    ExtractionCandidate,
)
from crawler.extraction.html.html_parser import (
    element_attribute,
    element_tag_name,
    element_visible_text,
)
from crawler.extraction.links.sitemap_extractor import (
    SitemapUrlExtractor,
    is_sitemap_document,
)
from crawler.extraction.modalities.page_element_index import (
    PageElementIndex,
    PageElementIndexBuilder,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.extraction import LinkExtractorSettings
    from crawler.extraction.candidates.url_candidate_resolution import (
        UrlCandidateResolution,
    )

ParsedDocument = BeautifulSoup | Tag

_TEXT_HINT_ATTRIBUTES = (
    "aria-label",
    "title",
    "alt",
    "label",
)

_SURROUNDING_TAGS = (
    "article",
    "section",
    "li",
    "div",
    "p",
)


def _is_nofollow_link(element: Any) -> bool:
    """Return whether an element declares a nofollow relation."""

    rel = element_attribute(element=element, name="rel")

    tokens: Iterator[str]
    if isinstance(rel, str):
        tokens = iter(rel.split())
    elif isinstance(rel, (tuple, list)):
        tokens = (token for token in rel if isinstance(token, str))
    else:
        return False

    return any(token.casefold() == "nofollow" for token in tokens)


class LinkExtractor:
    """Extract, resolve and deduplicate links from indexed page elements."""

    def __init__(
        self,
        *,
        settings: LinkExtractorSettings,
        candidate_resolution: UrlCandidateResolution,
        logger: ProjectLogger,
    ) -> None:
        if candidate_resolution is None:
            raise ValueError("candidate_resolution is required")

        self._settings = settings
        self._candidate_resolution = candidate_resolution
        self._logger = logger
        self._sitemap_extractor = SitemapUrlExtractor()

    def extract_candidates(
        self,
        *,
        index: PageElementIndex,
        base_url: str,
    ) -> tuple[ExtractionCandidate, ...]:
        """Return resolved link candidates from a typed page element index.

        Reads only ``index.link_elements`` and ``index.resource_link_elements``.
        Performs no ``find_all`` and no additional DOM traversal.
        """

        if not self._settings.enabled:
            return ()

        max_candidates = max(0, int(self._settings.max_links_per_page))
        if max_candidates == 0:
            return ()

        return self._resolve_candidates(
            base_url=base_url,
            raw_candidates=self._raw_candidates_from_index(
                index=index,
                limit=max_candidates,
            ),
            document_kind="html",
        )

    def extract_urls(
        self,
        *,
        base_url: str,
        document: ParsedDocument,
    ) -> tuple[str, ...]:
        """Return normalized URLs extracted from one document."""

        return tuple(
            candidate.url
            for candidate in self.iter_extract_candidates(
                base_url=base_url,
                document=document,
            )
        )

    def iter_extract_candidates(
        self,
        *,
        base_url: str,
        document: ParsedDocument,
    ) -> Iterator[ExtractionCandidate]:
        """Yield resolved and deduplicated extraction candidates.

        Sitemap documents still use the document-based sitemap extractor.
        HTML pages are indexed once, then processed via
        :meth:`extract_candidates` buckets.
        """

        max_candidates = max(
            0,
            int(self._settings.max_links_per_page),
        )

        if max_candidates == 0:
            return

        sitemap_document = is_sitemap_document(document)

        if sitemap_document:
            raw_candidates: (
                tuple[ExtractionCandidate, ...] | Iterator[ExtractionCandidate]
            ) = self._sitemap_extractor.extract(
                document=document,
                max_candidates=max_candidates,
            )
            yield from self._resolve_candidates(
                base_url=base_url,
                raw_candidates=raw_candidates,
                document_kind="sitemap",
            )
            return

        index = PageElementIndexBuilder().build(document=document)
        yield from self.extract_candidates(index=index, base_url=base_url)

    def _raw_candidates_from_index(
        self,
        *,
        index: PageElementIndex,
        limit: int,
    ) -> Iterator[ExtractionCandidate]:
        """Yield unresolved candidates from index link buckets only."""

        configured_tags = {
            str(tag).strip().casefold()
            for tag in self._settings.tags
            if str(tag).strip()
        }
        attribute = self._settings.attribute
        emitted = 0

        for element in index.link_elements:
            if emitted >= limit:
                return
            tag_name = element_tag_name(element=element)
            if tag_name not in configured_tags:
                continue
            candidate = self._candidate_from_element(
                element=element,
                tag_name=tag_name,
                attribute=attribute,
            )
            if candidate is None:
                continue
            yield candidate
            emitted += 1

        for element in index.resource_link_elements:
            if emitted >= limit:
                return
            if not self._include_resource_link(element=element):
                continue
            candidate = self._candidate_from_element(
                element=element,
                tag_name="link",
                attribute=attribute,
            )
            if candidate is None:
                continue
            yield candidate
            emitted += 1

    def _include_resource_link(self, *, element: Any) -> bool:
        """Decide whether a ``<link>`` element is a discovery candidate."""

        rel_values = _rel_tokens(element)
        if not rel_values:
            return False

        if (
            self._settings.extract_canonical_links
            and "canonical" in rel_values
        ):
            return True

        if self._settings.extract_feed_links:
            if "feed" in rel_values:
                return True
            if "alternate" in rel_values:
                type_value = element_attribute(element=element, name="type")
                if isinstance(type_value, str):
                    lowered = type_value.casefold()
                    if any(
                        token in lowered
                        for token in (
                            "rss",
                            "atom",
                            "xml",
                            "json",
                        )
                    ):
                        return True

        if self._settings.extract_media_links:
            as_value = element_attribute(element=element, name="as")
            if isinstance(as_value, str) and as_value.strip().casefold() in {
                "document",
                "fetch",
            }:
                return True

        # Explicit inclusion when settings.tags contains "link".
        if "link" in {
            str(tag).strip().casefold() for tag in self._settings.tags
        }:
            # Still skip pure asset resource links.
            if rel_values.intersection(
                {
                    "stylesheet",
                    "icon",
                    "shortcut",
                    "apple-touch-icon",
                    "manifest",
                    "dns-prefetch",
                    "preconnect",
                    "modulepreload",
                    "preload",
                }
            ):
                return False
            return True

        return False

    def _candidate_from_element(
        self,
        *,
        element: Any,
        tag_name: str,
        attribute: str,
    ) -> ExtractionCandidate | None:
        if not self._settings.include_nofollow_links and _is_nofollow_link(
            element
        ):
            return None

        candidate_url = element_attribute(element=element, name=attribute)
        if not isinstance(candidate_url, str):
            return None

        candidate_url = candidate_url.strip()
        if not candidate_url:
            return None

        return ExtractionCandidate(
            url=candidate_url,
            kind=None,
            context=build_link_task_context(
                element=element,
                tag_name=tag_name,
                source_attribute=attribute,
            ),
            source_type="discovered_link",
        )

    def _resolve_candidates(
        self,
        *,
        base_url: str,
        raw_candidates: (
            tuple[ExtractionCandidate, ...] | Iterator[ExtractionCandidate]
        ),
        document_kind: str,
    ) -> tuple[ExtractionCandidate, ...]:
        seen_urls: set[str] = set()
        accepted: list[ExtractionCandidate] = []
        rejected_count = 0
        duplicate_count = 0

        for candidate in raw_candidates:
            resolved_url = self._candidate_resolution.resolve(
                base_url=base_url,
                candidate=candidate.url,
            )

            if resolved_url is None:
                rejected_count += 1
                continue

            if resolved_url in seen_urls:
                duplicate_count += 1
                continue

            seen_urls.add(resolved_url)
            accepted.append(
                ExtractionCandidate(
                    url=resolved_url,
                    kind=candidate.kind,
                    context=candidate.context,
                    asset=candidate.asset,
                    source_type=candidate.source_type,
                )
            )

        self._logger.debug(
            "link_extracted",
            base_url=base_url,
            count=len(accepted),
            rejected_count=rejected_count,
            duplicate_count=duplicate_count,
            document_kind=document_kind,
            tags=tuple(self._settings.tags),
        )
        return tuple(accepted)


def build_link_task_context(
    *,
    element: Any,
    tag_name: str,
    source_attribute: str | None = None,
) -> CrawlTaskContext | None:
    """Build normalized crawl context for an HTML link element."""

    text_parts: list[str] = []

    element_text = element_visible_text(element=element)
    if element_text:
        text_parts.append(element_text)

    for attribute in _TEXT_HINT_ATTRIBUTES:
        attribute_value = element_attribute(element=element, name=attribute)

        if attribute_value is None:
            continue

        values: Iterator[object]
        if isinstance(attribute_value, str):
            values = iter((attribute_value,))
        elif isinstance(attribute_value, (tuple, list)):
            values = iter(attribute_value)
        else:
            values = iter((str(attribute_value),))

        for value in values:
            normalized_value = " ".join(str(value).split())

            if normalized_value:
                text_parts.append(normalized_value)

    normalized_parts: list[str] = []
    seen_parts: set[str] = set()

    for part in text_parts:
        normalized_part = " ".join(part.split())

        if not normalized_part:
            continue

        identity = normalized_part.casefold()

        if identity in seen_parts:
            continue

        seen_parts.add(identity)
        normalized_parts.append(normalized_part)

    surrounding_text: str | None = None

    find_parent = getattr(element, "find_parent", None)
    if callable(find_parent):
        figure = find_parent("figure")
        if figure is not None and bool(element_tag_name(element=figure)):
            find_caption = getattr(figure, "find", None)
            caption = (
                find_caption("figcaption") if callable(find_caption) else None
            )
            context_element = (
                caption
                if caption is not None
                and bool(element_tag_name(element=caption))
                else figure
            )
            figure_text = element_visible_text(element=context_element)
            if figure_text:
                surrounding_text = figure_text

        if surrounding_text is None:
            for surrounding_tag in _SURROUNDING_TAGS:
                ancestor = find_parent(surrounding_tag)
                if ancestor is None or not bool(
                    element_tag_name(element=ancestor)
                ):
                    continue
                ancestor_text = element_visible_text(element=ancestor)
                if not ancestor_text:
                    continue
                surrounding_text = ancestor_text
                break

    type_value = element_attribute(element=element, name="type")

    if isinstance(type_value, str):
        mime_hint: str | None = type_value
    elif isinstance(type_value, (tuple, list)):
        mime_hint = " ".join(str(value) for value in type_value if value)
    elif type_value is None:
        mime_hint = None
    else:
        mime_hint = str(type_value)

    return coerce_crawl_task_context(
        {
            "tag_name": tag_name,
            "source_tag": tag_name,
            "source_attribute": source_attribute,
            "text_hint": " ".join(normalized_parts) or None,
            "surrounding_text": surrounding_text,
            "mime_hint": mime_hint,
        }
    )


def _rel_tokens(element: Any) -> set[str]:
    rel = element_attribute(element=element, name="rel")
    if rel is None:
        return set()
    if isinstance(rel, str):
        return {
            token.strip().casefold() for token in rel.split() if token.strip()
        }
    if isinstance(rel, (tuple, list, set)):
        values: set[str] = set()
        for item in rel:
            if isinstance(item, str) and item.strip():
                values.add(item.strip().casefold())
        return values
    return set()


__all__ = [
    "LinkExtractor",
    "build_link_task_context",
]
