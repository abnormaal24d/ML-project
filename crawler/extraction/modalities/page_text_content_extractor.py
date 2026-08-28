"""Structural HTML text extraction for already-parsed page documents.

Must never re-parse HTML. One text-content traversal over the shared document
is expected per page after the structural index traversal. Preprocessing keeps
normalisation, privacy, and quality-only responsibilities.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any

import soupsieve

if TYPE_CHECKING:
    from config.collection.processors import PageTextExtractionSettings


_HEADING_TAGS = frozenset(("h1", "h2", "h3", "h4", "h5", "h6"))
_MARKDOWN_BLOCK_TAGS = _HEADING_TAGS | frozenset(("p", "li", "pre"))
_BOILERPLATE_TOKENS = frozenset(
    {
        "cookie",
        "cookies",
        "privacy",
        "subscribe",
        "subscription",
        "copyright",
        "login",
        "sign",
        "menu",
        "navigation",
        "footer",
        "share",
        "social",
        "terms",
        "conditions",
        "search",
        "contact",
        "newsletter",
    }
)


@dataclass(frozen=True, slots=True)
class PageTextContent:
    """DOM-free structural text extraction for one page."""

    text: str
    text_preview: str
    markdown: str
    headings: tuple[str, ...]
    char_count: int
    code_block_count: int
    boilerplate_ratio: float
    extraction_warnings: tuple[str, ...]


class PageTextContentExtractor:
    """Extract text, markdown, and headings from an already-parsed document.

    Must never re-parse HTML. Uses hierarchical DOM walks for main-section
    selection, dropped subtrees, headings, and markdown order. The page
    orchestrator is responsible for one parse and one structural index
    traversal; this class performs its own text-content traversal only.
    """

    def __init__(
        self,
        *,
        settings: PageTextExtractionSettings | None = None,
    ) -> None:
        from config.collection.processors import (
            PageTextExtractionSettings as _Settings,
        )

        resolved = settings or _Settings()
        if resolved.max_text_chars < 1:
            raise ValueError("max_text_chars must be >= 1")
        if resolved.preview_max_chars < 0:
            raise ValueError("preview_max_chars must be >= 0")

        self._max_text_chars = int(resolved.max_text_chars)
        self._preview_max_chars = int(resolved.preview_max_chars)
        self._drop_tags = tuple(
            tag.strip().lower()
            for tag in resolved.drop_tags
            if tag and str(tag).strip()
        )
        self._drop_selectors = tuple(
            selector.strip()
            for selector in resolved.drop_selectors
            if selector and str(selector).strip()
        )
        self._main_selectors = tuple(
            selector.strip()
            for selector in resolved.main_selectors
            if selector and str(selector).strip()
        )

    def extract(
        self,
        *,
        document: object,
    ) -> PageTextContent:
        """Extract structural text from a shared already-parsed document."""

        warnings: list[str] = []
        dropped = _collect_dropped_node_ids(
            document=document,
            tag_names=self._drop_tags,
            selectors=self._drop_selectors,
        )
        main_section = _find_main_section(
            document=document,
            selectors=self._main_selectors,
        )
        if main_section is None:
            warnings.append("main_section_not_found")
            main_section = _document_body_or_root(document)

        section = _extract_section_content(
            section=main_section,
            max_text_chars=self._max_text_chars,
            dropped_node_ids=dropped,
        )
        if section.truncated:
            warnings.append("text_truncated")
        if not section.text:
            warnings.append("empty_extracted_text")

        boilerplate_ratio = _compute_boilerplate_token_ratio(text=section.text)
        preview = section.text[: self._preview_max_chars].strip()

        return PageTextContent(
            text=section.text,
            text_preview=preview,
            markdown=section.markdown,
            headings=section.headings,
            char_count=len(section.text),
            code_block_count=section.code_block_count,
            boilerplate_ratio=boilerplate_ratio,
            extraction_warnings=tuple(warnings),
        )


@dataclass(slots=True)
class _MarkdownBlockContext:
    tag_name: str
    tokens: list[str]
    markdown_index: int


@dataclass(frozen=True, slots=True)
class _SectionContent:
    text: str
    markdown: str
    headings: tuple[str, ...]
    code_block_count: int
    truncated: bool


class _NormalizedTextAssembler:
    def __init__(self, *, max_chars: int) -> None:
        self._buffer = StringIO()
        self._has_token = False
        self._max_chars = max_chars
        self._length = 0
        self.truncated = False

    def add_tokens(self, tokens: Iterable[str]) -> tuple[str, ...]:
        accepted_tokens: list[str] = []
        for token in tokens:
            if self.truncated:
                break
            if self._has_token:
                if self._length >= self._max_chars:
                    self.truncated = True
                    break
                self._buffer.write(" ")
                self._length += 1

            remaining_chars = self._max_chars - self._length
            if remaining_chars <= 0:
                self.truncated = True
                break

            accepted_token = token[:remaining_chars]
            if len(accepted_token) < len(token):
                self.truncated = True

            if accepted_token:
                self._buffer.write(accepted_token)
                self._length += len(accepted_token)
                accepted_tokens.append(accepted_token)
                self._has_token = True

            if self.truncated:
                break

        return tuple(accepted_tokens)

    def build(self) -> str:
        return self._buffer.getvalue()


def _extract_section_content(
    *,
    section: Any,
    max_text_chars: int,
    dropped_node_ids: set[int],
) -> _SectionContent:
    text_assembler = _NormalizedTextAssembler(max_chars=max_text_chars)
    markdown_blocks: list[str] = []
    active_blocks: list[_MarkdownBlockContext] = []
    headings: list[str] = []
    code_block_count = 0

    def walk(parent: Any) -> bool:
        nonlocal code_block_count

        children = getattr(parent, "children", None)
        if children is None:
            return False

        for child in children:
            if _is_navigable_string(child):
                tokens = str(child).split()
                if not tokens:
                    continue
                accepted_tokens = text_assembler.add_tokens(tokens)
                for active_context in active_blocks:
                    active_context.tokens.extend(accepted_tokens)
                if text_assembler.truncated:
                    return True
                continue

            if not _is_tag(child):
                continue

            if id(child) in dropped_node_ids:
                continue

            tag_name = _tag_name(child)
            markdown_context: _MarkdownBlockContext | None = None
            if tag_name in _MARKDOWN_BLOCK_TAGS:
                markdown_context = _MarkdownBlockContext(
                    tag_name=tag_name,
                    tokens=[],
                    markdown_index=len(markdown_blocks),
                )
                markdown_blocks.append("")
                active_blocks.append(markdown_context)
                if tag_name == "pre":
                    code_block_count += 1

            should_stop = walk(child)

            if markdown_context is not None:
                active_blocks.pop()
                block_text = " ".join(markdown_context.tokens)
                if tag_name in _HEADING_TAGS and block_text:
                    headings.append(block_text)
                markdown_blocks[markdown_context.markdown_index] = (
                    _format_markdown_block(
                        tag_name=tag_name,
                        text=block_text,
                    )
                )

            if should_stop:
                return True

        return False

    if section is not None and id(section) not in dropped_node_ids:
        walk(section)

    return _SectionContent(
        text=text_assembler.build(),
        markdown="\n\n".join(block for block in markdown_blocks if block),
        headings=tuple(headings),
        code_block_count=code_block_count,
        truncated=text_assembler.truncated,
    )


def _format_markdown_block(*, tag_name: str, text: str) -> str:
    if not text:
        return ""
    if tag_name in _HEADING_TAGS:
        level = int(tag_name[1])
        return f"{'#' * level} {text}"
    if tag_name == "pre":
        return f"```\n{text}\n```"
    return text


def _collect_dropped_node_ids(
    *,
    document: Any,
    tag_names: tuple[str, ...],
    selectors: tuple[str, ...],
) -> set[int]:
    dropped: set[int] = set()
    finder = getattr(document, "find_all", None)
    if callable(finder) and tag_names:
        try:
            for node in finder(list(tag_names)):
                dropped.add(id(node))
        except (TypeError, ValueError):
            pass

    selector_api = getattr(document, "select", None)
    if callable(selector_api) and selectors:
        try:
            for node in selector_api(",".join(selectors)):
                dropped.add(id(node))
        except (TypeError, ValueError, KeyError):
            pass

    return dropped


def _find_main_section(
    *,
    document: Any,
    selectors: tuple[str, ...],
) -> Any | None:
    selector_api = getattr(document, "select", None)
    if callable(selector_api) and selectors:
        try:
            matches = list(selector_api(",".join(selectors)))
        except (TypeError, ValueError, KeyError):
            matches = []
        else:
            for selector in selectors:
                for node in matches:
                    if not _is_tag(node):
                        continue
                    try:
                        if soupsieve.match(selector, node):
                            return node
                    except (TypeError, ValueError, KeyError):
                        continue

    return _document_body_or_root(document)


def _document_body_or_root(document: Any) -> Any:
    body = getattr(document, "body", None)
    if body is not None:
        return body
    return document


def _compute_boilerplate_token_ratio(*, text: str) -> float:
    tokens = re.findall(r"\w+", text.casefold())
    if not tokens:
        return 0.0
    boilerplate_tokens = sum(
        1 for token in tokens if token in _BOILERPLATE_TOKENS
    )
    return max(0.0, min(1.0, boilerplate_tokens / len(tokens)))


def _tag_name(element: Any) -> str:
    name = getattr(element, "name", None)
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def _is_tag(element: Any) -> bool:
    name = getattr(element, "name", None)
    return isinstance(name, str) and bool(name.strip())


def _is_navigable_string(element: Any) -> bool:
    if _is_tag(element):
        return False
    # BeautifulSoup NavigableString and plain text nodes.
    return isinstance(element, str) or type(element).__name__ in {
        "NavigableString",
        "Comment",
        "CData",
        "Script",
        "Stylesheet",
        "TemplateString",
    }
