"""HTML and XML parsing with BeautifulSoup and an HTML stdlib fallback."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from html.parser import HTMLParser as StdlibHTMLParser
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

from logger.project_logger import ProjectLogger

bs4: ModuleType | None
try:
    bs4 = import_module("bs4")
except ImportError:
    bs4 = None

if TYPE_CHECKING:
    from config.collection.discovery import HtmlParserSettings


_XML_DECLARATION_RE = re.compile(
    r"\A\s*﻿?<\?xml[\s>]",
    re.IGNORECASE,
)

_XML_ROOT_RE = re.compile(
    r"""
    \A
    \s*
    ﻿?
    (?:<\?xml.*?\?>\s*)?
    (?:<!--.*?-->\s*)*
    (?:<!DOCTYPE[^>]*>\s*)*
    <
    (?:
        [A-Za-z_][A-Za-z0-9_.-]*:
    )?
    (?:
        rss
        | feed
        | rdf
        | urlset
        | sitemapindex
        | opml
        | svg
    )
    (?:\s|/?>)
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

_XML_ENCODING_RE = re.compile(
    rb"""
    <\?xml
    [^>]*?
    \bencoding
    \s*=\s*
    ["']
    ([A-Za-z0-9_.:-]+)
    ["']
    """,
    re.IGNORECASE | re.VERBOSE,
)

_HTML_META_CHARSET_RE = re.compile(
    rb"""
    \bcharset
    \s*=\s*
    ["']?
    \s*
    ([A-Za-z0-9_.:-]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DEFAULT_XML_PARSER_CANDIDATES = (
    "lxml-xml",
    "xml",
)


def element_attribute(
    *,
    element: Any,
    name: str,
) -> object:
    """Return one parser-neutral element attribute value."""

    getter = getattr(element, "get", None)

    if callable(getter):
        try:
            return getter(name)
        except (AttributeError, TypeError, ValueError):
            pass

    attrs = getattr(element, "attrs", None)
    if isinstance(attrs, dict):
        return attrs.get(name)

    attributes = getattr(element, "attributes", None)
    if isinstance(attributes, dict):
        return attributes.get(name)

    return None


def element_string_attribute(
    *,
    element: Any,
    name: str,
) -> str | None:
    """Return one stripped non-empty string attribute."""

    value = element_attribute(element=element, name=name)
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    return normalized or None


def element_string_attributes(
    *,
    element: Any,
    names: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Return string attributes in their declared order."""

    references: list[tuple[str, str]] = []

    for name in names:
        value = element_string_attribute(element=element, name=name)
        if value is not None:
            references.append((name, value))

    return tuple(references)


def element_tag_name(*, element: Any) -> str:
    """Return a normalized parser-neutral tag name."""

    name = getattr(element, "name", None)
    if not isinstance(name, str):
        name = getattr(element, "tag", None)
    if not isinstance(name, str):
        return ""
    return name.strip().casefold()


def element_visible_text(*, element: Any) -> str:
    """Return normalized visible element text."""

    getter = getattr(element, "get_text", None)
    if not callable(getter):
        return ""
    try:
        value = getter(" ", strip=True)
    except (AttributeError, TypeError, ValueError):
        return ""
    return value if isinstance(value, str) else ""


def element_raw_text(*, element: Any) -> str:
    """Return raw script-like element text without visible-text joining."""

    direct = getattr(element, "string", None)
    if isinstance(direct, str):
        return direct

    getter = getattr(element, "get_text", None)
    if not callable(getter):
        return ""
    try:
        value = getter()
    except (AttributeError, TypeError, ValueError):
        return ""
    return value if isinstance(value, str) else ""


def element_parent(*, element: Any) -> Any | None:
    """Return an element parent when the parser exposes one."""

    return getattr(element, "parent", None)


def element_has_attribute(
    *,
    element: Any,
    name: str,
) -> bool:
    """Return whether an attribute exists, including valueless attributes."""

    has_attr = getattr(element, "has_attr", None)
    if callable(has_attr):
        try:
            return bool(has_attr(name))
        except (AttributeError, TypeError, ValueError):
            pass

    for mapping_name in ("attrs", "attributes"):
        attributes = getattr(element, mapping_name, None)
        if isinstance(attributes, dict):
            return name in attributes

    return False


_VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


class HtmlParser:
    """Parse HTML or XML into a navigable document structure."""

    def __init__(
        self,
        settings: HtmlParserSettings,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._logger = logger

        # Preserve the existing attribute name because tests or composition
        # code may inspect it.
        self._parser_candidates = self._resolve_parser_candidates()
        self._xml_parser_candidates = self._resolve_xml_parser_candidates()

    def parse(
        self,
        *,
        body: bytes,
        encoding: str | None,
    ) -> Any:
        """Return a parsed document for the supplied markup payload."""

        text = _decode_markup(
            body=body,
            encoding=encoding,
        )

        if _looks_like_xml(text):
            return self._parse_xml(text)

        document = self._parse_html(text)

        if document is not None:
            return document

        return self._parse_html_with_stdlib(text)

    def _parse_html(
        self,
        text: str,
    ) -> Any | None:
        if bs4 is None or not self._parser_candidates:
            return None

        for candidate in self._parser_candidates:
            try:
                # BeautifulSoup emits this warning when XML was not detected
                # by our lightweight prefix inspection. Convert it into an
                # exception so the payload can be reparsed correctly as XML.
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "error",
                        category=bs4.XMLParsedAsHTMLWarning,
                    )
                    document = bs4.BeautifulSoup(
                        text,
                        features=candidate,
                    )

            except bs4.XMLParsedAsHTMLWarning:
                self._logger.debug(
                    "xml_document_detected_by_html_parser",
                    parser=f"beautifulsoup:{candidate}",
                )
                return self._parse_xml(text)

            except Exception as exc:
                self._logger.warning(
                    "html_document_parse_candidate_failed",
                    parser=f"beautifulsoup:{candidate}",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                continue

            self._logger.debug(
                "html_document_parsed",
                parser=f"beautifulsoup:{candidate}",
            )
            return document

        return None

    def _parse_xml(
        self,
        text: str,
    ) -> Any:
        if bs4 is None:
            raise RuntimeError(
                "XML document detected, but beautifulsoup4 is not installed"
            )

        for candidate in self._xml_parser_candidates:
            try:
                document = bs4.BeautifulSoup(
                    text,
                    features=candidate,
                )
            except Exception as exc:
                self._logger.warning(
                    "xml_document_parse_candidate_failed",
                    parser=f"beautifulsoup:{candidate}",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                continue

            self._logger.debug(
                "html_document_parsed",
                parser=f"beautifulsoup:{candidate}",
            )
            return document

        # The existing stdlib fallback uses html.parser. It is suitable for
        # degraded HTML tag discovery, but it is not a reliable XML parser.
        # Failing here prevents silently losing RSS, Atom, SVG, or sitemap
        # structure.
        raise RuntimeError(
            "XML document detected, but no XML parser is available; "
            "install lxml"
        )

    def _parse_html_with_stdlib(
        self,
        text: str,
    ) -> HtmlDocument:
        if not self._settings.allow_stdlib_fallback:
            if bs4 is None:
                raise RuntimeError(
                    "beautifulsoup4 is required to parse HTML documents"
                )

            raise RuntimeError(
                "none of the configured BeautifulSoup HTML parsers "
                "could parse the document"
            )

        parser = StdlibHtmlDocumentParser()

        try:
            parser.feed(text)
            parser.close()
        except Exception as exc:
            raise RuntimeError(
                "the stdlib HTML fallback could not parse the document"
            ) from exc

        self._logger.debug(
            "html_document_parsed",
            parser="stdlib_fallback",
        )

        return HtmlDocument(tuple(parser.elements))

    def _resolve_parser_candidates(self) -> tuple[str, ...]:
        """Resolve usable configured HTML parser candidates."""

        if bs4 is None or not self._settings.prefer_beautiful_soup:
            return ()

        configured = self._settings.parser_candidates or (
            self._settings.parser,
        )

        return self._resolve_beautiful_soup_candidates(
            candidates=tuple(configured),
            report_missing=True,
            parser_kind="html",
        )

    def _resolve_xml_parser_candidates(self) -> tuple[str, ...]:
        """Resolve installed BeautifulSoup XML parser candidates."""

        if bs4 is None or not self._settings.prefer_beautiful_soup:
            return ()

        configured_html_candidates = self._settings.parser_candidates or (
            self._settings.parser,
        )

        xml_candidates: list[str] = []

        for candidate in configured_html_candidates:
            normalized = str(candidate).strip().lower()

            if normalized in {"xml", "lxml-xml"}:
                xml_candidates.append(normalized)
            elif normalized == "lxml":
                # BeautifulSoup's "lxml" feature selects the HTML builder.
                # The corresponding XML feature is "lxml-xml".
                xml_candidates.append("lxml-xml")

        xml_candidates.extend(_DEFAULT_XML_PARSER_CANDIDATES)

        return self._resolve_beautiful_soup_candidates(
            candidates=_unique_strings(xml_candidates),
            report_missing=False,
            parser_kind="xml",
        )

    def _resolve_beautiful_soup_candidates(
        self,
        *,
        candidates: tuple[str, ...],
        report_missing: bool,
        parser_kind: str,
    ) -> tuple[str, ...]:
        if bs4 is None:
            return ()

        resolved: list[str] = []

        for candidate in _unique_strings(candidates):
            try:
                bs4.BeautifulSoup(
                    "",
                    features=candidate,
                )
            except Exception as exc:
                if report_missing:
                    self._logger.warning(
                        "html_parser_dependency_missing",
                        parser=candidate,
                        parser_kind=parser_kind,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        fallback_candidates=[
                            fallback
                            for fallback in candidates
                            if fallback != candidate
                        ],
                    )
                continue

            resolved.append(candidate)

        return tuple(resolved)


@dataclass(frozen=True, slots=True)
class HtmlElement:
    """Simple element wrapper used by the stdlib HTML fallback."""

    tag: str
    attributes: dict[str, object]
    parent: HtmlElement | None = None

    def get(
        self,
        name: str,
        default: Any = None,
    ) -> Any:
        """Return an element attribute."""

        return self.attributes.get(name, default)


class HtmlDocument:
    """Minimal document abstraction compatible with basic extractors."""

    def __init__(
        self,
        elements: tuple[HtmlElement, ...],
    ) -> None:
        self._elements = elements

    @property
    def descendants(self) -> tuple[HtmlElement, ...]:
        """Return fallback elements in document order."""

        return self._elements

    def find_all(
        self,
        tag_name: str,
    ) -> tuple[HtmlElement, ...]:
        """Return all fallback elements matching the requested tag."""

        normalized_tag_name = tag_name.casefold()
        return tuple(
            element
            for element in self._elements
            if element.tag.casefold() == normalized_tag_name
        )


class StdlibHtmlDocumentParser(StdlibHTMLParser):
    """Collect start tags and parent relationships for fallback indexing."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[HtmlElement] = []
        self._open_elements: list[HtmlElement] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Record one opening HTML tag."""

        normalized_tag = tag.casefold()
        element = HtmlElement(
            tag=normalized_tag,
            attributes=_fallback_attributes(attrs),
            parent=self._open_elements[-1] if self._open_elements else None,
        )
        self.elements.append(element)

        if normalized_tag not in _VOID_ELEMENTS:
            self._open_elements.append(element)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Record a self-closing HTML tag once."""

        normalized_tag = tag.casefold()
        self.elements.append(
            HtmlElement(
                tag=normalized_tag,
                attributes=_fallback_attributes(attrs),
                parent=(
                    self._open_elements[-1] if self._open_elements else None
                ),
            )
        )

    def handle_endtag(self, tag: str) -> None:
        """Close the nearest matching open element."""

        normalized_tag = tag.casefold()
        for index in range(len(self._open_elements) - 1, -1, -1):
            if self._open_elements[index].tag == normalized_tag:
                del self._open_elements[index:]
                return


def _fallback_attributes(
    attrs: list[tuple[str, str | None]],
) -> dict[str, object]:
    attributes: dict[str, object] = {}

    for key, value in attrs:
        normalized_key = key.casefold()
        if value is None:
            attributes[normalized_key] = None
        elif normalized_key == "rel":
            attributes[normalized_key] = tuple(
                part
                for part in (item.strip() for item in value.split())
                if part
            )
        else:
            attributes[normalized_key] = value

    return attributes


def _decode_markup(
    *,
    body: bytes,
    encoding: str | None,
) -> str:
    """Decode markup without silently discarding invalid bytes."""

    for candidate in _candidate_encodings(
        body=body,
        encoding=encoding,
    ):
        try:
            return body.decode(
                candidate,
                errors="strict",
            )
        except (LookupError, UnicodeError):
            continue

    return body.decode(
        "utf-8",
        errors="replace",
    )


def _candidate_encodings(
    *,
    body: bytes,
    encoding: str | None,
) -> tuple[str, ...]:
    candidates: list[str] = []

    # Check UTF-32 before UTF-16 because the UTF-32 little-endian BOM starts
    # with the same two bytes as the UTF-16 little-endian BOM.
    if body.startswith(
        (
            b"\x00\x00\xfe\xff",
            b"\xff\xfe\x00\x00",
        )
    ):
        candidates.append("utf-32")
    elif body.startswith(
        (
            b"\xff\xfe",
            b"\xfe\xff",
        )
    ):
        candidates.append("utf-16")
    elif body.startswith(b"\xef\xbb\xbf"):
        candidates.append("utf-8-sig")

    if encoding is not None:
        normalized_encoding = encoding.strip()

        if normalized_encoding:
            candidates.append(normalized_encoding)

    xml_encoding_match = _XML_ENCODING_RE.search(body[:1024])

    if xml_encoding_match is not None:
        declared_xml_encoding = _decode_ascii_group(
            xml_encoding_match.group(1)
        )

        if declared_xml_encoding is not None:
            candidates.append(declared_xml_encoding)

    html_charset_match = _HTML_META_CHARSET_RE.search(body[:4096])

    if html_charset_match is not None:
        declared_html_encoding = _decode_ascii_group(
            html_charset_match.group(1)
        )

        if declared_html_encoding is not None:
            candidates.append(declared_html_encoding)

    candidates.extend(
        (
            "utf-8",
            "cp1252",
            "latin-1",
        )
    )

    return _unique_strings(candidates)


def _decode_ascii_group(
    value: bytes,
) -> str | None:
    try:
        decoded = value.decode(
            "ascii",
            errors="strict",
        )
    except UnicodeError:
        return None

    normalized = decoded.strip()

    return normalized or None


def _looks_like_xml(text: str) -> bool:
    """Return whether the markup appears to be XML."""

    sample = text[:4096]

    return (
        _XML_DECLARATION_RE.search(sample) is not None
        or _XML_ROOT_RE.search(sample) is not None
    )


def _unique_strings(
    values: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    unique: list[str] = []
    encountered: set[str] = set()

    for value in values:
        normalized = str(value).strip()

        if not normalized:
            continue

        identity = normalized.casefold()

        if identity in encountered:
            continue

        encountered.add(identity)
        unique.append(normalized)

    return tuple(unique)
