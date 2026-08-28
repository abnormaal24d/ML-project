"""Asset-specific HTML element context and reference helpers."""

from __future__ import annotations

from typing import Any

from crawler.extraction.assets.candidate.asset_extraction_records import (
    as_optional_float,
    as_optional_int,
    clean_string,
)
from crawler.extraction.html.html_parser import (
    element_attribute,
    element_parent,
    element_string_attribute,
    element_tag_name,
    element_visible_text,
)

_SRCSET_ATTRIBUTES: tuple[str, ...] = ("srcset", "data-srcset")


def element_dom_path(*, element: Any) -> str | None:
    parts: list[str] = []
    current = element

    while current is not None and len(parts) < 8:
        name = element_tag_name(element=current)
        if not name:
            break
        parts.append(name)
        current = element_parent(element=current)

    if not parts:
        return None

    return " > ".join(reversed(parts))


def context_metadata(context: dict[str, object]) -> dict[str, object]:
    metadata = context.get("metadata")
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def srcset_references(
    *,
    element: Any,
) -> tuple[tuple[str, str], ...]:
    """Return srcset candidates while preserving existing parse semantics."""

    references: list[tuple[str, str]] = []

    for attribute in _SRCSET_ATTRIBUTES:
        value = element_string_attribute(element=element, name=attribute)
        if value is None:
            continue
        for candidate in _parse_srcset_urls(value):
            references.append((attribute, candidate))

    return tuple(references)


def build_element_asset_context(
    *,
    element: Any,
    tag_name: str,
    parent_text_metadata: dict[str, object],
) -> dict[str, object]:
    alt_text = clean_string(element_attribute(element=element, name="alt"))

    # Caption is intentionally distinct from alt text. This prevents the
    # same text from receiving both the alt and caption quality bonuses.
    caption_text: str | None = None

    if tag_name == "img":
        try:
            figure = element.find_parent("figure")
        except (AttributeError, TypeError, ValueError):
            figure = None

        if figure is None:
            parent = element_parent(element=element)
            if element_tag_name(element=parent) == "figure":
                figure = parent

        if figure is not None:
            try:
                caption = figure.find("figcaption")
            except (AttributeError, TypeError, ValueError):
                caption = None

            if caption is not None:
                caption_text = clean_string(
                    element_visible_text(element=caption)
                    or getattr(caption, "string", None)
                )

    surrounding_parts: list[str] = []
    parent = element_parent(element=element)

    if parent is not None:
        parent_text = clean_string(element_visible_text(element=parent))
        if parent_text:
            surrounding_parts.append(parent_text)

    for navigation_method_name in ("find_previous", "find_next"):
        navigation_method = getattr(element, navigation_method_name, None)
        if not callable(navigation_method):
            continue

        try:
            nearby_node = navigation_method(string=True)
        except (AttributeError, TypeError, ValueError):
            continue

        nearby_text = clean_string(
            nearby_node
            if isinstance(nearby_node, str)
            else getattr(nearby_node, "string", None)
        )
        if nearby_text and nearby_text not in surrounding_parts:
            surrounding_parts.append(nearby_text)

    surrounding_text = " ".join(surrounding_parts)[:500] or None
    metadata = dict(parent_text_metadata)

    return {
        "source_tag": tag_name,
        "alt_text": alt_text,
        "caption_text": caption_text,
        "surrounding_text": surrounding_text,
        "mime_hint": clean_string(
            element_attribute(element=element, name="type")
        ),
        "width": as_optional_int(
            element_attribute(element=element, name="width")
        ),
        "height": as_optional_int(
            element_attribute(element=element, name="height")
        ),
        "duration_seconds": as_optional_float(
            element_attribute(element=element, name="duration")
        )
        or as_optional_float(
            element_attribute(element=element, name="data-duration")
        ),
        "metadata": {
            **metadata,
            "html_dom_path": element_dom_path(element=element),
        },
    }


def _parse_srcset_urls(value: str) -> tuple[str, ...]:
    urls: list[str] = []

    for part in value.split(","):
        tokens = part.strip().split()
        if not tokens:
            continue
        candidate = tokens[0].strip()
        if candidate:
            urls.append(candidate)

    return tuple(urls)
