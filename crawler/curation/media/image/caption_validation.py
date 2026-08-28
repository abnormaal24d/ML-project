"""Detect unusable text in curated image captions."""

from __future__ import annotations

import re

_HTML_GARBAGE_MARKERS = (
    "<link ",
    "<meta ",
    "<script",
    "<style",
    "</",
    "[if ie",
    "rel=",
    "stylesheet",
    'type="text/css"',
    "bootstrap/css",
)

_BOILERPLATE_PHRASES = (
    "view list",
    "gallery grid",
    "sort by",
    "shelf order",
    "click here",
    "read more",
    "skip to",
)

_UI_TOKENS = {
    "view",
    "list",
    "gallery",
    "grid",
    "sort",
    "search",
    "menu",
    "share",
    "filter",
    "previous",
    "next",
    "go",
    "order",
    "results",
    "stylesheet",
    "script",
    "link",
    "rel",
}


def has_caption_garbage(text: str | None) -> bool:
    normalized = _normalize_caption_text(text)
    if not normalized:
        return False
    if any(marker in normalized for marker in _HTML_GARBAGE_MARKERS):
        return True

    angle_count = normalized.count("<") + normalized.count(">")
    if angle_count >= 2:
        return True

    tokens = normalized.split()
    htmlish_tokens = sum(
        token.startswith("<")
        or token.endswith(">")
        or "=" in token
        or token in {"href", "src", "rel", "class", "type"}
        for token in tokens
    )
    return (htmlish_tokens / len(tokens)) >= 0.25


def is_boilerplate_caption(text: str | None) -> bool:
    normalized = _normalize_caption_text(text)
    if not normalized:
        return False
    if any(
        re.search(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", normalized)
        for phrase in _BOILERPLATE_PHRASES
    ):
        return True

    tokens = _caption_tokens(normalized)
    ui_count = sum(token in _UI_TOKENS for token in tokens)
    ui_ratio = ui_count / len(tokens)
    if len(tokens) <= 3 and ui_count == len(tokens):
        return True
    if len(tokens) <= 5 and ui_count >= 2:
        return True
    if len(tokens) >= 4 and ui_ratio >= 0.35:
        return True
    if normalized.startswith(("credit:", "photo credit:", "image credit:")):
        return len(tokens) <= 8
    return False


def _caption_tokens(normalized: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+(?:-[^\W_]+)*", normalized))


def _normalize_caption_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).casefold().strip().split())
