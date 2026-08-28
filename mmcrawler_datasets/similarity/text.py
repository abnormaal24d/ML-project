"""Shared text shingle similarity helpers."""

from __future__ import annotations

from config.environment.default_values import DEFAULT_SHINGLE_WIDTH


def to_token_shingles(
    *,
    text: str,
    width: int = DEFAULT_SHINGLE_WIDTH,
) -> set[str]:
    """Return the set of token shingles for ``text``."""

    tokens = text.lower().split()
    if len(tokens) <= width:
        return {" ".join(tokens)} if tokens else set()
    return {
        " ".join(tokens[index : index + width])
        for index in range(0, len(tokens) - width + 1)
    }


def build_text_shingle_profile(
    text: str,
    *,
    width: int = DEFAULT_SHINGLE_WIDTH,
) -> frozenset[str]:
    """Return the canonical public token-shingle profile."""

    return frozenset(to_token_shingles(text=text, width=max(1, width)))


def text_shingle_similarity(
    left: set[str] | frozenset[str] | tuple[str, ...],
    right: set[str] | frozenset[str] | tuple[str, ...],
) -> float:
    """Return canonical Jaccard similarity for two public profiles."""

    left_set = frozenset(str(value) for value in left)
    right_set = frozenset(str(value) for value in right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


__all__ = [
    "build_text_shingle_profile",
    "text_shingle_similarity",
    "to_token_shingles",
]
