"""Quality scoring for discovered asset candidates."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlsplit

CAPTION_QUALITY_BONUS = 30.0
ALT_TEXT_QUALITY_BONUS = 20.0
SURROUNDING_TEXT_QUALITY_BONUS = 15.0
TIMED_MEDIA_DURATION_QUALITY_BONUS = 10.0
MIME_HINT_QUALITY_BONUS = 5.0
DECORATIVE_ASSET_QUALITY_PENALTY = 50.0
DECORATIVE_ASSET_TOKENS = frozenset(
    {"logo", "icon", "sprite", "tracking", "pixel", "spacer", "beacon"}
)


def score_asset_candidate(candidate: Any) -> float:
    """Return a scheduling score for an extracted asset candidate."""

    score = 0.0

    if getattr(candidate, "caption_text", None):
        score += CAPTION_QUALITY_BONUS
    if getattr(candidate, "alt_text", None):
        score += ALT_TEXT_QUALITY_BONUS
    if getattr(candidate, "surrounding_text", None):
        score += SURROUNDING_TEXT_QUALITY_BONUS
    if getattr(candidate, "kind", None) in {"audio", "video"} and getattr(
        candidate, "duration_seconds", None
    ):
        score += TIMED_MEDIA_DURATION_QUALITY_BONUS
    if getattr(candidate, "mime_hint", None):
        score += MIME_HINT_QUALITY_BONUS

    if _has_decorative_asset_token(url=str(getattr(candidate, "url", ""))):
        score -= DECORATIVE_ASSET_QUALITY_PENALTY

    return score


def bounded_priority_from_quality(score: float) -> int:
    """Convert quality score into a small scheduler priority boost."""

    return int(round(max(-50.0, min(80.0, float(score)))))


def _has_decorative_asset_token(*, url: str) -> bool:
    parsed = urlsplit(url)
    tokens: set[str] = set()
    for part in parsed.path.lower().replace(".", "/").split("/"):
        tokens.update(token for token in part.replace("_", "-").split("-"))
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        tokens.add(key.strip().lower())
        tokens.add(value.strip().lower())
    return bool(tokens.intersection(DECORATIVE_ASSET_TOKENS))
