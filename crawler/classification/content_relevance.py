"""Score content relevance from classification metadata.

Uses ContentCategory enum values (via .value or direct comparison after normalization)
to avoid magic strings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.classification.content_category_detector import ContentCategory
from crawler.classification.media_kind import MediaKind

if TYPE_CHECKING:
    from config.settings.classification import ContentRelevanceScorerSettings


class ContentRelevanceScorer:
    """
    Score content relevance from kind, semantic category, and payload size.
    """

    def __init__(self, *, settings: ContentRelevanceScorerSettings) -> None:
        self._settings = settings

    def score(
        self,
        *,
        kind: MediaKind,
        category: ContentCategory | None,
        byte_size: int,
    ) -> float | None:
        """Compute a relevance score in [0, 1]."""
        if not self._settings.enabled:
            return None

        score = self._settings.base_score
        if kind is MediaKind.PAGE:
            score += self._settings.page_bonus

        if category is ContentCategory.DOCUMENTATION:
            score += self._settings.documentation_bonus
        elif category is ContentCategory.ACADEMIC:
            score += self._settings.academic_bonus
        elif category is ContentCategory.MEDIA:
            score += self._settings.media_bonus
        elif category is ContentCategory.BOILERPLATE:
            score -= self._settings.boilerplate_penalty

        if byte_size < self._settings.small_payload_threshold_bytes:
            score -= self._settings.small_payload_penalty
        if byte_size > self._settings.large_payload_threshold_bytes:
            score += self._settings.large_payload_bonus

        return round(self._clamp(score), self._settings.rounding_digits)

    @staticmethod
    def _clamp(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
