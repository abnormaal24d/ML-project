"""Calculate crawl task discovery signal scores from outcome counters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from crawler.numeric import coerce_finite_float
from crawler.scheduling.scheduling_value_parser import coerce_float, coerce_int

if TYPE_CHECKING:
    from config.collection.discovery import DiscoveryFeedbackSettings


@dataclass(frozen=True, slots=True)
class DiscoverySignalScores:
    """Normalized score values used to build discovery feedback signals."""

    info_gain: float
    novelty_gain: float
    acceptance_ratio: float
    duplicate_pressure: float
    truncation_pressure: float
    rejection_pressure: float


class DiscoverySignalScorer:
    """Calculate discovery feedback scores from crawl outcome counters."""

    _POSITIVE_OUTCOMES = frozenset(
        {
            "accepted",
            "complete",
            "completed",
            "done",
            "fetched",
            "processed",
            "success",
        }
    )
    _FAILED_OUTCOMES = frozenset(
        {
            "cancelled_error",
            "error",
            "failed",
            "failure",
            "fetch_error",
            "http_error",
            "parse_error",
            "rejected",
            "timeout",
        }
    )
    _CANCELLED_OUTCOMES = frozenset({"cancelled", "interrupted"})
    _DEFERRED_OUTCOMES = frozenset({"deferred"})
    _DROPPED_OUTCOMES = frozenset(
        {
            "dropped",
            "duplicate",
            "filtered",
            "ignored",
            "skipped",
            "truncated",
        }
    )

    def __init__(self, settings: DiscoveryFeedbackSettings) -> None:
        self._settings = settings

    def score(
        self,
        *,
        discovered: int | None = None,
        scheduled: int | None = None,
        filtered: int | None = None,
        rejected: int | None = None,
        truncated: int | None = None,
        duplicates: int | None = None,
        relevance_score: float | None = None,
        quality_score: float | None = None,
    ) -> DiscoverySignalScores:
        """Score discovery counters and optional quality hints."""
        discovered_count = _non_negative_int(discovered)
        scheduled_count = _non_negative_int(scheduled)
        filtered_count = _non_negative_int(filtered)
        rejected_count = _non_negative_int(rejected)
        truncated_count = _non_negative_int(truncated)
        duplicate_count = _non_negative_int(duplicates)

        observed_total = max(
            1,
            discovered_count,
            scheduled_count + filtered_count + rejected_count,
        )

        acceptance_ratio = _clamp(scheduled_count / observed_total)
        duplicate_pressure = _clamp(duplicate_count / observed_total)
        truncation_pressure = _clamp(truncated_count / observed_total)

        penalized_rejections = rejected_count + (
            filtered_count * self._settings.filtered_rejection_penalty_scale
        )
        rejection_pressure = _clamp(penalized_rejections / observed_total)

        novelty_denominator = max(1, scheduled_count, discovered_count)
        novelty_gain = _clamp(
            max(0, scheduled_count - duplicate_count) / novelty_denominator
        )

        if _has_signal_inputs(
            discovered=discovered,
            scheduled=scheduled,
            filtered=filtered,
            rejected=rejected,
            truncated=truncated,
            duplicates=duplicates,
            relevance_score=relevance_score,
            quality_score=quality_score,
        ):
            base_info_gain = self._weighted_success_info_gain(
                acceptance_ratio=acceptance_ratio,
                novelty_gain=novelty_gain,
                truncation_pressure=truncation_pressure,
                rejection_pressure=rejection_pressure,
            )
        else:
            base_info_gain = self._settings.default_info_gain

        info_gain = self._blend_quality_hints(
            base_info_gain,
            relevance_score=relevance_score,
            quality_score=quality_score,
            blend_weight=self._settings.quality_hint_blend_weight,
        )

        return DiscoverySignalScores(
            info_gain=_clamp(info_gain),
            novelty_gain=novelty_gain,
            acceptance_ratio=acceptance_ratio,
            duplicate_pressure=duplicate_pressure,
            truncation_pressure=truncation_pressure,
            rejection_pressure=rejection_pressure,
        )

    def outcome_info_gain(
        self,
        *,
        outcome: str,
        discovered: int | None = None,
        scheduled: int | None = None,
        filtered: int | None = None,
        rejected: int | None = None,
        truncated: int | None = None,
        duplicates: int | None = None,
        relevance_score: float | None = None,
        quality_score: float | None = None,
    ) -> float:
        """Score a task outcome as a host-budget information gain."""
        normalized_outcome = (outcome or "").strip().lower()

        if normalized_outcome in self._FAILED_OUTCOMES:
            return _clamp(self._settings.failed_info_gain)

        if normalized_outcome in self._DROPPED_OUTCOMES:
            return _clamp(self._settings.dropped_info_gain)

        if normalized_outcome in self._CANCELLED_OUTCOMES:
            return _clamp(self._settings.cancelled_info_gain)

        scores = self.score(
            discovered=discovered,
            scheduled=scheduled,
            filtered=filtered,
            rejected=rejected,
            truncated=truncated,
            duplicates=duplicates,
            relevance_score=relevance_score,
            quality_score=quality_score,
        )

        if normalized_outcome in self._POSITIVE_OUTCOMES:
            return scores.info_gain

        if normalized_outcome in self._DEFERRED_OUTCOMES:
            return _clamp(min(scores.info_gain * 0.45, 0.18))

        default_info_gain = self._settings.default_info_gain
        return _clamp((scores.info_gain + default_info_gain) / 2.0)

    def _weighted_success_info_gain(
        self,
        *,
        acceptance_ratio: float,
        novelty_gain: float,
        truncation_pressure: float,
        rejection_pressure: float,
    ) -> float:
        acceptance_weight = self._settings.success_weight_acceptance
        novelty_weight = self._settings.success_weight_novelty
        retained_after_truncation_weight = (
            self._settings.success_weight_retained_after_truncation
        )
        low_rejection_weight = self._settings.success_weight_low_rejection

        total_weight = (
            acceptance_weight
            + novelty_weight
            + retained_after_truncation_weight
            + low_rejection_weight
        )

        return _clamp(
            (
                (acceptance_ratio * acceptance_weight)
                + (novelty_gain * novelty_weight)
                + (
                    (1.0 - truncation_pressure)
                    * retained_after_truncation_weight
                )
                + ((1.0 - rejection_pressure) * low_rejection_weight)
            )
            / total_weight
        )

    @staticmethod
    def _blend_quality_hints(
        base: float,
        *,
        relevance_score: float | None,
        quality_score: float | None,
        blend_weight: float,
    ) -> float:
        hints = [
            _clamp(hint)
            for hint in (relevance_score, quality_score)
            if hint is not None
        ]
        if not hints:
            return _clamp(base)

        bounded_weight = _clamp(blend_weight)
        blended_hint = sum(hints) / len(hints)

        return _clamp(
            ((1.0 - bounded_weight) * _clamp(base))
            + (bounded_weight * blended_hint)
        )

    def calculate_info_gain(
        self,
        *,
        outcome: str,
        fields: dict[str, object] | None,
    ) -> float:
        """Calculate bounded information gain for a task outcome."""
        info_gain = self.outcome_info_gain(
            outcome=outcome,
            discovered=field_as_int(fields, "discovered"),
            scheduled=field_as_int(fields, "scheduled"),
            filtered=field_as_int(fields, "filtered"),
            rejected=field_as_int(fields, "rejected"),
            truncated=field_as_int(fields, "truncated"),
            duplicates=field_as_int(fields, "duplicates"),
            relevance_score=field_as_float(fields, "relevance_score"),
            quality_score=field_as_float(fields, "quality_score"),
        )
        return coerce_score(
            info_gain,
            default=self._settings.default_info_gain,
        )

    def blend_quality(
        self,
        *,
        relevance: float,
        info_gain: float,
        topic_score: float,
    ) -> float:
        """Blend task relevance, information gain and topicality."""
        return coerce_score(
            (0.4 * relevance) + (0.4 * info_gain) + (0.2 * topic_score),
            default=self._settings.default_host_quality,
        )

    def ewma(self, previous: float, current: float) -> float:
        """Calculate the configured EWMA value."""

        safe_previous = coerce_score(
            previous,
            default=self._settings.default_info_gain,
        )
        safe_current = coerce_score(current, default=safe_previous)
        return coerce_score(
            (self._settings.ewma_alpha * safe_current)
            + ((1.0 - self._settings.ewma_alpha) * safe_previous),
            default=safe_previous,
        )

    @staticmethod
    def default_relevance_for_outcome(outcome: str) -> float:
        """Return the fallback relevance signal for a task outcome."""
        normalized_outcome = (outcome or "").strip().lower()

        if normalized_outcome == "success":
            return 0.5
        if normalized_outcome == "deferred":
            return 0.2
        if normalized_outcome == "cancelled":
            return 0.15
        if normalized_outcome == "dropped":
            return 0.05
        if normalized_outcome in {"failed", "error", "rejected"}:
            return 0.05

        return 0.2


def _has_signal_inputs(
    *,
    discovered: int | None,
    scheduled: int | None,
    filtered: int | None,
    rejected: int | None,
    truncated: int | None,
    duplicates: int | None,
    relevance_score: float | None,
    quality_score: float | None,
) -> bool:
    return any(
        value is not None
        for value in (
            discovered,
            scheduled,
            filtered,
            rejected,
            truncated,
            duplicates,
            relevance_score,
            quality_score,
        )
    )


def _non_negative_int(value: int | None) -> int:
    parsed = coerce_int(value)
    return 0 if parsed is None else max(0, parsed)


def _clamp(value: float) -> float:
    return coerce_finite_float(
        value,
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    )


def coerce_score(value: object | None, *, default: float) -> float:
    """Coerce a score into the inclusive 0..1 range."""
    numeric_value = coerce_float(value, allow_bool=True)
    if numeric_value is None:
        return float(default)

    if numeric_value < 0.0:
        return 0.0
    if numeric_value > 1.0:
        return 1.0

    return numeric_value


def field_as_int(fields: dict[str, object] | None, key: str) -> int | None:
    """Read an optional integer field."""
    if fields is None:
        return None
    return coerce_int(fields.get(key), allow_bool=True)


def field_as_float(fields: dict[str, object] | None, key: str) -> float | None:
    """Read an optional float field."""
    if fields is None:
        return None
    return coerce_float(fields.get(key), allow_bool=True)


def field_as_str(fields: dict[str, object] | None, key: str) -> str | None:
    """Read a stripped optional string field."""
    if fields is None:
        return None

    value = fields.get(key)
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    return text
