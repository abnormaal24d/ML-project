from __future__ import annotations

import pytest

from config.collection.discovery import DiscoveryFeedbackSettings
from crawler.scheduling.host_control.discovery_signal_scorer import (
    DiscoverySignalScorer,
)


def test_scorer_respects_configured_default_info_gain_below_half() -> None:
    scorer = DiscoverySignalScorer(
        DiscoveryFeedbackSettings(default_info_gain=0.2)
    )

    assert scorer.score().info_gain == pytest.approx(0.2)


def test_scorer_uses_configured_ewma_alpha() -> None:
    scorer = DiscoverySignalScorer(DiscoveryFeedbackSettings(ewma_alpha=0.8))

    assert scorer.ewma(0.2, 1.0) == pytest.approx(0.84)


def test_scorer_uses_configured_outcome_info_gains() -> None:
    scorer = DiscoverySignalScorer(
        DiscoveryFeedbackSettings(
            failed_info_gain=0.77,
            dropped_info_gain=0.66,
            cancelled_info_gain=0.55,
        )
    )

    assert scorer.outcome_info_gain(outcome="failed") == pytest.approx(0.77)
    assert scorer.outcome_info_gain(outcome="dropped") == pytest.approx(0.66)
    assert scorer.outcome_info_gain(outcome="cancelled") == pytest.approx(0.55)
