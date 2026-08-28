"""Policy-flow tests for initial host feedback state."""

from __future__ import annotations

import pytest

from config.collection.discovery import (
    DiscoveryFeedbackSettings,
    SchedulingSettings,
)
from config.collection.governance import UrlFilterSettings
from crawler.extraction.hosts_extractor import HostExtractor
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.scheduling.host_control.discovery_signal_scorer import (
    DiscoverySignalScorer,
)
from crawler.scheduling.host_control.host_budget_tracker import (
    HostBudgetTracker,
)
from crawler.scheduling.host_control.host_feedback_aggregator import (
    HostFeedbackAggregator,
)
from tests.support.logging import TEST_LOGGER


def test_custom_initial_feedback_policy_is_preserved() -> None:
    aggregator = HostFeedbackAggregator(
        max_hosts=None,
        default_info_gain=0.2,
        default_host_quality=0.25,
        host_extractor=HostExtractor(logger=TEST_LOGGER),
        host_normalizer=HostNormalizer(),
    )

    seed_feedback = aggregator.get_or_create("seed.example")
    unknown_feedback = aggregator.get_or_create("unknown.example")

    assert seed_feedback.info_gain_ewma == 0.2
    assert seed_feedback.quality_ewma == 0.25
    assert unknown_feedback.info_gain_ewma == 0.2
    assert unknown_feedback.quality_ewma == 0.25


def test_tracker_uses_policy_aware_runtime_fallbacks() -> None:
    feedback_settings = DiscoveryFeedbackSettings(
        default_info_gain=0.2,
        default_host_quality=0.25,
        seed_host_quality=0.9,
    )
    aggregator = HostFeedbackAggregator(
        max_hosts=None,
        default_info_gain=feedback_settings.default_info_gain,
        default_host_quality=feedback_settings.default_host_quality,
        host_extractor=HostExtractor(logger=TEST_LOGGER),
        host_normalizer=HostNormalizer(),
    )
    tracker = HostBudgetTracker(
        settings=SchedulingSettings(discovery_feedback=feedback_settings),
        url_filter_settings=UrlFilterSettings(),
        logger=TEST_LOGGER,
        signal_scorer=DiscoverySignalScorer(feedback_settings),
        feedback_aggregator=aggregator,
        host_extractor=HostExtractor(logger=TEST_LOGGER),
        host_normalizer=HostNormalizer(),
        seed_urls=("https://seed.example/start",),
    )

    assert tracker.host_quality("seed.example") == pytest.approx(0.25)
    assert tracker.host_quality("unknown.example") == pytest.approx(0.25)
    assert tracker.host_info_gain("unknown.example") == pytest.approx(0.2)

    aggregator.get_or_create("seed.example").quality_ewma = float("nan")
    aggregator.get_or_create("unknown.example").info_gain_ewma = float("nan")

    assert tracker.host_quality("seed.example") == pytest.approx(0.25)
    assert tracker.host_info_gain("unknown.example") == pytest.approx(0.2)
