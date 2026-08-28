"""Integration tests: priority calculator results drive queue pop order.

Convention under test: lower numeric priority score means higher scheduling
priority. These tests connect CrawlTaskPriorityCalculator -> HostTaskQueue
-> pop() with real scheduler signals and assert the combined behavior.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from config.collection.discovery import UrlPrioritySettings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.scheduling.priority.crawl_task_priority_calculator import (
    CrawlTaskPriorityCalculator,
)
from crawler.scheduling.queueing.host_task_queue import HostTaskQueue
from tests.support.logging import TEST_LOGGER


class _IdentityNormalizer:
    def normalize(self, host: object) -> object:
        return host


class _HostExtractor:
    def extract(self, url: str) -> str | None:
        return urlsplit(url).hostname


class _FakeHostBudgetTracker:
    def __init__(
        self,
        *,
        qualities: dict[str, float] | None = None,
        info_gains: dict[str, float] | None = None,
        seed_hosts: set[str] | None = None,
    ) -> None:
        self._qualities = dict(qualities or {})
        self._info_gains = dict(info_gains or {})
        self._seed_hosts = set(seed_hosts or ())

    def host_quality(self, host: str) -> float:
        return self._qualities.get(host, 0.5)

    def host_info_gain(self, host: str) -> float:
        return self._info_gains.get(host, 0.5)

    def is_seed_host(self, host: str) -> bool:
        return host in self._seed_hosts


def _config(**overrides: object) -> UrlPrioritySettings:
    base: dict[str, object] = dict(
        seed_priority=-5,
        discovered_priority=0,
        depth_penalty=2,
        feed_priority="medium",
        host_quality_boost_scale=12.0,
        host_noise_penalty_scale=12.0,
        info_gain_boost_scale=6.0,
        external_host_exploration_penalty=8,
        low_info_gain_penalty_enabled=True,
        low_info_gain_penalty_threshold=0.12,
        low_info_gain_penalty=5,
    )
    base.update(overrides)
    return UrlPrioritySettings(**base)


class _CalculatorQueueHarness:
    def __init__(self, config: UrlPrioritySettings) -> None:
        self._budget_tracker = _FakeHostBudgetTracker()
        self.calculator = CrawlTaskPriorityCalculator(
            config=config,
            logger=TEST_LOGGER,
            host_extractor=_HostExtractor(),
            host_budget_tracker=self._budget_tracker,
        )
        self.queue = HostTaskQueue(host_normalizer=_IdentityNormalizer())
        self._sequence = 0

    def set_quality(self, values: dict[str, float]) -> None:
        self._budget_tracker._qualities.update(values)

    def set_info_gain(self, values: dict[str, float]) -> None:
        self._budget_tracker._info_gains.update(values)

    def set_seed_hosts(self, *hosts: str) -> None:
        self._budget_tracker._seed_hosts.update(hosts)

    def priority(
        self,
        *,
        url: str,
        depth: int,
        source_type: str = "seed",
        kind: MediaKind = MediaKind.PAGE,
    ) -> int:
        return self.calculator.resolve(
            url=url,
            depth=depth,
            source_type=source_type,
            kind=kind,
        )

    def push(
        self,
        *,
        url: str,
        depth: int,
        source_type: str = "seed",
        kind: MediaKind = MediaKind.PAGE,
    ) -> int:
        priority = self.priority(
            url=url,
            depth=depth,
            source_type=source_type,
            kind=kind,
        )
        task = CrawlTask(
            url=url,
            source_name="test",
            kind=kind,
            source_type=source_type,
            depth=depth,
        )
        self.queue.push(
            host="bucket.example",
            priority=priority,
            sequence=self._sequence,
            task=task,
        )
        self._sequence += 1
        return priority

    def pop_urls(self, count: int) -> list[str]:
        return [self.queue.pop().url for _ in range(count)]


def test_seed_pops_before_discovered() -> None:
    harness = _CalculatorQueueHarness(_config())
    harness.set_seed_hosts("nasa.example", "usgs.example")

    seed = harness.push(
        url="https://nasa.example/", depth=0, source_type="seed"
    )
    discovered = harness.push(
        url="https://usgs.example/page",
        depth=0,
        source_type="discovered_link",
    )

    assert seed < discovered
    assert harness.pop_urls(2) == [
        "https://nasa.example/",
        "https://usgs.example/page",
    ]


def test_shallower_depth_pops_first() -> None:
    harness = _CalculatorQueueHarness(_config())
    harness.set_seed_hosts("deep.example")

    scores = {
        depth: harness.push(
            url=f"https://deep.example/d{depth}",
            depth=depth,
            source_type="seed",
        )
        for depth in (0, 1, 2)
    }

    assert scores[0] < scores[1] < scores[2]
    assert harness.pop_urls(3) == [
        "https://deep.example/d0",
        "https://deep.example/d1",
        "https://deep.example/d2",
    ]


def test_feed_priority_lower_score_pops_first() -> None:
    harnesses = {
        level: _CalculatorQueueHarness(_config(feed_priority=level))
        for level in ("high", "medium", "low")
    }
    for _level, harness in harnesses.items():
        harness.set_seed_hosts("feed.example")

    scores = {
        level: harness.priority(
            url=f"https://feed.example/{level}",
            depth=0,
            source_type="seed",
            kind=MediaKind.FEED,
        )
        for level, harness in harnesses.items()
    }

    assert scores["high"] < scores["medium"] < scores["low"]

    combined = _CalculatorQueueHarness(_config(feed_priority="medium"))
    combined.set_seed_hosts("feed.example")
    for level in ("high", "medium", "low"):
        combined.push(
            url=f"https://feed.example/{level}",
            depth=0,
            source_type="seed",
            kind=MediaKind.FEED,
        )

    assert combined.pop_urls(3) == [
        "https://feed.example/high",
        "https://feed.example/medium",
        "https://feed.example/low",
    ]


def test_high_quality_host_pops_first() -> None:
    harness = _CalculatorQueueHarness(_config())
    harness.set_seed_hosts("quality.example", "poor.example")
    harness.set_quality({"quality.example": 1.0, "poor.example": 0.0})

    high = harness.push(
        url="https://quality.example/", depth=0, source_type="seed"
    )
    low = harness.push(
        url="https://poor.example/", depth=0, source_type="seed"
    )

    assert high < low
    assert harness.pop_urls(2) == [
        "https://quality.example/",
        "https://poor.example/",
    ]


def test_noise_penalty_defers_noisy_host() -> None:
    harness = _CalculatorQueueHarness(_config())
    harness.set_seed_hosts("clean.example", "noisy.example")
    harness.set_quality({"clean.example": 1.0, "noisy.example": 0.0})

    clean = harness.push(
        url="https://clean.example/", depth=0, source_type="seed"
    )
    noisy = harness.push(
        url="https://noisy.example/", depth=0, source_type="seed"
    )

    assert clean < noisy
    assert harness.pop_urls(2) == [
        "https://clean.example/",
        "https://noisy.example/",
    ]


def test_external_host_exploration_penalty_defers_external_url() -> None:
    harness = _CalculatorQueueHarness(_config())
    harness.set_seed_hosts("internal.example")

    internal = harness.push(
        url="https://internal.example/page",
        depth=0,
        source_type="discovered_link",
    )
    external = harness.push(
        url="https://external.example/page",
        depth=0,
        source_type="discovered_link",
    )

    assert internal < external
    assert harness.pop_urls(2) == [
        "https://internal.example/page",
        "https://external.example/page",
    ]


def test_low_information_penalty_defers_low_info_host() -> None:
    harness = _CalculatorQueueHarness(_config())
    harness.set_seed_hosts("informative.example", "lowinfo.example")
    harness.set_info_gain({"informative.example": 0.5, "lowinfo.example": 0.1})

    informative = harness.push(
        url="https://informative.example/", depth=0, source_type="seed"
    )
    low_info = harness.push(
        url="https://lowinfo.example/", depth=0, source_type="seed"
    )

    assert informative < low_info
    assert harness.pop_urls(2) == [
        "https://informative.example/",
        "https://lowinfo.example/",
    ]


def test_depth_penalty_is_monotonic_and_queue_follows() -> None:
    harness = _CalculatorQueueHarness(_config())
    harness.set_seed_hosts("monotone.example")

    scores = [
        harness.priority(
            url=f"https://monotone.example/d{depth}",
            depth=depth,
            source_type="seed",
        )
        for depth in (0, 1, 2)
    ]

    assert scores[0] < scores[1] < scores[2]
    for depth in (0, 1, 2):
        harness.push(
            url=f"https://monotone.example/d{depth}",
            depth=depth,
            source_type="seed",
        )
    assert harness.pop_urls(3) == [
        "https://monotone.example/d0",
        "https://monotone.example/d1",
        "https://monotone.example/d2",
    ]


def test_more_noise_never_pops_earlier() -> None:
    harness = _CalculatorQueueHarness(_config())
    harness.set_seed_hosts("high.example", "mid.example", "low.example")
    harness.set_quality(
        {"high.example": 1.0, "mid.example": 0.5, "low.example": 0.0}
    )

    scores = [
        harness.priority(
            url=f"https://{host}.example/",
            depth=0,
            source_type="seed",
        )
        for host in ("high", "mid", "low")
    ]

    assert scores[0] < scores[1] < scores[2]
    for host in ("high", "mid", "low"):
        harness.push(
            url=f"https://{host}.example/",
            depth=0,
            source_type="seed",
        )
    assert harness.pop_urls(3) == [
        "https://high.example/",
        "https://mid.example/",
        "https://low.example/",
    ]


def test_more_information_gain_never_pops_later() -> None:
    harness = _CalculatorQueueHarness(_config())
    harness.set_seed_hosts(
        "lowgain.example", "midgain.example", "highgain.example"
    )
    harness.set_info_gain(
        {
            "lowgain.example": 0.2,
            "midgain.example": 0.5,
            "highgain.example": 1.0,
        }
    )

    scores = [
        harness.priority(
            url=f"https://{host}.example/",
            depth=0,
            source_type="seed",
        )
        for host in ("lowgain", "midgain", "highgain")
    ]

    assert scores[0] > scores[1] > scores[2]
    for host in ("lowgain", "midgain", "highgain"):
        harness.push(
            url=f"https://{host}.example/",
            depth=0,
            source_type="seed",
        )
    assert harness.pop_urls(3) == [
        "https://highgain.example/",
        "https://midgain.example/",
        "https://lowgain.example/",
    ]
