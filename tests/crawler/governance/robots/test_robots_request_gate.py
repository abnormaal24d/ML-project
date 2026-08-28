"""Behaviour tests for RobotsRequestGate access enforcement."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from config.collection.governance import RobotsSettings
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.robots.robots_check_result import (
    RobotsCheckResult,
    RobotsConfidence,
    RobotsDecision,
)
from crawler.governance.robots.robots_request_gate import RobotsRequestGate
from crawler.runtime.metrics.collection_metrics import CollectionMetrics
from tests.support.logging import TEST_LOGGER


class _FakeChecker:
    """Scripted robots checker double."""

    def __init__(
        self,
        *,
        result: RobotsCheckResult | None = None,
        snapshot: Any | None = None,
    ) -> None:
        self._result = result
        self._snapshot = snapshot or SimpleNamespace(
            discovery_factor=0.5,
            priority_penalty=0.1,
            hostility_score=0.25,
            crawl_delay_seconds=None,
        )
        self.calls: list[str] = []

    async def check(self, url: str) -> RobotsCheckResult:
        self.calls.append(url)
        if self._result is None:
            return RobotsCheckResult(
                robots_url="https://example.test/robots.txt",
                decision=RobotsDecision.ALLOWED,
                confidence=RobotsConfidence.AUTHORITATIVE_ALLOW,
                reason="robots_rules_allow",
                source="robots_rules",
                is_authoritative=True,
            )
        return self._result

    def host_rules_snapshot(self, *, url: str) -> Any:
        del url
        return self._snapshot


class _FakeRateLimiter:
    def __init__(self) -> None:
        self.crawl_delay_calls: list[tuple[str | None, float | None]] = []

    async def set_host_crawl_delay(
        self,
        *,
        host: str | None,
        crawl_delay_seconds: float | None,
    ) -> None:
        self.crawl_delay_calls.append((host, crawl_delay_seconds))


class _FakeRegistrar:
    def __init__(self) -> None:
        self.advice: list[Any] = []

    async def register_host_rules_advice(
        self,
        *,
        url: str,
        advice: Any,
    ) -> None:
        del url
        self.advice.append(advice)


def _allowed_result(
    *,
    crawl_delay_seconds: float | None = None,
) -> RobotsCheckResult:
    return RobotsCheckResult(
        robots_url="https://example.test/robots.txt",
        decision=RobotsDecision.ALLOWED,
        confidence=RobotsConfidence.AUTHORITATIVE_ALLOW,
        reason="robots_rules_allow",
        source="robots_rules",
        is_authoritative=True,
        crawl_delay_seconds=crawl_delay_seconds,
    )


def _denied_result() -> RobotsCheckResult:
    return RobotsCheckResult(
        robots_url="https://example.test/robots.txt",
        decision=RobotsDecision.DISALLOWED,
        confidence=RobotsConfidence.AUTHORITATIVE_DENY,
        reason="robots_rules_disallow",
        source="robots_rules",
        is_authoritative=True,
        http_status=200,
    )


def _unknown_result(
    *,
    confidence: RobotsConfidence,
    reason: str = "http_503_unknown",
    retry_after_seconds: float | None = None,
    http_status: int = 503,
) -> RobotsCheckResult:
    return RobotsCheckResult(
        robots_url="https://example.test/robots.txt",
        decision=RobotsDecision.UNKNOWN,
        confidence=confidence,
        reason=reason,
        source="robots_error_resolver",
        is_authoritative=False,
        http_status=http_status,
        retry_after_seconds=retry_after_seconds,
    )


def _gate(
    *,
    result: RobotsCheckResult,
    mode: str = "enforce",
    registrar: _FakeRegistrar | None = None,
) -> tuple[
    RobotsRequestGate,
    _FakeChecker,
    _FakeRateLimiter,
    CollectionMetrics,
]:
    settings = RobotsSettings(mode=mode)
    checker = _FakeChecker(result=result)
    limiter = _FakeRateLimiter()
    host_normalizer = HostNormalizer()
    metrics = CollectionMetrics(
        enabled=True,
        logger=TEST_LOGGER,
        host_normalizer=host_normalizer,
    )
    gate = RobotsRequestGate(
        checker=checker,  # type: ignore[arg-type]
        settings=settings,
        rate_limiter=limiter,  # type: ignore[arg-type]
        host_normalizer=host_normalizer,
        metrics=metrics,
        logger=TEST_LOGGER,
    )
    if registrar is not None:
        gate.set_scheduler_advice_registrar(
            registrar.register_host_rules_advice
        )
    return gate, checker, limiter, metrics


@pytest.mark.asyncio
async def test_allowed_url_passes_without_metrics() -> None:
    gate, checker, limiter, metrics = _gate(result=_allowed_result())

    await gate.authorize(url="https://example.test/private")

    assert checker.calls == ["https://example.test/private"]
    assert limiter.crawl_delay_calls == []
    assert metrics.snapshot().blacklist_total == 0


@pytest.mark.asyncio
async def test_disallowed_url_raises_ignored_with_robots_reason() -> None:
    gate, _checker, _limiter, metrics = _gate(result=_denied_result())

    with pytest.raises(IgnoredFetchError) as raised:
        await gate.authorize(url="https://example.test/private")

    assert raised.value.reason == "robots_disallowed"
    assert raised.value.status_code == 200
    assert raised.value.final_url == "https://example.test/private"
    assert raised.value.metrics_recorded is True
    assert metrics.snapshot().blacklist_total == 1
    assert dict(metrics.snapshot().blacklist_by_reason) == {
        "robots_disallowed": 1
    }


@pytest.mark.asyncio
async def test_transient_unknown_raises_retryable_with_retry_after() -> None:
    gate, _checker, _limiter, _metrics = _gate(
        result=_unknown_result(
            confidence=RobotsConfidence.TRANSIENT_UNKNOWN,
            retry_after_seconds=12.0,
        )
    )

    with pytest.raises(RetryableFetchError) as raised:
        await gate.authorize(url="https://example.test/private")

    assert raised.value.retry_class == "robots"
    assert raised.value.retry_error_kind == "http_503_unknown"
    assert raised.value.retry_after_seconds == 12.0
    assert raised.value.status_code == 503


@pytest.mark.asyncio
async def test_hostile_unknown_blocks_by_default() -> None:
    gate, _checker, _limiter, _metrics = _gate(
        result=_unknown_result(
            confidence=RobotsConfidence.HOSTILE_UNKNOWN,
            reason="http_403_unknown",
            http_status=403,
        )
    )

    with pytest.raises(IgnoredFetchError) as raised:
        await gate.authorize(url="https://example.test/private")

    assert raised.value.reason == "robots_disallowed"
    assert raised.value.status_code == 403


@pytest.mark.asyncio
async def test_observe_mode_allows_and_records_would_block() -> None:
    gate, _checker, _limiter, metrics = _gate(
        result=_denied_result(),
        mode="observe",
    )

    await gate.authorize(url="https://example.test/private")

    snapshot = metrics.snapshot()
    assert snapshot.blacklist_total == 1
    assert dict(snapshot.blacklist_by_reason).get("robots_would_block") == 1


@pytest.mark.asyncio
async def test_disabled_mode_skips_checker_entirely() -> None:
    settings = RobotsSettings(mode="disabled")
    checker = _FakeChecker(result=_denied_result())
    gate = RobotsRequestGate(
        checker=checker,  # type: ignore[arg-type]
        settings=settings,
        rate_limiter=_FakeRateLimiter(),  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        logger=TEST_LOGGER,
    )

    await gate.authorize(url="https://example.test/private")

    assert checker.calls == []


@pytest.mark.asyncio
async def test_crawl_delay_is_applied_to_rate_limiter() -> None:
    gate, _checker, limiter, _metrics = _gate(
        result=_allowed_result(crawl_delay_seconds=2.5)
    )

    await gate.authorize(url="https://example.test/private")

    assert limiter.crawl_delay_calls == [("example.test", 2.5)]


@pytest.mark.asyncio
async def test_no_crawl_delay_means_no_rate_limiter_call() -> None:
    gate, _checker, limiter, _metrics = _gate(result=_allowed_result())

    await gate.authorize(url="https://example.test/private")

    assert limiter.crawl_delay_calls == []


@pytest.mark.asyncio
async def test_host_advice_is_registered_via_late_bound_registrar() -> None:
    registrar = _FakeRegistrar()
    gate, _checker, _limiter, _metrics = _gate(
        result=_denied_result(),
        registrar=registrar,
    )

    with pytest.raises(IgnoredFetchError):
        await gate.authorize(
            url="https://example.test/private",
        )

    assert len(registrar.advice) == 1
    advice = registrar.advice[0]
    assert advice.discovery_factor == 0.5
    assert advice.priority_penalty == 0.1
    assert advice.hostility_score == 0.25


@pytest.mark.asyncio
async def test_blacklisted_check_result_is_enforced_as_block() -> None:
    blacklisted = RobotsCheckResult(
        robots_url="",
        decision=RobotsDecision.DISALLOWED,
        confidence=RobotsConfidence.AUTHORITATIVE_DENY,
        reason="blacklisted",
        source="blacklist",
        is_authoritative=True,
    )
    gate, _checker, _limiter, _metrics = _gate(result=blacklisted)

    with pytest.raises(IgnoredFetchError) as raised:
        await gate.authorize(
            url="https://example.test/private",
        )

    assert raised.value.reason == "robots_disallowed"
