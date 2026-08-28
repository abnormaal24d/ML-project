"""HTTP response-status contract: acceptance, retry, ignore, and hints.

Unit-level only. Exercises FetchResponseStatusRules and the pure
rate-limit header parsers exactly as production classifies status codes.
"""

from __future__ import annotations

import datetime
from datetime import timezone

import pytest

from config.collection.fetching import FetcherSettings
from config.collection.http_rules import HttpStatusRulesSettings
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.fetching.response.rate_limit_hints import (
    ResponseRateLimitHints,
    parse_retry_after_seconds,
)
from crawler.fetching.response.status_rules import FetchResponseStatusRules
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.rate_limit.rate_limit_rules import RateLimitRules
from crawler.governance.rate_limit.rate_limit_slot_scheduler import (
    RateLimitSlotScheduler,
)
from crawler.governance.rate_limit.rate_limit_state_registry import (
    RateLimitStateRegistry,
)
from crawler.governance.rate_limit.rate_limiter import RateLimiter
from tests.support.logging import TEST_LOGGER


class _Logger:
    def debug(self, *args: object, **kwargs: object) -> None:
        return None

    def info(self, *args: object, **kwargs: object) -> None:
        return None

    def warning(self, *args: object, **kwargs: object) -> None:
        return None


def _rules(
    *,
    raise_for_non_success: bool = True,
    retryable: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504),
    accepted_non_success: tuple[int, ...] = (304,),
) -> FetchResponseStatusRules:
    return FetchResponseStatusRules(
        settings=FetcherSettings(
            raise_for_non_success_status=raise_for_non_success
        ),
        status_rules=HttpStatusRulesSettings(
            retryable=retryable,
            accepted_non_success=accepted_non_success,
        ),
        logger=_Logger(),  # type: ignore[arg-type]
    )


def _rate_limiter(
    *,
    honor_retry_after: bool = True,
    max_retry_after_seconds: float = 60.0,
    min_requests_per_second: float = 1.0,
    max_requests_per_second: float = 1.0,
    backoff_factor: float = 1.0,
    ramp_up_factor: float = 1.0,
) -> RateLimiter:
    host_normalizer = HostNormalizer()
    return RateLimiter(
        state_registry=RateLimitStateRegistry(
            host_normalizer=host_normalizer,
            default_adaptive_requests_per_second=1.0,
            default_effective_requests_per_second=1.0,
        ),
        slot_scheduler=RateLimitSlotScheduler(
            burst_size=1,
            logger=TEST_LOGGER,
            reservation_log_threshold_seconds=1.0,
        ),
        adaptive_rules=RateLimitRules(
            logger=TEST_LOGGER,
            min_requests_per_second_value=min_requests_per_second,
            max_requests_per_second_value=max_requests_per_second,
            backoff_factor=backoff_factor,
            ramp_up_factor=ramp_up_factor,
            error_cooldown_seconds=0.0,
            feedback_status_codes=frozenset({429}),
            default_crawl_delay_seconds=None,
        ),
        default_effective_requests_per_second=1.0,
        honor_retry_after=honor_retry_after,
        max_retry_after_seconds=max_retry_after_seconds,
        logger=TEST_LOGGER,
    )


def test_200_ok_is_accepted_without_raising() -> None:
    _rules().handle(status_code=200, url="https://example.test/a", host="t")


def test_200_ok_accepted_even_when_raise_disabled() -> None:
    rules = _rules(raise_for_non_success=False)
    rules.handle(status_code=200, url="https://example.test/a", host="t")


def test_304_not_modified_is_overridden_accepted_non_success() -> None:
    _rules().handle(status_code=304, url="https://example.test/a", host="t")


def test_204_no_content_handled_as_success_class_without_raising() -> None:
    _rules().handle(
        status_code=204, url="https://example.test/empty", host="t"
    )


def test_204_no_content_passes_when_raise_disabled() -> None:
    _rules(raise_for_non_success=False).handle(
        status_code=204, url="https://example.test/empty", host="t"
    )


def test_404_not_found_is_dropped_without_retry() -> None:
    with pytest.raises(IgnoredFetchError) as raised:
        _rules().handle(
            status_code=404, url="https://example.test/missing", host="t"
        )
    assert raised.value.reason == "non_success_status_404"
    assert raised.value.status_code == 404
    assert raised.value.final_url == "https://example.test/missing"


def test_403_forbidden_is_authoritative_deny() -> None:
    with pytest.raises(IgnoredFetchError) as raised:
        _rules().handle(
            status_code=403,
            url="https://example.test/denied",
            host="example.test",
            final_url="https://example.test/denied?final=1",
        )
    assert raised.value.reason == "server_denied_403"
    assert raised.value.observed_bytes == 0
    assert raised.value.status_code == 403
    assert raised.value.final_url == "https://example.test/denied?final=1"


def test_429_rate_limited_is_retryable_status_retry() -> None:
    with pytest.raises(RetryableFetchError) as raised:
        _rules().handle(
            status_code=429, url="https://example.test/a", host="t"
        )
    assert raised.value.retry_class == "status_retry"
    assert raised.value.retry_error_kind == "http_429"
    assert raised.value.status_code == 429


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_500_series_server_failures_are_retryable(status_code: int) -> None:
    with pytest.raises(RetryableFetchError) as raised:
        _rules().handle(
            status_code=status_code, url="https://example.test/a", host="t"
        )
    assert raised.value.retry_error_kind == f"http_{status_code}"


def test_non_retryable_server_status_is_ignored_failure() -> None:
    with pytest.raises(IgnoredFetchError) as raised:
        _rules().handle(
            status_code=501, url="https://example.test/a", host="t"
        )
    assert raised.value.reason == "non_success_status_501"


def test_accepted_non_success_override_allows_404() -> None:
    rules = _rules(accepted_non_success=(304, 404))
    rules.handle(status_code=404, url="https://example.test/a", host="t")


def test_accepted_non_success_override_applies_to_200_class() -> None:
    rules = _rules(accepted_non_success=(304, 404))
    rules.handle(status_code=200, url="https://example.test/a", host="t")


def test_retry_after_seconds_numeric_value() -> None:
    now = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert parse_retry_after_seconds("120", now=now) == 120.0
    assert parse_retry_after_seconds("0", now=now) == 0.0
    assert parse_retry_after_seconds("-5", now=now) is None


@pytest.mark.parametrize("value", ["inf", "-inf", "nan"])
def test_retry_after_nonfinite_values_are_rejected(value: str) -> None:
    now = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert parse_retry_after_seconds(value, now=now) is None


def test_retry_after_seconds_http_date_future() -> None:
    now = datetime.datetime(2015, 10, 20, 0, 0, 0, tzinfo=timezone.utc)
    delay = parse_retry_after_seconds(
        "Wed, 21 Oct 2015 07:28:00 GMT",
        now=now,
    )
    assert delay is not None
    assert delay > 0.0


def test_retry_after_past_http_date_clamped_to_zero() -> None:
    now = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    delay = parse_retry_after_seconds(
        "Wed, 21 Oct 2015 07:28:00 GMT",
        now=now,
    )
    assert delay == 0.0


@pytest.mark.parametrize(
    "value",
    [None, "", "   ", "not-a-date", "12.5oct2024"],
)
def test_retry_after_malformed_or_empty_values_return_none(
    value: str | None,
) -> None:
    now = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert parse_retry_after_seconds(value, now=now) is None


def test_hints_from_headers_parse_retry_after_seconds() -> None:
    now = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    hints = ResponseRateLimitHints.from_headers(
        {"Retry-After": "42", "X-RateLimit-Remaining": "0"},
        now=now,
    )
    assert hints.retry_after_seconds == 42.0
    assert hints.rate_limit_remaining == 0
    assert hints.rate_limit_reset_seconds is None
    assert hints.has_delay is True


def test_hints_delay_from_remaining_zero_and_future_reset() -> None:
    now = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    future_epoch = now.timestamp() + 30.0
    hints = ResponseRateLimitHints.from_headers(
        {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": f"{future_epoch:.0f}",
        },
        now=now,
    )
    assert hints.rate_limit_reset_seconds is not None
    assert hints.rate_limit_reset_seconds > 0.0
    assert hints.has_delay is True


def test_hints_no_delay_when_remaining_positive() -> None:
    now = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    hints = ResponseRateLimitHints.from_headers(
        {"x-ratelimit-remaining": "7"},
        now=now,
    )
    assert hints.retry_after_seconds is None
    assert hints.rate_limit_reset_seconds is None
    assert hints.has_delay is False


def test_hints_retry_after_zero_yields_no_delay() -> None:
    now = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    hints = ResponseRateLimitHints.from_headers(
        {"Retry-After": "0"},
        now=now,
    )
    assert hints.retry_after_seconds == 0.0
    assert hints.has_delay is False


def test_hints_malformed_rate_limit_headers_are_ignored() -> None:
    now = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    hints = ResponseRateLimitHints.from_headers(
        {"x-ratelimit-remaining": "abc", "x-ratelimit-reset": "abc"},
        now=now,
    )
    assert hints.rate_limit_remaining is None
    assert hints.rate_limit_reset_seconds is None
    assert hints.has_delay is False


@pytest.mark.parametrize("value", ["inf", "-inf", "nan"])
def test_hints_nonfinite_rate_limit_reset_is_ignored(value: str) -> None:
    now = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    hints = ResponseRateLimitHints.from_headers(
        {"x-ratelimit-remaining": "0", "x-ratelimit-reset": value},
        now=now,
    )
    assert hints.rate_limit_reset_seconds is None
    assert hints.has_delay is False


def test_hints_short_epoch_reset_is_absolute_seconds() -> None:
    now = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    hints = ResponseRateLimitHints.from_headers(
        {"x-ratelimit-reset": "120"},
        now=now,
    )
    assert hints.rate_limit_reset_seconds == 120.0
    assert hints.has_delay is False


@pytest.mark.asyncio
async def test_rate_limiter_caps_server_supplied_delays() -> None:
    limiter = _rate_limiter(max_retry_after_seconds=30.0)

    applied = await limiter.apply_response_rate_limit_hints(
        host="example.test",
        retry_after_seconds=120.0,
    )

    assert applied == 30.0


@pytest.mark.asyncio
async def test_rate_limiter_can_ignore_retry_after_without_ignoring_reset() -> (
    None
):
    limiter = _rate_limiter(
        honor_retry_after=False,
        max_retry_after_seconds=30.0,
    )

    retry_after_applied = await limiter.apply_response_rate_limit_hints(
        host="retry.example.test",
        retry_after_seconds=20.0,
    )
    reset_applied = await limiter.apply_response_rate_limit_hints(
        host="reset.example.test",
        retry_after_seconds=20.0,
        rate_limit_remaining=0,
        rate_limit_reset_seconds=10.0,
    )

    assert retry_after_applied is None
    assert reset_applied == 10.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value", [float("inf"), float("-inf"), float("nan"), -1.0]
)
async def test_rate_limiter_rejects_invalid_direct_hint_values(
    value: float,
) -> None:
    limiter = _rate_limiter()

    applied = await limiter.apply_response_rate_limit_hints(
        host="example.test",
        retry_after_seconds=value,
    )

    assert applied is None


@pytest.mark.asyncio
async def test_rate_limiter_backs_off_and_ramps_up_within_configured_bounds() -> (
    None
):
    limiter = _rate_limiter(
        min_requests_per_second=0.25,
        max_requests_per_second=2.0,
        backoff_factor=0.5,
        ramp_up_factor=2.0,
    )
    host = "adaptive.example.test"

    await limiter.report_result(
        host=host,
        status_code=429,
        latency_seconds=0.1,
    )
    assert limiter.host_requests_per_second(host) == 0.5

    await limiter.report_result(
        host=host,
        status_code=200,
        latency_seconds=0.1,
    )
    assert limiter.host_requests_per_second(host) == 1.0

    await limiter.report_result(
        host=host,
        status_code=200,
        latency_seconds=0.1,
    )
    await limiter.report_result(
        host=host,
        status_code=200,
        latency_seconds=0.1,
    )
    assert limiter.host_requests_per_second(host) == 2.0


class _RecordingFeedbackRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self.skipped: list[dict[str, object]] = []

    async def record(self, **payload: object) -> None:
        self.records.append(payload)

    async def record_skipped(self, **payload: object) -> None:
        self.skipped.append(payload)


class _RecordingSuppressionStore:
    def __init__(self) -> None:
        self.results: list[tuple[str, int]] = []

    def record_response_status(
        self,
        *,
        host: str,
        status_code: int,
    ) -> None:
        self.results.append((host, status_code))


@pytest.mark.asyncio
async def test_server_denied_403_registers_non_success_feedback() -> None:
    from types import SimpleNamespace

    from crawler.fetching.feedback.attempt_recorder import (
        FetchAttemptOutcomeRecorder,
    )

    feedback_recorder = _RecordingFeedbackRecorder()
    suppression_store = _RecordingSuppressionStore()
    recorder = FetchAttemptOutcomeRecorder(
        feedback_recorder=feedback_recorder,  # type: ignore[arg-type]
        host_suppression_store=suppression_store,  # type: ignore[arg-type]
        status_rules=HttpStatusRulesSettings(),
        logger=_Logger(),  # type: ignore[arg-type]
    )
    exc = IgnoredFetchError(
        reason="server_denied_403",
        status_code=403,
        observed_bytes=0,
    )
    await recorder.record_ignored_fetch(
        context=SimpleNamespace(
            url="https://example.test/denied",
            host="example.test",
            requested_kind="page",
            acceptance_mode="strict",
        ),
        status_code=403,
        started_at=0.0,
        exc=exc,
    )

    assert feedback_recorder.records
    assert feedback_recorder.records[0]["status_code"] == 403
    assert (("example.test", 403)) in suppression_store.results
    assert exc.metrics_recorded is True
