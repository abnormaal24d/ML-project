"""Execute optional HEAD transport validation before GET requests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING

import aiohttp

from crawler.classification.media_kind import MediaKind
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.fetching.media.strategy import (
    HeadPreflightAction,
    HeadPreflightResult,
)

if TYPE_CHECKING:
    from config.collection.fetching import FetcherSettings
    from config.collection.http_rules import TimeoutRulesSettings
    from crawler.fetching.network.preflight.response_evaluator import (
        HeadPreflightResponseEvaluator,
    )
    from crawler.fetching.network.request import AiohttpRequestRunner
    from crawler.fetching.request.context import FetchRequestContext
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from logger.project_logger import ProjectLogger

FeedbackReporter = Callable[..., Awaitable[None]]


@dataclass(slots=True)
class _HeadPreflightUsefulnessState:
    useless_count: int = 0
    suppressed_until_monotonic: float = 0.0


class HeadPreflightExecutor:
    """Own HEAD eligibility, execution, feedback, and adaptive suppression."""

    def __init__(
        self,
        *,
        settings: FetcherSettings,
        timeout_rules: TimeoutRulesSettings,
        request_runner: AiohttpRequestRunner,
        response_evaluator: HeadPreflightResponseEvaluator,
        logger: ProjectLogger,
        feedback_reporter: FeedbackReporter | None,
        host_normalizer: HostNormalizer,
        host_allowlist: frozenset[str],
        monotonic_seconds: Callable[[], float],
    ) -> None:
        self._settings = settings
        self._timeout_rules = timeout_rules
        self._request_runner = request_runner
        self._response_evaluator = response_evaluator
        self._logger = logger
        self._feedback_reporter = feedback_reporter
        self._host_normalizer = host_normalizer
        self._host_allowlist = host_allowlist
        self._monotonic_seconds = monotonic_seconds
        self._usefulness_by_host: dict[str, _HeadPreflightUsefulnessState] = {}
        self._usefulness_lock = RLock()

    async def run(
        self,
        *,
        context: FetchRequestContext,
        session: aiohttp.ClientSession,
        request_headers: dict[str, str],
    ) -> HeadPreflightResult:
        """Run HEAD preflight for a host when enabled by settings."""

        url = context.url
        host_key = context.host
        if not self._should_run_head_preflight(
            host=host_key,
            requested_kind=context.requested_kind,
        ):
            return HeadPreflightResult(attempted=False)

        status_code_for_feedback: int | None = None
        started_at = self._monotonic_seconds()
        final_host_for_feedback: str | None = host_key

        try:
            timeout = aiohttp.ClientTimeout(
                total=self._timeout_rules.head_preflight_timeout_seconds,
            )
            async with self._request_runner.perform(
                session=session,
                method="HEAD",
                url=context.url,
                source_name=context.source_name,
                base_headers=request_headers,
                timeout=timeout,
                defer_if_rate_limited=False,
                enrich_headers=None,
            ) as response:
                status_code = int(response.status)
                status_code_for_feedback = status_code
                response_url = getattr(response, "url", None)
                final_host_for_feedback = (
                    getattr(response_url, "host", None) or host_key
                )

                result = self._response_evaluator.handle_head_result(
                    context=context,
                    url=url,
                    host=host_key,
                    response=response,
                )
                self._record_usefulness(host=host_key, result=result)
                return result

        except IgnoredFetchError as exc:
            if not exc.reason.startswith("redirect_"):
                raise
            self._logger.debug(
                "head_preflight_redirect_rejected",
                url=url,
                host=host_key,
                reason=exc.reason,
                final_url=exc.final_url,
                status_code=exc.status_code,
                acceptance_mode=context.acceptance_mode,
            )
            return HeadPreflightResult(
                attempted=True,
                allowed=True,
                status_code=exc.status_code,
                final_url=exc.final_url,
                failure_type=type(exc).__name__,
                soft_rejected=True,
                rejection_reason=exc.reason,
                record_kind=context.requested_kind,
            )
        except RetryableFetchError as exc:
            if (
                exc.retry_class != "local_transport_metadata"
                or exc.retry_error_kind != "peer_address_unavailable"
            ):
                raise
            self._logger.debug(
                "head_preflight_peer_address_unavailable",
                url=url,
                host=host_key,
                acceptance_mode=context.acceptance_mode,
            )
            return HeadPreflightResult(
                attempted=True,
                allowed=True,
                status_code=None,
                failure_type=type(exc).__name__,
                soft_rejected=True,
                rejection_reason="peer_address_unavailable",
                record_kind=context.requested_kind,
            )
        except TimeoutError as exc:
            status_code_for_feedback = 504
            self._logger.debug(
                "head_preflight_failed",
                url=url,
                host=host_key,
                error_type=type(exc).__name__,
                acceptance_mode=context.acceptance_mode,
            )
            return HeadPreflightResult(
                attempted=True,
                allowed=True,
                status_code=504,
                failure_type=type(exc).__name__,
                soft_rejected=True,
                rejection_reason="head_preflight_failed",
                record_kind=context.requested_kind,
            )
        except aiohttp.ClientError as exc:
            status_code_for_feedback = 503
            self._logger.debug(
                "head_preflight_failed",
                url=url,
                host=host_key,
                error_type=type(exc).__name__,
                acceptance_mode=context.acceptance_mode,
            )
            return HeadPreflightResult(
                attempted=True,
                allowed=True,
                status_code=503,
                failure_type=type(exc).__name__,
                soft_rejected=True,
                rejection_reason="head_preflight_failed",
                record_kind=context.requested_kind,
            )
        finally:
            await self._record_feedback_safely(
                url=url,
                host=final_host_for_feedback or host_key,
                status_code=status_code_for_feedback,
                started_at=started_at,
            )

    async def _record_feedback_safely(
        self,
        *,
        url: str,
        host: str | None,
        status_code: int | None,
        started_at: float,
    ) -> None:
        if (
            status_code is None
            or not self._settings.head_preflight_counts_toward_rate_feedback
            or self._feedback_reporter is None
        ):
            return
        try:
            await self._feedback_reporter(
                host=host,
                status_code=status_code,
                latency_seconds=self._monotonic_seconds() - started_at,
                bytes_downloaded=0,
                quality_score=None,
                count_toward_rate_feedback=True,
                count_toward_metrics=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._logger.debug(
                "head_preflight_feedback_failed",
                url=url,
                host=host,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    def _should_run_head_preflight(
        self,
        *,
        host: str,
        requested_kind: MediaKind,
    ) -> bool:
        if not self._settings.head_preflight_enabled:
            return False

        allowed_kinds = self._settings.head_preflight_task_kinds
        if allowed_kinds and requested_kind.value not in allowed_kinds:
            return False

        if not self._settings.head_preflight_for_all_hosts:
            if not self._host_allowlist or host not in self._host_allowlist:
                return False

        if requested_kind in {MediaKind.AUDIO, MediaKind.VIDEO}:
            return True
        return not self._is_usefulness_suppressed(host=host)

    def _record_usefulness(
        self,
        *,
        host: str | None,
        result: HeadPreflightResult,
    ) -> None:
        if not self._settings.head_preflight_adaptive_skip_enabled:
            return
        canonical_host = self._host_normalizer.normalize(host)
        if canonical_host is None:
            return

        if self._was_useful(result=result):
            with self._usefulness_lock:
                self._usefulness_by_host.pop(canonical_host, None)
            return

        threshold = int(self._settings.head_preflight_useless_host_threshold)
        cooldown_seconds = float(
            self._settings.head_preflight_useless_host_cooldown_seconds
        )
        if cooldown_seconds <= 0:
            return

        with self._usefulness_lock:
            state = self._usefulness_by_host.setdefault(
                canonical_host,
                _HeadPreflightUsefulnessState(),
            )
            state.useless_count += 1
            if state.useless_count < threshold:
                return
            state.suppressed_until_monotonic = (
                self._monotonic_seconds() + cooldown_seconds
            )
            useless_count = state.useless_count

        self._logger.debug(
            "head_preflight_adaptively_suppressed",
            host=canonical_host,
            useless_count=useless_count,
            cooldown_seconds=cooldown_seconds,
        )

    def _is_usefulness_suppressed(self, *, host: str | None) -> bool:
        if not self._settings.head_preflight_adaptive_skip_enabled:
            return False
        canonical_host = self._host_normalizer.normalize(host)
        if canonical_host is None:
            return False

        now = self._monotonic_seconds()
        with self._usefulness_lock:
            state = self._usefulness_by_host.get(canonical_host)
            if state is None:
                return False
            if state.suppressed_until_monotonic <= now:
                if state.suppressed_until_monotonic > 0.0:
                    self._usefulness_by_host.pop(canonical_host, None)
                return False
            return True

    @staticmethod
    def _was_useful(*, result: HeadPreflightResult) -> bool:
        return (
            result.action != HeadPreflightAction.FETCH_FULL
            or result.content_length is not None
            or result.content_type is not None
            or not result.allowed
        )
