"""Host-level feedback registration for completed crawler tasks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from math import isfinite
from typing import Any

from crawler.extraction.hosts_extractor import HostExtractor
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.rate_limit.rate_limiter import RateLimiter
from logger.project_logger import ProjectLogger

RegisterHostAdviceCallback = Callable[..., Awaitable[None]]


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(
        value, (str, bytes, bytearray, int, float)
    ):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


class CrawlerHostFeedback:
    """Apply crawler host advice persistence and crawl-delay feedback."""

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        register_host_advice: RegisterHostAdviceCallback | None,
        rate_limiter: RateLimiter | None,
        host_extractor: HostExtractor | None,
        host_normalizer: HostNormalizer,
    ) -> None:
        self._logger = logger
        self._register_host_advice = register_host_advice
        self._rate_limiter = rate_limiter
        self._host_extractor = host_extractor
        self._host_normalizer = host_normalizer

    async def register(self, *, url: str, advice: Any | None) -> None:
        """Register host-level feedback for a crawled URL."""

        if advice is None:
            return

        crawl_delay_seconds = _coerce_float(
            getattr(advice, "crawl_delay_seconds", None)
        )
        if crawl_delay_seconds is not None and (
            not isfinite(crawl_delay_seconds) or crawl_delay_seconds < 0.0
        ):
            crawl_delay_seconds = None

        if self._register_host_advice is not None:
            await self._register_host_advice(url=url, advice=advice)

        host = self._canonical_host_from_url_or_none(url)
        if (
            self._rate_limiter is not None
            and host is not None
            and crawl_delay_seconds is not None
        ):
            await self._rate_limiter.set_host_crawl_delay(
                host=host,
                crawl_delay_seconds=crawl_delay_seconds,
            )

        self._logger.debug(
            "crawler_host_advice_feedback_registered",
            url=url,
            host=host,
            discovery_factor=getattr(advice, "discovery_factor", None),
            priority_penalty=getattr(advice, "priority_penalty", None),
            hostility_score=getattr(advice, "hostility_score", None),
            crawl_delay_seconds=crawl_delay_seconds,
        )

    def _canonical_host_from_url_or_none(self, url: str) -> str | None:
        if self._host_extractor is None:
            return None

        host = self._host_extractor.extract(url)
        if host is None:
            return None

        return self._host_normalizer.normalize(host)
