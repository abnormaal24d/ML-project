"""Fetch, decode, and parse robots.txt documents into RobotFileParser."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING
from urllib.robotparser import RobotFileParser

from crawler.governance.robots.robots_fetcher import RobotsFetchResult

if TYPE_CHECKING:
    from crawler.governance.robots.robots_fetcher import RobotsFetcher
    from logger.project_logger import ProjectLogger

BROTLI_AVAILABLE = find_spec("brotli") is not None

CrawlDelayTable = tuple[tuple[tuple[str, ...], float], ...]


class ParsedRobotsFileParser(RobotFileParser):
    """Robot parser enriched with the exact crawl-delay directives."""

    crawl_delay_seconds_table: CrawlDelayTable


def extract_crawl_delay_table(content: str) -> CrawlDelayTable:
    """Extract crawl-delay directives that urllib's parser drops.

    urllib only records integer crawl-delay values and ignores groups that
    declare a delay without any allow/disallow rule; fractional delays are
    common in real robots documents.
    """

    groups: list[tuple[list[str], float | None]] = []
    current_group: int = -1

    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip().lower()
        value = raw_value.strip()

        if key == "user-agent" and value:
            groups.append(([value], None))
            current_group += 1
        elif key == "crawl-delay" and current_group >= 0:
            try:
                delay = float(value)
            except ValueError:
                continue
            if delay >= 0:
                groups[current_group] = (
                    groups[current_group][0],
                    delay,
                )

    return tuple(
        (tuple(useragents), delay)
        for useragents, delay in groups
        if delay is not None
    )


def crawl_delay_seconds_for_agent(
    *,
    table: CrawlDelayTable,
    user_agent: str,
) -> float | None:
    """Return the first crawl-delay matching the agent, else the default."""

    agent = user_agent.strip().lower()
    default_delay: float | None = None

    for useragents, delay in table:
        if any(ua == "*" for ua in useragents):
            default_delay = delay
        elif any(agent.startswith(ua.strip().lower()) for ua in useragents):
            return delay

    return default_delay


@dataclass(frozen=True, slots=True)
class RobotsParserLoadResult:
    """Parsed robots document and transport metadata from one fetch."""

    parser: RobotFileParser
    fetch_result: RobotsFetchResult


class RobotsParserLoader:
    """Load and parse robots.txt documents."""

    _MAX_ROBOTS_BODY_BYTES = 512 * 1024

    def __init__(
        self,
        *,
        user_agent: str,
        accept_language_header: str | None,
        accept_compressed: bool,
        fetcher: RobotsFetcher,
        logger: ProjectLogger,
    ) -> None:
        self._logger = logger
        self._user_agent = user_agent
        self._accept_language_header = accept_language_header
        self._accept_compressed = accept_compressed
        self._fetcher = fetcher

    async def load(
        self,
        robots_url: str,
        timeout: float,
    ) -> RobotsParserLoadResult:
        """Fetch and parse the robots file for the URL."""

        headers = self._request_headers()
        self._logger.debug(
            "robots_parser_load_started",
            extra={
                "robots_url": robots_url,
                "timeout_seconds": timeout,
                "user_agent": headers["User-Agent"],
            },
        )

        fetch_result = await self._fetcher.fetch(
            robots_url=robots_url,
            headers=headers,
            timeout_seconds=timeout,
            max_body_bytes=self._MAX_ROBOTS_BODY_BYTES,
        )

        return RobotsParserLoadResult(
            parser=self._parse_document(fetch_result=fetch_result),
            fetch_result=fetch_result,
        )

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": self._user_agent,
            "Accept": "text/plain,*/*;q=0.1",
        }
        if self._accept_language_header:
            headers["Accept-Language"] = self._accept_language_header

        if self._accept_compressed:
            encodings = ["gzip", "deflate"]
            if BROTLI_AVAILABLE:
                encodings.append("br")
            headers["Accept-Encoding"] = ", ".join(encodings)

        return headers

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parse_document(
        self,
        *,
        fetch_result: RobotsFetchResult,
    ) -> RobotFileParser:
        content = fetch_result.body.decode("utf-8", errors="ignore")
        parser = ParsedRobotsFileParser()
        parser.set_url(fetch_result.final_url)
        parser.parse(content.splitlines())

        parser.crawl_delay_seconds_table = extract_crawl_delay_table(content)

        self._logger.debug(
            "robots_parser_load_completed",
            extra={
                "robots_url": fetch_result.requested_url,
                "final_url": fetch_result.final_url,
                "status_code": fetch_result.status_code,
                "byte_size": len(fetch_result.body),
                "line_count": len(content.splitlines()),
                "latency_seconds": round(fetch_result.latency_seconds, 4),
            },
        )
        return parser
