"""Crawler-wide exception taxonomy.

This module contains only crawler-domain base exceptions that are shared
across crawler subpackages. Subpackage-specific exceptions should stay close
to their owner, for example:

- crawler.fetching.errors for HTTP/fetch failures
- this module for shared runtime and worker lifecycle failures
- storage-specific exceptions should remain near their owning storage component
"""


class CrawlerError(Exception):
    """Base class for all crawler-domain failures."""


class RetryableCrawlerError(CrawlerError):
    """Base class for crawler failures that may be retried safely."""


class CrawlerTimeoutError(RetryableCrawlerError, TimeoutError):
    """Base class for crawler-domain timeout failures."""


class ParsingError(CrawlerError):
    """Raised when content parsing fails in a non-retryable way."""


# Consolidated from runtime/errors per audit P1
class CrawlerRuntimeError(CrawlerError):
    """Raised when crawler runtime orchestration or accounting fails."""


class CrawlerDrainStalledError(CrawlerRuntimeError):
    """Raised when scheduler drain stalls with busy workers."""


# Consolidated from worker/errors per audit P1
class WorkerPoolError(CrawlerRuntimeError):
    """Base worker-pool error."""


class WorkerPoolFailedError(WorkerPoolError):
    """Raised when one or more workers failed and fail-fast is enabled."""


class AnalysisLaneFailedError(CrawlerRuntimeError):
    """Raised when an analysis lane can no longer process work safely."""
