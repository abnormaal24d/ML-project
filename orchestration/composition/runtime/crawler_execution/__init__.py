"""Crawler execution subgraph composition.

Re-exports the public API so callers can continue using:

    from orchestration.composition.runtime.crawler_execution import build_crawler_execution
"""

from orchestration.composition.runtime.crawler_execution.contracts import (
    CrawlerExecutionOverrides,
    CrawlerExecutionServices,
)
from orchestration.composition.runtime.crawler_execution.composition import (
    build_crawler_execution,
)

__all__ = [
    "CrawlerExecutionOverrides",
    "CrawlerExecutionServices",
    "build_crawler_execution",
]
