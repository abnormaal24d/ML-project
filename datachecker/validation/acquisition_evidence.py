"""Consumer-owned acquisition evidence contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CrawlAcquisitionEvidence(Protocol):
    """Evidence of crawl acquisition health.

    Consumer-owned: datachecker defines the contract it needs;
    the crawler producer provides a concrete dataclass that
    structurally satisfies this protocol.
    """

    @property
    def object_records_total(self) -> int: ...

    @property
    def successful_requests_total(self) -> int: ...

    @property
    def quality_score(self) -> float: ...
