"""Manage host feedback state, LRU eviction and expanded-host membership."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING


@dataclass(slots=True)
class HostFeedback:
    """State for adaptive host-level scheduling feedback."""

    observed_tasks: int
    budgeted_tasks: int
    info_gain_ewma: float
    quality_ewma: float


if TYPE_CHECKING:
    from crawler.extraction.hosts_extractor import HostExtractor
    from crawler.governance.domains.host_normalizer import HostNormalizer


class HostFeedbackAggregator:
    """Manage host feedback state, LRU eviction and expanded-host membership."""

    def __init__(
        self,
        *,
        max_hosts: int | None,
        default_info_gain: float,
        default_host_quality: float,
        host_extractor: HostExtractor,
        host_normalizer: HostNormalizer,
    ) -> None:
        self._max_hosts = max_hosts
        self._default_info_gain = default_info_gain
        self._default_host_quality = default_host_quality
        self._host_extractor = host_extractor
        self._host_normalizer = host_normalizer
        self._feedback: OrderedDict[str, HostFeedback] = OrderedDict()
        self._expanded_hosts: set[tuple[str, str]] = set()

    def get(self, host: str) -> HostFeedback | None:
        """Return existing feedback for a host."""
        normalized = self._host_normalizer.normalize(host)
        return (
            self._feedback.get(normalized) if normalized is not None else None
        )

    def host_for_url(self, url: str) -> str | None:
        """Return the host key used for scheduler feedback for a URL."""
        return self._host_normalizer.normalize(
            self._host_extractor.extract(url)
        )

    def get_or_create(self, host: str) -> HostFeedback:
        """Return existing feedback or create a bounded default state."""
        host = self._host_normalizer.require(host)
        feedback = self._feedback.get(host)
        if feedback is not None:
            self._feedback.move_to_end(host)
            return feedback

        if (
            self._max_hosts is not None
            and len(self._feedback) >= self._max_hosts
        ):
            self._feedback.popitem(last=False)

        feedback = HostFeedback(
            observed_tasks=0,
            budgeted_tasks=0,
            info_gain_ewma=self._default_info_gain,
            quality_ewma=self._default_host_quality,
        )
        self._feedback[host] = feedback
        return feedback

    def is_expanded_host(self, source_name: str, host: str) -> bool:
        """Return whether a host was accepted by adaptive expansion."""
        normalized = self._host_normalizer.normalize(host)
        if normalized is None:
            return False
        return (source_name, normalized) in self._expanded_hosts

    def mark_expanded(self, source_name: str, host: str) -> None:
        """Mark a host as accepted through adaptive expansion."""
        normalized = self._host_normalizer.require(host)
        self._expanded_hosts.add((source_name, normalized))

    @property
    def expanded_host_count(self) -> int:
        """Return the number of adaptively expanded hosts."""
        return len(self._expanded_hosts)
