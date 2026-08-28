"""Per-host byte budget so one host cannot crowd out other modalities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.governance.domains.host_normalizer import HostNormalizer

_MEDIA_KINDS = frozenset(
    {
        MediaKind.IMAGE,
        MediaKind.AUDIO,
        MediaKind.DOCUMENT,
        MediaKind.VIDEO,
    }
)


@dataclass
class HostMediaByteBudget:
    """Track downloaded bytes per host and enforce per-kind caps."""

    host_normalizer: HostNormalizer
    max_bytes_per_host: int
    max_bytes_per_host_by_kind: dict[str, int]
    _host_totals: dict[str, int] = field(default_factory=dict)
    _host_kind_totals: dict[tuple[str, MediaKind], int] = field(
        default_factory=dict,
    )

    def record_download(
        self,
        *,
        host: str | None,
        kind: MediaKind,
        byte_count: int,
    ) -> None:
        if byte_count <= 0:
            return
        normalized_host = self.host_normalizer.require(host)
        self._host_totals[normalized_host] = self._host_totals.get(
            normalized_host, 0
        ) + int(byte_count)
        if kind in _MEDIA_KINDS:
            key = (normalized_host, kind)
            self._host_kind_totals[key] = self._host_kind_totals.get(
                key, 0
            ) + int(byte_count)

    def host_budget_exhausted(
        self,
        *,
        task: CrawlTask,
        host: str | None,
    ) -> tuple[bool, str | None]:
        normalized_host = self.host_normalizer.require(host)

        kind = task.kind
        host_total = self._host_totals.get(normalized_host, 0)
        if host_total >= self.max_bytes_per_host:
            return True, "host_media_byte_budget_exceeded"

        if kind in _MEDIA_KINDS:
            kind_limit = self.max_bytes_per_host_by_kind.get(kind.value)
            if kind_limit is not None and kind_limit > 0:
                kind_total = self._host_kind_totals.get(
                    (normalized_host, kind),
                    0,
                )
                if kind_total >= int(kind_limit):
                    return True, (f"host_{kind.value}_byte_budget_exceeded")

        return False, None

    def snapshot(self) -> dict[str, object]:
        return {
            "host_totals": dict(self._host_totals),
            "host_kind_totals": {
                f"{host}:{kind.value}": total
                for (host, kind), total in self._host_kind_totals.items()
            },
        }
