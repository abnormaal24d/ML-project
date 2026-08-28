"""Fetch request context model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.classification.media_kind import MediaKind
    from crawler.fetching.acceptance.decision import FetchAcceptance


@dataclass(frozen=True, slots=True)
class FetchRequestContext:
    """Immutable request context for a logical fetch operation."""

    url: str
    host: str
    source_name: str
    requested_kind: MediaKind
    acceptance_mode: str
    acceptance: FetchAcceptance
    task_context: dict[str, object] = field(default_factory=dict)
