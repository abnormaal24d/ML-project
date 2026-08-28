"""Resolve and admit asset extraction candidates."""

from __future__ import annotations

import re
from dataclasses import replace

from crawler.extraction.assets.candidate.asset_candidate_deduper import (
    AssetCandidateDeduper,
)
from crawler.extraction.assets.candidate.asset_extraction_records import (
    AssetCandidate,
    AssetExtractionCandidateState,
)
from crawler.extraction.candidates.url_candidate_resolution import (
    UrlCandidateResolution,
)

_SRCSET_DESCRIPTOR_ARTIFACT_PATTERN = re.compile(
    r"^(?:/?\d+/?|\d+(?:\.\d+)?[wx])$",
    flags=re.IGNORECASE,
)


class AssetCandidateResolution:
    """Resolve and deduplicate assets through one public entry point."""

    def __init__(
        self,
        *,
        url_candidate_resolution: UrlCandidateResolution,
        deduper: AssetCandidateDeduper,
    ) -> None:
        if url_candidate_resolution is None:
            raise ValueError("url_candidate_resolution is required")
        if deduper is None:
            raise ValueError("deduper is required")

        self._url_candidate_resolution = url_candidate_resolution
        self._deduper = deduper

    def add_candidate(
        self,
        *,
        state: AssetExtractionCandidateState,
        candidate: AssetCandidate,
    ) -> bool:
        """Resolve and add or merge one asset candidate."""

        resolved_candidate = self._resolve_candidate(
            base_url=state.base_url,
            candidate=candidate,
        )
        if resolved_candidate is None:
            return False

        return self._deduper.add_or_merge(
            results=state.results,
            seen=state.seen,
            candidate=resolved_candidate,
        )

    def _resolve_candidate(
        self,
        *,
        base_url: str,
        candidate: AssetCandidate,
    ) -> AssetCandidate | None:
        raw_url = candidate.url.strip()
        if not raw_url:
            return None

        if _SRCSET_DESCRIPTOR_ARTIFACT_PATTERN.fullmatch(raw_url):
            return None

        resolved_url = self._url_candidate_resolution.resolve(
            base_url=base_url,
            candidate=raw_url,
        )
        if resolved_url is None:
            return None

        return replace(
            candidate,
            url=resolved_url,
            metadata=dict(candidate.metadata),
        )


__all__ = ["AssetCandidateResolution"]
