"""Detect video fetch modes and partial metadata probe states."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.fetching.results.result import FetchResult

_logger = logging.getLogger(__name__)


def is_partial_video_probe(*, result: FetchResult) -> bool:
    """Return True if the payload was only a partial (truncated) probe."""
    payload = result.payload
    is_partial = bool(payload is not None and payload.truncated)
    if is_partial:
        _logger.debug(
            "video_partial_probe_detected",
            extra={
                "url": getattr(result, "final_url", None),
                "fetch_mode": getattr(
                    getattr(payload, "fetch_mode", None),
                    "__str__",
                    lambda: None,
                )(),
            },
        )
    return is_partial


def is_head_only_video_probe(*, result: FetchResult) -> bool:
    """Return True for head-only responses that were oversized (no body fetched)."""
    payload = result.payload
    if payload is None:
        oversized = _read_content_length_header(result=result) is not None
    else:
        oversized = (
            str(payload.fetch_mode or "").strip().lower()
            == "head_only_oversized"
        )
    if oversized:
        _logger.debug(
            "video_head_only_probe_detected",
            extra={"url": getattr(result, "final_url", None)},
        )
    return oversized


def is_embed_video_metadata(*, result: FetchResult) -> bool:
    """Return True when the fetch result is an embed metadata payload only."""
    payload = result.payload
    is_embed = bool(
        payload is not None
        and str(payload.fetch_mode or "").strip().lower() == "embed_metadata"
    )
    if is_embed:
        _logger.debug(
            "video_embed_metadata_detected",
            extra={"url": getattr(result, "final_url", None)},
        )
    return is_embed


def is_oversized_video_metadata_probe(*, result: FetchResult) -> bool:
    """Return True when a metadata-only probe was oversized (source larger than observed)."""
    payload = result.payload
    if payload is None:
        return False

    fetch_mode = str(payload.fetch_mode or "").strip().lower()
    if fetch_mode not in {"metadata_only", "metadata_probe"}:
        return False

    source_content_length = payload.source_content_length
    if source_content_length is None:
        return False
    oversized = int(source_content_length) > int(
        payload.observed_bytes or payload.byte_size
    )
    if oversized:
        _logger.debug(
            "video_oversized_metadata_probe_detected",
            extra={
                "url": getattr(result, "final_url", None),
                "source_content_length": source_content_length,
            },
        )
    return oversized


def _read_content_length_header(*, result: FetchResult) -> int | None:
    """Extract Content-Length from headers, return None on parse failure (with warning)."""
    for key, value in result.headers.items():
        if str(key).lower() != "content-length":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            _logger.warning(
                "header_content_length_parse_failed",
                extra={
                    "key": key,
                    "value": value,
                    "url": getattr(result, "final_url", None),
                },
            )
            return None
    return None
