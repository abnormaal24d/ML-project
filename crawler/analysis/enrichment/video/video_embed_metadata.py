"""Build analysis metadata from embed-only video fetch payloads."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crawler.fetching.results.result import FetchResult


def embed_metadata(*, result: FetchResult) -> dict[str, Any]:
    try:
        decoded = result.read_body_required().decode("utf-8", errors="replace")
        payload = json.loads(decoded)
    except (OSError, json.JSONDecodeError, UnicodeError, ValueError):
        payload = {}

    metadata: dict[str, Any] = {
        "asset_fetch_mode": "embed_metadata",
        "metadata_status": "embed_metadata",
    }
    if isinstance(payload, dict):
        task_context = payload.get("task_context")
        if isinstance(task_context, dict):
            metadata.update(
                {str(key): value for key, value in task_context.items()}
            )
        for key in (
            "embed_url",
            "embed_host",
            "source_page_url",
            "parent_title",
            "parent_text_preview",
            "discovery_reason",
            "requested_kind",
            "final_url",
        ):
            value = payload.get(key)
            if value is not None:
                metadata[key] = value
    metadata.setdefault("embed_url", result.final_url)
    return metadata
