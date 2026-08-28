"""Build normalized crawl task context payloads."""

from __future__ import annotations

from collections.abc import Mapping

from crawler.crawl_tasks.crawl_task_context import CrawlTaskContext


def _normalize_context_text(
    value: object,
    *,
    max_chars: int,
) -> str | None:
    """Normalize free-form discovery text into one compact line."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    return text[:max_chars]


def _coerce_int(value: object) -> int | None:
    """Return the value as an integer when possible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text:
        return None

    try:
        return int(text)
    except ValueError:
        return None


def _coerce_float(value: object) -> float | None:
    """Return the value as a float when possible."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def _coerce_bool(value: object) -> bool | None:
    """Return the value as a boolean when explicit."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_base_context_fields(
    value: Mapping[str, object],
) -> dict[str, object | None]:
    """Parse core HTML and media hint fields from one context mapping."""
    return {
        "tag_name": _normalize_context_text(
            value.get("tag_name"), max_chars=64
        ),
        "source_tag": _normalize_context_text(
            value.get("source_tag"),
            max_chars=64,
        ),
        "source_attribute": _normalize_context_text(
            value.get("source_attribute"),
            max_chars=64,
        ),
        "text_hint": _normalize_context_text(
            value.get("text_hint"), max_chars=280
        ),
        "surrounding_text": _normalize_context_text(
            value.get("surrounding_text"),
            max_chars=500,
        ),
        "mime_hint": _normalize_context_text(
            value.get("mime_hint"), max_chars=128
        ),
        "alt_text": _normalize_context_text(
            value.get("alt_text"), max_chars=280
        ),
        "caption_text": _normalize_context_text(
            value.get("caption_text"),
            max_chars=500,
        ),
        "width": _coerce_int(value.get("width")),
        "height": _coerce_int(value.get("height")),
        "duration_seconds": _coerce_float(value.get("duration_seconds")),
        "embed_url": _normalize_context_text(
            value.get("embed_url"), max_chars=500
        ),
        "embed_host": _normalize_context_text(
            value.get("embed_host"), max_chars=253
        ),
        "poster_url": _normalize_context_text(
            value.get("poster_url"), max_chars=500
        ),
    }


def _parse_logging_fields(
    value: Mapping[str, object],
) -> dict[str, object | None]:
    """Parse observability fields from one context mapping."""
    return {
        "discovery_reason": _normalize_context_text(
            value.get("discovery_reason"),
            max_chars=128,
        ),
        "selection_reason": _normalize_context_text(
            value.get("selection_reason"),
            max_chars=128,
        ),
        "admission_reason": _normalize_context_text(
            value.get("admission_reason"),
            max_chars=128,
        ),
        "asset_discovery_stage": _normalize_context_text(
            value.get("asset_discovery_stage"),
            max_chars=64,
        ),
        "boilerplate_reason": _normalize_context_text(
            value.get("boilerplate_reason"),
            max_chars=128,
        ),
    }


def _parse_coverage_fields(
    value: Mapping[str, object],
) -> dict[str, object | None]:
    """Parse page-coverage fields from one context mapping."""
    return {
        "source_page_url": _normalize_context_text(
            value.get("source_page_url"),
            max_chars=500,
        ),
        "source_page_depth": _coerce_int(value.get("source_page_depth")),
        "parent_text_hash": _normalize_context_text(
            value.get("parent_text_hash"),
            max_chars=128,
        ),
        "parent_text_preview": _normalize_context_text(
            value.get("parent_text_preview"),
            max_chars=500,
        ),
        "parent_title": _normalize_context_text(
            value.get("parent_title"),
            max_chars=280,
        ),
        "html_dom_path": _normalize_context_text(
            value.get("html_dom_path"),
            max_chars=280,
        ),
    }


def _parse_workflow_fields(
    value: Mapping[str, object],
) -> dict[str, object | None]:
    """Parse workflow and asset fields from one context mapping."""
    return {
        "asset_quality_score": _coerce_float(value.get("asset_quality_score")),
        "asset_fetch_mode": _normalize_context_text(
            value.get("asset_fetch_mode"),
            max_chars=64,
        ),
        "candidate_strength": _coerce_float(value.get("candidate_strength")),
        "media_identity": _normalize_context_text(
            value.get("media_identity"),
            max_chars=768,
        ),
        "asset_rejection_reason": _normalize_context_text(
            value.get("asset_rejection_reason"),
            max_chars=128,
        ),
        "is_boilerplate_asset": _coerce_bool(
            value.get("is_boilerplate_asset")
        ),
    }


def coerce_crawl_task_context(value: object) -> CrawlTaskContext | None:
    """Return a normalized crawl task context when any fields are present."""
    if value is None:
        return None
    if isinstance(value, CrawlTaskContext):
        return value if value.to_dict() else None
    if not isinstance(value, Mapping):
        return None

    fields = {
        **_parse_base_context_fields(value),
        **_parse_logging_fields(value),
        **_parse_coverage_fields(value),
        **_parse_workflow_fields(value),
    }

    if not any(fields.values()):
        return None

    is_boilerplate_asset = fields.get("is_boilerplate_asset")
    return CrawlTaskContext(
        tag_name=_normalize_context_text(fields.get("tag_name"), max_chars=64),
        source_tag=_normalize_context_text(
            fields.get("source_tag"), max_chars=64
        ),
        source_attribute=_normalize_context_text(
            fields.get("source_attribute"),
            max_chars=64,
        ),
        text_hint=_normalize_context_text(
            fields.get("text_hint"), max_chars=280
        ),
        surrounding_text=_normalize_context_text(
            fields.get("surrounding_text"),
            max_chars=500,
        ),
        mime_hint=_normalize_context_text(
            fields.get("mime_hint"), max_chars=128
        ),
        alt_text=_normalize_context_text(
            fields.get("alt_text"), max_chars=280
        ),
        caption_text=_normalize_context_text(
            fields.get("caption_text"),
            max_chars=500,
        ),
        width=_coerce_int(fields.get("width")),
        height=_coerce_int(fields.get("height")),
        duration_seconds=_coerce_float(fields.get("duration_seconds")),
        asset_quality_score=_coerce_float(fields.get("asset_quality_score")),
        asset_fetch_mode=_normalize_context_text(
            fields.get("asset_fetch_mode"),
            max_chars=64,
        ),
        asset_rejection_reason=_normalize_context_text(
            fields.get("asset_rejection_reason"),
            max_chars=128,
        ),
        embed_url=_normalize_context_text(
            fields.get("embed_url"), max_chars=500
        ),
        embed_host=_normalize_context_text(
            fields.get("embed_host"), max_chars=253
        ),
        poster_url=_normalize_context_text(
            fields.get("poster_url"), max_chars=500
        ),
        parent_text_hash=_normalize_context_text(
            fields.get("parent_text_hash"),
            max_chars=128,
        ),
        parent_text_preview=_normalize_context_text(
            fields.get("parent_text_preview"),
            max_chars=500,
        ),
        parent_title=_normalize_context_text(
            fields.get("parent_title"), max_chars=280
        ),
        html_dom_path=_normalize_context_text(
            fields.get("html_dom_path"), max_chars=280
        ),
        asset_discovery_stage=_normalize_context_text(
            fields.get("asset_discovery_stage"),
            max_chars=64,
        ),
        source_page_url=_normalize_context_text(
            fields.get("source_page_url"),
            max_chars=500,
        ),
        discovery_reason=_normalize_context_text(
            fields.get("discovery_reason"),
            max_chars=128,
        ),
        candidate_strength=_coerce_float(fields.get("candidate_strength")),
        selection_reason=_normalize_context_text(
            fields.get("selection_reason"),
            max_chars=128,
        ),
        admission_reason=_normalize_context_text(
            fields.get("admission_reason"),
            max_chars=128,
        ),
        media_identity=_normalize_context_text(
            fields.get("media_identity"),
            max_chars=768,
        ),
        source_page_depth=_coerce_int(fields.get("source_page_depth")),
        is_boilerplate_asset=(
            is_boilerplate_asset
            if isinstance(is_boilerplate_asset, bool)
            else None
        ),
        boilerplate_reason=_normalize_context_text(
            fields.get("boilerplate_reason"),
            max_chars=128,
        ),
    )
