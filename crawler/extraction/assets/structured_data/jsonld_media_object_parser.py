"""JSON-LD media object traversal and candidate iteration."""

from __future__ import annotations

import json
from typing import Any, cast

from crawler.classification.media_kind_registry import (
    match_extension,
)
from crawler.extraction.assets.candidate.asset_extraction_records import (
    clean_string,
)
from crawler.extraction.assets.html.html_asset_element_reader import (
    element_dom_path,
)

_VIDEO_JSONLD_KEYS = frozenset(
    {
        "video",
        "videourl",
        "embedurl",
        "trailer",
    }
)
_AUDIO_JSONLD_KEYS = frozenset(
    {
        "audio",
        "audioobject",
        "audiourl",
        "associatedmedia",
    }
)
_KIND_SCOPED_URL_KEYS = frozenset(
    {
        "contenturl",
        "embedurl",
        "encoding",
    }
)
_IMAGE_JSONLD_KEYS = frozenset(
    {
        "image",
        "thumbnail",
        "thumbnailurl",
        "primaryimageofpage",
    }
)
_BASED_ON_JSONLD_KEYS = frozenset({"isbasedon"})
_TRANSCRIPT_JSONLD_KEYS = frozenset(
    {
        "transcript",
        "caption",
        "captions",
        "subtitle",
        "subtitles",
    }
)


def parse_jsonld_payload(text: str) -> object | None:
    stripped = text.strip()
    if not stripped:
        return None

    try:
        return cast("object", json.loads(stripped))
    except json.JSONDecodeError:
        return None


def iter_jsonld_media_candidates(
    payload: object,
    *,
    inherited_kind: str | None = None,
) -> tuple[tuple[str, str, dict[str, object]], ...]:
    results: list[tuple[str, str, dict[str, object]]] = []

    def visit(
        value: object,
        current_kind: str | None,
        current_metadata: dict[str, object],
    ) -> None:
        if isinstance(value, dict):
            next_kind = jsonld_type_kind(value.get("@type")) or current_kind
            node_metadata = {
                **current_metadata,
                **jsonld_media_metadata(value=value, kind=next_kind),
            }
            for raw_key, raw_child in value.items():
                key = str(raw_key).strip().lower()
                key_kind = jsonld_key_kind(
                    key=key,
                    inherited_kind=next_kind,
                )
                if key_kind is not None:
                    _collect_jsonld_key_urls(
                        key=key,
                        value=raw_child,
                        kind=key_kind,
                        metadata=node_metadata,
                        results=results,
                    )
                visit(raw_child, key_kind or next_kind, node_metadata)
            return

        if isinstance(value, list):
            for item in value:
                visit(item, current_kind, current_metadata)

    visit(payload, inherited_kind, {})
    return tuple(results)


def jsonld_media_metadata(
    *,
    value: dict[object, object],
    kind: str | None,
) -> dict[str, object]:
    if kind not in {"image", "audio", "video"}:
        return {}

    name = clean_string(value.get("name"))
    description = clean_string(value.get("description"))
    upload_date = clean_string(value.get("uploadDate"))
    metadata: dict[str, object] = {}

    if name:
        metadata["jsonld_name"] = name
    if description:
        metadata["jsonld_description"] = description
    if upload_date:
        metadata["jsonld_upload_date"] = upload_date

    text_hint = description or name
    if text_hint:
        metadata["text_hint"] = text_hint
        metadata["caption_text"] = text_hint

    return metadata


def jsonld_type_kind(value: object) -> str | None:
    values = value if isinstance(value, list) else (value,)
    for item in values:
        text = str(item).strip().lower()
        if text.endswith("imageobject") or text == "image":
            return "image"
        if text.endswith("videoobject") or text == "video":
            return "video"
        if (
            text.endswith("audioobject")
            or text == "audio"
            or text.endswith("/audio")
        ):
            return "audio"

    return None


def jsonld_key_kind(
    *,
    key: str,
    inherited_kind: str | None,
) -> str | None:
    if key in _KIND_SCOPED_URL_KEYS and inherited_kind is not None:
        return inherited_kind

    if key in _BASED_ON_JSONLD_KEYS and inherited_kind is not None:
        return inherited_kind

    if key in _IMAGE_JSONLD_KEYS:
        return "image"

    if key in _TRANSCRIPT_JSONLD_KEYS:
        return "document"

    if key in _VIDEO_JSONLD_KEYS:
        return "video"
    if key in _AUDIO_JSONLD_KEYS:
        return "audio"
    return None


def _collect_jsonld_key_urls(
    *,
    key: str,
    value: object,
    kind: str,
    metadata: dict[str, object],
    results: list[tuple[str, str, dict[str, object]]],
) -> None:
    _collect_jsonld_urls(
        value=value,
        kind=kind,
        metadata=metadata,
        results=results,
    )


def _collect_jsonld_urls(
    *,
    value: Any,
    kind: str,
    metadata: dict[str, object],
    results: list[tuple[str, str, dict[str, object]]],
) -> None:
    if isinstance(value, str):
        candidate = clean_string(value)
        if candidate:
            matched_kind = match_extension(candidate)
            results.append(
                (
                    candidate,
                    matched_kind.value if matched_kind is not None else kind,
                    metadata,
                )
            )
        return

    if isinstance(value, list):
        for item in value:
            _collect_jsonld_urls(
                value=item,
                kind=kind,
                metadata=metadata,
                results=results,
            )
        return

    if isinstance(value, dict):
        url = (
            value.get("contentUrl")
            or value.get("contenturl")
            or value.get("embedUrl")
            or value.get("embedurl")
            or value.get("thumbnailUrl")
            or value.get("thumbnailurl")
            or value.get("url")
            or value.get("@id")
        )
        candidate = clean_string(url)
        if candidate:
            matched_kind = match_extension(candidate)
            results.append(
                (
                    candidate,
                    matched_kind.value if matched_kind is not None else kind,
                    metadata,
                )
            )


def build_jsonld_script_context(
    *,
    element: Any,
    parent_text_metadata: dict[str, object],
) -> dict[str, object]:
    metadata = dict(parent_text_metadata)
    return {
        "source_tag": "script",
        "alt_text": None,
        "caption_text": None,
        "surrounding_text": None,
        "mime_hint": None,
        "width": None,
        "height": None,
        "duration_seconds": None,
        "metadata": {
            **metadata,
            "html_dom_path": element_dom_path(element=element),
        },
    }
