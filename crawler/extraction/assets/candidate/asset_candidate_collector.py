"""Asset candidate collection, rejection, and enrichment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

from crawler.classification.media_kind import MediaKind
from crawler.classification.media_kind_registry import (
    match_extension,
)
from crawler.classification.mime_type_resolver import (
    normalize_mime_type,
)
from crawler.extraction.assets.candidate.asset_candidate_quality import (
    score_asset_candidate,
)
from crawler.extraction.assets.candidate.asset_extraction_records import (
    AssetCandidate,
    AssetCandidateDraft,
    AssetDiscoveryResult,
    AssetExtractionCandidateState,
    AssetKind,
    as_optional_float,
    as_optional_int,
    as_optional_str,
    build_asset_discovery_result,
)

if TYPE_CHECKING:
    from crawler.extraction.assets.candidate.asset_candidate_resolution import (
        AssetCandidateResolution,
    )
    from crawler.extraction.assets.embedded_media.video_embed_detector import (
        VideoEmbedDetector,
    )


class AssetCandidateCollector:
    """Enrich drafts and submit them to the candidate resolver."""

    def __init__(
        self,
        *,
        candidate_resolution: AssetCandidateResolution,
        embed_detector: VideoEmbedDetector,
    ) -> None:
        self._candidate_resolution = candidate_resolution
        self._embed_detector = embed_detector

    def resolve_kind(
        self,
        *,
        configured_kind: str | None,
        raw_url: str,
        source_tag: str | None = None,
        source_attribute: str | None = None,
        context: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> AssetKind | None:
        inferred = match_extension(raw_url)
        if inferred is MediaKind.IMAGE:
            return "image"
        if inferred is MediaKind.AUDIO:
            return "audio"
        if inferred is MediaKind.VIDEO:
            return "video"
        if inferred is MediaKind.DOCUMENT:
            return "document"

        if self._embed_detector.is_video_embed_url(raw_url):
            return "video"

        if configured_kind == "image":
            return "image"
        if configured_kind == "audio":
            return "audio"
        if configured_kind == "video":
            return "video"

        if configured_kind == "document":
            if is_strong_document_candidate(
                raw_url=raw_url,
                source_tag=source_tag,
                source_attribute=source_attribute,
                context=context,
                metadata=metadata,
            ):
                return "document"
            return None

        return None

    def add(
        self,
        *,
        state: AssetExtractionCandidateState,
        draft: AssetCandidateDraft,
    ) -> None:
        kind = self.resolve_kind(
            configured_kind=draft.kind,
            raw_url=draft.candidate,
            source_tag=draft.source_tag,
            source_attribute=draft.source_attribute,
            context=draft.context,
            metadata=draft.metadata,
        )
        if kind is None:
            if draft.kind == "document":
                state.rejected.append(
                    build_asset_rejection_payload(
                        parent_url=state.base_url,
                        asset_url=draft.candidate,
                        kind="document",
                        source_tag=draft.source_tag,
                        source_attribute=draft.source_attribute,
                        reason=REJECTION_WEAK_DOCUMENT_CANDIDATE,
                    )
                )
            return
        self._store_asset_candidate(state=state, draft=draft, kind=kind)

    def collect_drafts(
        self,
        *,
        drafts: Sequence[AssetCandidateDraft],
        page_url: str,
        max_results: int | None = None,
    ) -> AssetDiscoveryResult:
        """Resolve, dedupe, and group drafts into an immutable discovery result.

        Reference extractors produce drafts only. This method is the single
        admission path for budgets, kind finalization, and preference merging.
        """

        results: list[AssetCandidate] = []
        seen: dict[str, int] = {}
        rejected: list[dict[str, object]] = []
        state = AssetExtractionCandidateState(
            results=results,
            seen=seen,
            rejected=rejected,
            base_url=page_url,
        )
        limit = max_results if max_results is not None else None

        for draft in drafts:
            if limit is not None and len(state.results) >= limit:
                break
            self.add(state=state, draft=draft)

        return build_asset_discovery_result(
            parent_url=page_url,
            candidates=tuple(
                sorted(results, key=score_asset_candidate, reverse=True)
            ),
            rejected=tuple(rejected),
        )

    def _store_asset_candidate(
        self,
        *,
        state: AssetExtractionCandidateState,
        draft: AssetCandidateDraft,
        kind: AssetKind,
    ) -> None:
        metadata = dict(draft.metadata)

        if kind == "image" and is_boilerplate_image_url(draft.candidate):
            state.rejected.append(
                build_asset_rejection_payload(
                    parent_url=state.base_url,
                    asset_url=draft.candidate,
                    kind="image",
                    source_tag=draft.source_tag,
                    source_attribute=draft.source_attribute,
                    reason=REJECTION_BOILERPLATE_IMAGE,
                )
            )
            return

        if kind == "video" and self._should_attach_embed_metadata(
            draft.candidate
        ):
            metadata = {
                **metadata,
                **self._embed_detector.video_embed_metadata(
                    url=draft.candidate
                ),
            }

        metadata.setdefault("source_page_url", state.base_url)
        metadata.setdefault(
            "asset_discovery_stage",
            asset_discovery_stage(
                source_tag=draft.source_tag,
                source_attribute=draft.source_attribute,
            ),
        )
        metadata.setdefault(
            "discovery_reason",
            asset_discovery_reason(
                source_tag=draft.source_tag,
                source_attribute=draft.source_attribute,
                kind=kind,
                metadata=metadata,
            ),
        )
        metadata.setdefault(
            "candidate_strength",
            candidate_strength(
                candidate=draft.candidate,
                kind=kind,
                metadata=metadata,
                source_tag=draft.source_tag,
                source_attribute=draft.source_attribute,
            ),
        )
        metadata.setdefault("source_tag", draft.source_tag)
        metadata.setdefault("source_attribute", draft.source_attribute)
        metadata.setdefault("is_boilerplate_asset", False)

        preview_candidate = AssetCandidate(
            url=draft.candidate,
            kind=kind,
            parent_url=state.base_url,
            source_attribute=draft.source_attribute,
            source_tag=draft.source_tag,
            alt_text=as_optional_str(draft.context.get("alt_text")),
            caption_text=as_optional_str(draft.context.get("caption_text")),
            surrounding_text=as_optional_str(
                draft.context.get("surrounding_text")
            ),
            mime_hint=as_optional_str(draft.context.get("mime_hint")),
            width=as_optional_int(draft.context.get("width")),
            height=as_optional_int(draft.context.get("height")),
            duration_seconds=as_optional_float(
                draft.context.get("duration_seconds")
            ),
            metadata=metadata,
        )
        candidate = replace(
            preview_candidate,
            metadata={
                **preview_candidate.metadata,
                "asset_quality_score": score_asset_candidate(
                    preview_candidate
                ),
            },
        )

        self._candidate_resolution.add_candidate(
            state=state,
            candidate=candidate,
        )

    def _should_attach_embed_metadata(self, candidate: str) -> bool:
        inferred = match_extension(candidate)
        if inferred is MediaKind.VIDEO:
            return False

        return self._embed_detector.is_video_embed_url(candidate)


REJECTION_BOILERPLATE_IMAGE = "boilerplate_image_url"
REJECTION_WEAK_DOCUMENT_CANDIDATE = "weak_document_candidate"
_BOILERPLATE_IMAGE_TOKENS = (
    "logo",
    "favicon",
    "sprite",
    "tracking",
    "pixel",
    "avatar",
    "placeholder",
    "blank",
    "spacer",
)


def build_asset_rejection_payload(
    *,
    parent_url: str,
    asset_url: str,
    reason: str,
    kind: str | None = None,
    source_tag: str | None = None,
    source_attribute: str | None = None,
) -> dict[str, object]:
    """Build a diagnostic rejection payload for skipped asset candidates."""

    payload: dict[str, object] = {
        "parent_url": parent_url,
        "asset_url": asset_url,
        "reason": reason,
    }
    if kind is not None:
        payload["kind"] = kind
    if source_tag is not None:
        payload["source_tag"] = source_tag
    if source_attribute is not None:
        payload["source_attribute"] = source_attribute
    return payload


def is_boilerplate_image_url(url: str) -> bool:
    """Return whether an image URL is likely decorative boilerplate."""

    lowered = url.lower()
    return any(token in lowered for token in _BOILERPLATE_IMAGE_TOKENS)


def asset_discovery_stage(*, source_tag: str, source_attribute: str) -> str:
    if source_attribute == "style":
        return "css_background"
    if source_attribute == "jsonld":
        return "schema_org_json_ld"
    if source_attribute == "inline_json":
        return "inline_player_config"
    if source_attribute in {"srcset", "data-srcset"}:
        return "srcset"
    return f"html_{source_tag}"


def asset_discovery_reason(
    *,
    source_tag: str,
    source_attribute: str,
    kind: str,
    metadata: dict[str, object],
) -> str:
    if source_attribute == "jsonld":
        return "schema_video_object" if kind == "video" else "schema_org_media"
    if source_attribute == "inline_json":
        return "inline_player_config"

    meta_name = str(metadata.get("meta_name") or "").strip().lower()
    if meta_name.startswith("og:video"):
        return "og_video"
    if meta_name.startswith("twitter:player"):
        return "twitter_player"
    if source_tag == "iframe" and kind == "video":
        return "iframe_player"
    if source_tag == "video":
        return (
            "html_video_tag"
            if source_attribute != "poster"
            else "video_poster"
        )
    if source_tag == "source" and kind == "video":
        return "html_video_source"
    if source_tag == "track":
        return "html_track_transcript"
    if source_tag == "audio":
        return "html_audio_tag"
    if source_tag == "img":
        return "html_image_tag"
    if source_attribute == "style":
        return "css_background_image"

    return asset_discovery_stage(
        source_tag=source_tag,
        source_attribute=source_attribute,
    )


def candidate_strength(
    *,
    candidate: str,
    kind: str,
    metadata: dict[str, object],
    source_tag: str,
    source_attribute: str,
) -> float:
    if match_extension(candidate) is MediaKind[kind.upper()]:
        return 1.0
    if source_tag == "track" and kind == "document":
        return 0.9
    if source_attribute in {"jsonld", "inline_json"}:
        return 0.85
    if source_tag == "meta":
        return 0.8
    if metadata.get("asset_fetch_mode") == "embed_metadata":
        return 0.75
    if source_tag in {"iframe", "embed"} and kind == "video":
        return 0.7
    if metadata.get("poster_url"):
        return 0.65
    return 0.5


_DOCUMENT_MIME_HINTS = frozenset(
    {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/json",
        "application/xml",
        "text/csv",
        "text/tab-separated-values",
        "text/vtt",
        "application/zip",
        "application/epub+zip",
    }
)

_DOCUMENT_QUERY_KEYS = frozenset(
    {
        "download",
        "file",
        "filename",
        "format",
        "output",
        "mime",
        "type",
        "content_type",
    }
)

_DOCUMENT_QUERY_VALUES = frozenset(
    {
        "pdf",
        "doc",
        "docx",
        "ppt",
        "pptx",
        "xls",
        "xlsx",
        "csv",
        "tsv",
        "json",
        "xml",
        "zip",
        "epub",
        "vtt",
        "srt",
        "txt",
        "text",
    }
)

_DOCUMENT_PATH_TOKENS = frozenset(
    {
        "/download",
        "/downloads/",
        "/file/",
        "/files/",
        "/document/",
        "/documents/",
        "/publication/",
        "/publications/",
        "/report/",
        "/reports/",
    }
)


def is_strong_document_candidate(
    *,
    raw_url: str,
    source_tag: str | None,
    source_attribute: str | None,
    context: Mapping[str, object] | None,
    metadata: Mapping[str, object] | None,
) -> bool:
    """Return whether a configured document candidate is strong enough.

    This blocks weak schema.org/jsonld text fields such as NASA person/name
    URLs from becoming document crawl tasks, while still accepting real
    document files, download endpoints, transcript tracks, and explicit
    document MIME hints.
    """

    if match_extension(raw_url) is MediaKind.DOCUMENT:
        return True

    mime_hint = _first_text_value(
        context,
        metadata,
        keys=("mime_hint", "content_type", "contentType", "encodingFormat"),
    )
    if _looks_like_document_mime(mime_hint):
        return True

    normalized_source_tag = str(source_tag or "").strip().lower()
    normalized_source_attribute = str(source_attribute or "").strip().lower()

    if normalized_source_tag == "track":
        return True

    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        return False

    path = (parsed.path or "").strip().lower()
    query = parse_qs(parsed.query, keep_blank_values=True)

    if not path and not query:
        return False

    if any(token in path for token in _DOCUMENT_PATH_TOKENS):
        return True

    for key, values in query.items():
        normalized_key = str(key).strip().lower()
        normalized_values = {
            str(value).strip().lower()
            for value in values
            if str(value).strip()
        }

        if normalized_key in _DOCUMENT_QUERY_KEYS:
            if not normalized_values:
                return True
            if normalized_values & _DOCUMENT_QUERY_VALUES:
                return True

    if normalized_source_attribute == "jsonld":
        return False

    return False


def _looks_like_document_mime(value: str | None) -> bool:
    if not value:
        return False

    normalized = normalize_mime_type(value)
    if normalized in _DOCUMENT_MIME_HINTS:
        return True

    return normalized is not None and normalized.startswith("application/")


def _first_text_value(
    *mappings: Mapping[str, object] | None,
    keys: tuple[str, ...],
) -> str | None:
    for mapping in mappings:
        if not mapping:
            continue
        for key in keys:
            value = mapping.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return None
