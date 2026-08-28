"""Media landing-page and candidate-page classification for discovery."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task_context import CrawlTaskContext

# Audio-related page signals (used for candidate classification in discovery)
_AUDIO_PAGE_TOKENS = frozenset(
    (
        "audio",
        "podcast",
        "podcasts",
        "rss",
        "episode",
        "listen",
        "recording",
        "transcript",
        "mp3",
    )
)

# Document / data / publication page signals
_DOCUMENT_PAGE_TOKENS = frozenset(
    (
        "abstract",
        "archive",
        "citation",
        "circular",
        "data-release",
        "datarelease",
        "dataset",
        "doi",
        "download",
        "downloads",
        "fact-sheet",
        "factsheet",
        "full-text",
        "guide",
        "handbook",
        "manual",
        "open-file",
        "openfile",
        "paper",
        "papers",
        "pdf",
        "professional-paper",
        "publication",
        "publications",
        "pubs",
        "repository",
        "report",
        "reports",
        "research-report",
        "scientific-investigations",
        "supplemental",
        "supplementary",
        "technical-report",
    )
)

# Video / stream page signals
_VIDEO_PAGE_TOKENS = frozenset(
    (
        "video",
        "videos",
        "watch",
        "stream",
        "playlist",
        "embed",
        "broadcast",
        "webcast",
        "livestream",
    )
)

# Combined signal tokens for media pages
_MEDIA_PAGE_SIGNAL_TOKENS = (
    _AUDIO_PAGE_TOKENS | _DOCUMENT_PAGE_TOKENS | _VIDEO_PAGE_TOKENS
)

_VIDEO_PAGE_HOSTS = frozenset({"plus.nasa.gov"})
_VIDEO_PAGE_PATH_FRAGMENTS = frozenset(
    (
        "/broadcast",
        "/gallery/videos",
        "/live",
        "/livestream",
        "/media-gallery/videos",
        "/video",
        "/videos",
        "/watch",
        "/webcast",
        "/multimedia-gallery/videos",
        "/streaming-services/",
    )
)

# Broader context signals that may indicate media content
_MEDIA_CONTEXT_SIGNAL_TOKENS = frozenset(
    (
        "audio",
        "broadcast",
        "caption",
        "document",
        "episode",
        "gallery",
        "image",
        "listen",
        "livestream",
        "multimedia",
        "pdf",
        "photo",
        "podcast",
        "recording",
        "stream",
        "subtitle",
        "transcript",
        "video",
        "watch",
    )
)

_DECORATIVE_ASSET_SIGNAL_TOKENS = (
    "avatar",
    "badge",
    "banner",
    "button",
    "favicon",
    "icon",
    "logo",
    "share",
    "social",
    "sprite",
    "thumbnail",
    "thumb",
)

_CANDIDATE_KINDS = ("document", "audio", "video")
_KIND_TOKENS: dict[str, Collection[str]] = {
    "document": _DOCUMENT_PAGE_TOKENS,
    "audio": _AUDIO_PAGE_TOKENS,
    "video": _VIDEO_PAGE_TOKENS,
}


@dataclass(frozen=True, slots=True)
class CandidatePageClassification:
    """Classification result for a page URL exploration candidate."""

    kind: str | None
    confidence: float
    reason: str


def classify_candidate_page(
    url: str,
    *,
    context: CrawlTaskContext | None = None,
    candidate_kinds: Collection[str] | None = None,
) -> CandidatePageClassification:
    """Classify a page within the requested candidate kinds."""

    allowed_kinds = (
        set(_CANDIDATE_KINDS)
        if candidate_kinds is None
        else {
            str(kind).strip().lower()
            for kind in candidate_kinds
            if str(kind).strip().lower() in _CANDIDATE_KINDS
        }
    )
    best_kind: str | None = None
    best_confidence = 0.0
    best_reason = "no_candidate_signal"

    for kind in _CANDIDATE_KINDS:
        if kind not in allowed_kinds:
            continue
        confidence, reason = _score_candidate_kind(
            url=url,
            context=context,
            kind=kind,
            tokens=_KIND_TOKENS[kind],
        )
        if confidence > best_confidence:
            best_kind = kind if confidence >= 0.45 else None
            best_confidence = confidence
            best_reason = reason

    if best_confidence < 0.45:
        return CandidatePageClassification(
            kind=None,
            confidence=best_confidence,
            reason=(
                best_reason if best_confidence > 0.0 else "no_candidate_signal"
            ),
        )
    return CandidatePageClassification(
        kind=best_kind,
        confidence=best_confidence,
        reason=best_reason,
    )


def is_video_candidate_page_url(
    url: str,
    *,
    context: CrawlTaskContext | None = None,
) -> bool:
    """Return whether a page URL is a useful video exploration candidate."""

    return classify_candidate_page(url, context=context).kind == "video"


def is_audio_candidate_page_url(
    url: str,
    *,
    context: CrawlTaskContext | None = None,
) -> bool:
    """Return whether a page URL is useful for audio exploration."""

    return classify_candidate_page(url, context=context).kind == "audio"


def is_document_candidate_page_url(
    url: str,
    *,
    context: CrawlTaskContext | None = None,
) -> bool:
    """Return whether a page URL is useful for document exploration."""

    return (
        candidate_page_kind_confidence(
            url,
            kind="document",
            context=context,
        )
        >= 0.45
    )


def candidate_page_kind_confidence(
    url: str,
    *,
    kind: str,
    context: CrawlTaskContext | None = None,
) -> float:
    """Return independent candidate-page confidence for one target kind."""

    normalized_kind = str(kind).strip().lower()
    tokens = _KIND_TOKENS.get(normalized_kind)
    if tokens is None:
        return 0.0

    confidence, _reason = _score_candidate_kind(
        url=url,
        context=context,
        kind=normalized_kind,
        tokens=tokens,
    )
    return confidence


def is_multimodal_media_page_url(
    url: str,
    *,
    context: CrawlTaskContext | None = None,
) -> bool:
    """Return whether a page URL looks like a media landing page."""

    path = urlparse(url).path.lower()
    segments = tuple(segment for segment in path.split("/") if segment)
    path_matches = any(
        token in segment
        for segment in segments
        for token in _MEDIA_PAGE_SIGNAL_TOKENS
    )
    if path_matches:
        return True

    return multimodal_context_signal_score(context=context) >= 2


def _score_candidate_kind(
    *,
    url: str,
    context: CrawlTaskContext | None,
    kind: str,
    tokens: Collection[str],
) -> tuple[float, str]:
    if kind == "video":
        host_reason = _video_host_or_path_reason(url=url)
        if host_reason is not None:
            return 0.95, host_reason

    parsed = urlparse(url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    haystack = f"{path} {query}"

    path_hits = [token for token in tokens if token in haystack]
    if path_hits:
        confidence = min(1.0, 0.55 + 0.1 * len(path_hits))
        return confidence, f"{kind}_url_token:{path_hits[0]}"

    combined = _combined_context_text(context=context) if context else ""
    context_hits = [token for token in tokens if token in combined]
    if context_hits:
        confidence = min(0.9, 0.45 + 0.08 * len(context_hits))
        return confidence, f"{kind}_context_token:{context_hits[0]}"

    return 0.0, f"{kind}_no_signal"


def _video_host_or_path_reason(*, url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path.lower()
    if host in _VIDEO_PAGE_HOSTS:
        return f"video_host:{host}"
    for fragment in _VIDEO_PAGE_PATH_FRAGMENTS:
        if fragment in path:
            return f"video_path_fragment:{fragment.strip('/')}"
    return None


def multimodal_context_signal_score(
    *,
    context: CrawlTaskContext | None,
) -> int:
    """Return a compact multimodal-signal score from discovery context."""

    if context is None:
        return 0

    score = 0
    tag_name = (context.tag_name or "").lower()
    mime_hint = (context.mime_hint or "").lower()
    combined_text = _combined_context_text(context=context)

    if tag_name in {"audio", "video", "track"}:
        score += 2
    elif tag_name in {"source", "img", "figure"}:
        score += 1

    if mime_hint.startswith(("audio/", "video/", "image/", "text/")):
        score += 1
    elif mime_hint == "application/pdf":
        score += 1

    token_hits = sum(
        1 for token in _MEDIA_CONTEXT_SIGNAL_TOKENS if token in combined_text
    )
    if token_hits > 0:
        score += min(2, token_hits)

    return score


def is_probably_decorative_asset_context(
    *,
    context: CrawlTaskContext | None,
) -> bool:
    """Return whether discovery context points to decorative assets."""

    if context is None:
        return False

    combined_text = _combined_context_text(context=context)
    return any(
        token in combined_text for token in _DECORATIVE_ASSET_SIGNAL_TOKENS
    )


def _combined_context_text(*, context: CrawlTaskContext) -> str:
    return " ".join(
        part.lower()
        for part in (
            context.text_hint,
            context.surrounding_text,
            context.mime_hint,
            getattr(context, "caption_text", None),
            getattr(context, "alt_text", None),
            getattr(context, "parent_title", None),
            getattr(context, "asset_discovery_stage", None),
            getattr(context, "source_tag", None),
            getattr(context, "source_attribute", None),
        )
        if part
    )
