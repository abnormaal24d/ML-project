"""HTML and link attribute inference for asset kind hints."""

from __future__ import annotations

from crawler.classification.media_kind import MediaKind
from crawler.classification.media_kind_registry import match_mime_type

IGNORED_LINK_RELS = frozenset(
    {
        "canonical",
        "dns-prefetch",
        "manifest",
        "modulepreload",
        "preconnect",
    }
)
ICON_LINK_RELS = frozenset(
    {"icon", "apple-touch-icon", "apple-touch-icon-precomposed", "image_src"}
)
FEED_MIME_TYPES = frozenset(
    {
        "application/rss+xml",
        "application/atom+xml",
        "application/xml",
        "text/xml",
    }
)
SCRIPT_MIME_TYPES = frozenset(
    {"application/javascript", "application/x-javascript", "text/javascript"}
)
STYLESHEET_MIME_TYPES = frozenset({"text/css", "text/x-scss", "text/x-less"})
PRELOAD_ASSET_KINDS = {
    "image": MediaKind.IMAGE.value,
    "audio": MediaKind.AUDIO.value,
    "video": MediaKind.VIDEO.value,
}
META_IMAGE_KEYS = frozenset(
    {
        "og:image",
        "og:image:url",
        "twitter:image",
        "twitter:image:src",
    }
)
META_VIDEO_KEYS = frozenset(
    {
        "og:video",
        "og:video:url",
        "og:video:secure_url",
        "twitter:player",
        "twitter:player:stream",
    }
)
META_AUDIO_KEYS = frozenset(
    {
        "og:audio",
        "og:audio:url",
    }
)


def normalize_rel_values(*, rel: object) -> set[str]:
    """Normalize an HTML rel attribute into a lower-cased token set."""
    if rel is None:
        return set()
    if isinstance(rel, str):
        return {
            token.strip().lower() for token in rel.split() if token.strip()
        }
    if isinstance(rel, (list, tuple, set)):
        values: set[str] = set()
        for item in rel:
            if isinstance(item, str) and item.strip():
                values.add(item.strip().lower())
        return values
    return set()


def infer_meta_kind_from_names(candidates: set[str]) -> str | None:
    if candidates.intersection(META_IMAGE_KEYS):
        return MediaKind.IMAGE.value
    if candidates.intersection(META_VIDEO_KEYS):
        return MediaKind.VIDEO.value
    if candidates.intersection(META_AUDIO_KEYS):
        return MediaKind.AUDIO.value
    return None


def infer_link_kind_from_attributes(
    *,
    rel_values: set[str],
    media_type: object,
    preload_as: object,
    include_icon_link_assets: bool,
    include_stylesheets_as_documents: bool,
    include_script_assets: bool,
    include_font_assets: bool,
) -> str | None:
    if not rel_values:
        return None

    if rel_values.intersection(IGNORED_LINK_RELS):
        return None

    if isinstance(media_type, str):
        lowered = media_type.lower()

        if "oembed" in lowered:
            return None

        if lowered.startswith("image/"):
            if lowered == "image/svg+xml":
                return None
            if include_icon_link_assets:
                return MediaKind.IMAGE.value
            if "icon" not in rel_values and "image_src" not in rel_values:
                return MediaKind.IMAGE.value
            return None

        matched_kind = match_mime_type(lowered)
        if matched_kind in {MediaKind.AUDIO, MediaKind.VIDEO}:
            return matched_kind.value

        if lowered in STYLESHEET_MIME_TYPES:
            return (
                MediaKind.DOCUMENT.value
                if include_stylesheets_as_documents
                else None
            )

        if lowered in SCRIPT_MIME_TYPES:
            return MediaKind.DOCUMENT.value if include_script_assets else None

        if lowered.startswith("font/"):
            return MediaKind.DOCUMENT.value if include_font_assets else None

        if lowered in FEED_MIME_TYPES:
            return MediaKind.FEED.value

    if rel_values.intersection(ICON_LINK_RELS):
        return MediaKind.IMAGE.value if include_icon_link_assets else None

    if "stylesheet" in rel_values:
        return (
            MediaKind.DOCUMENT.value
            if include_stylesheets_as_documents
            else None
        )

    if "preload" in rel_values:
        return infer_preload_link_kind(
            preload_as=preload_as,
            include_stylesheets_as_documents=include_stylesheets_as_documents,
            include_script_assets=include_script_assets,
            include_font_assets=include_font_assets,
        )

    if rel_values.intersection({"alternate", "enclosure"}):
        if isinstance(media_type, str) and "oembed" in media_type.lower():
            return None
        if isinstance(media_type, str):
            lowered = media_type.lower()
            matched_kind = match_mime_type(lowered)
            if matched_kind is not None:
                return matched_kind.value
            if lowered in FEED_MIME_TYPES:
                return MediaKind.FEED.value
        return MediaKind.FEED.value

    return None


def infer_preload_link_kind(
    *,
    preload_as: object,
    include_stylesheets_as_documents: bool,
    include_script_assets: bool,
    include_font_assets: bool,
) -> str | None:
    if not isinstance(preload_as, str):
        return None

    lowered = preload_as.lower()
    if lowered in PRELOAD_ASSET_KINDS:
        return PRELOAD_ASSET_KINDS[lowered]
    if lowered == "style":
        return (
            MediaKind.DOCUMENT.value
            if include_stylesheets_as_documents
            else None
        )
    if lowered == "script":
        return MediaKind.DOCUMENT.value if include_script_assets else None
    if lowered == "font":
        return MediaKind.DOCUMENT.value if include_font_assets else None
    return None
