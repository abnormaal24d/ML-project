"""Build crawl tasks from discovered page links and embedded assets."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import (
    SplitResult,
    parse_qsl,
    unquote,
    urlencode,
    urlsplit,
    urlunsplit,
)

from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.discovery.task_identity import (
    discovered_task_identity_from_parts,
)
from crawler.extraction.assets.candidate.asset_extraction_records import (
    AssetDiscoveryResult,
)
from crawler.extraction.candidates.url_candidate_resolution import (
    ExtractionCandidate,
)
from logger.project_logger import ProjectLogger
from shared.runtime_primitives import IdGenerator

if TYPE_CHECKING:
    from config.collection.extraction import UrlExtractorSettings
    from crawler.extraction.extensions_detector import ExtensionDetector


_SITEMAP_SOURCE_TYPES = frozenset(
    {
        "sitemap_page",
        "sitemap_reference",
    }
)

_ASSET_SOURCE_TYPE = "embedded_asset"

_MEDIA_FOCUS_KINDS = frozenset(
    {
        "image",
        "audio",
        "video",
    }
)

_ASSET_GROUP_ORDER = (
    "image",
    "audio",
    "video",
    "document",
)

_EMAIL_ADDRESS_RE = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w.-])",
    re.IGNORECASE,
)


class DiscoveryTaskBuilder:
    """Build discovered crawl tasks from page links and asset results."""

    def __init__(
        self,
        *,
        settings: UrlExtractorSettings,
        extension_detector: ExtensionDetector,
        id_generator: IdGenerator,
        logger: ProjectLogger,
    ) -> None:
        if settings is None:
            raise ValueError("settings is required")
        if extension_detector is None:
            raise ValueError("extension_detector is required")
        if id_generator is None:
            raise ValueError("id_generator is required")

        self._settings = settings
        self._extension_detector = extension_detector
        self._id_generator = id_generator
        self._logger = logger

    def build_page_tasks(
        self,
        *,
        source_name: str,
        parent_url: str,
        parent_depth: int,
        links: tuple[ExtractionCandidate, ...],
        assets: AssetDiscoveryResult,
        max_tasks: int | None,
        focus_kinds: tuple[str, ...] = (),
        base_url: str | None = None,
    ) -> tuple[CrawlTask, ...]:
        """Return validated and deduplicated tasks for one page discovery pass."""

        # Preserve all semantic fields provided by link extraction. In
        # particular, sitemap_page and sitemap_reference must survive so
        # extension inference cannot convert .xml URLs back into feeds.
        link_candidates = tuple(
            ExtractionCandidate(
                url=candidate.url,
                kind=candidate.kind,
                context=candidate.context,
                asset=candidate.asset,
                source_type=(
                    _normalize_source_type(candidate.source_type)
                    or "discovered_link"
                ),
            )
            for candidate in links
        )

        ordered_assets = ordered_asset_candidates(
            discovery=assets,
            focus_kinds=focus_kinds,
        )

        asset_candidates = tuple(
            ExtractionCandidate(
                url=asset.url,
                kind=asset.kind,
                context=None,
                asset=asset,
                source_type=_ASSET_SOURCE_TYPE,
            )
            for asset in ordered_assets
        )

        candidates = list(link_candidates + asset_candidates)

        if should_extract_assets_first(
            focus_kinds=focus_kinds,
        ):
            candidates.sort(
                key=lambda candidate: candidate.asset is None,
            )

        skip_base_url = base_url or parent_url
        task_parent_url = parent_url or base_url or ""

        seen: set[str] = set()
        tasks: list[CrawlTask] = []

        for candidate in candidates:
            if max_tasks is not None and len(tasks) >= max_tasks:
                break

            source_type = _resolve_source_type(
                candidate=candidate,
            )

            if _is_asset_candidate(
                candidate=candidate,
                source_type=source_type,
            ):
                if not self._settings.include_assets:
                    continue
            else:
                # sitemap_page and sitemap_reference are link-discovery
                # sources and must respect the include_links setting.
                if not self._settings.include_links:
                    continue

            url = str(candidate.url or "").strip()

            if not url:
                continue

            skip_reason = discovered_url_skip_reason(
                url,
                base_url=skip_base_url,
            )

            if skip_reason is not None:
                self._logger.debug(
                    "discovered_url_skipped",
                    url=url,
                    base_url=skip_base_url,
                    source_type=source_type,
                    reason=skip_reason,
                )
                continue

            kind = _resolve_candidate_kind(
                candidate=candidate,
                source_type=source_type,
                url=url,
                extension_detector=self._extension_detector,
            )

            if kind == "feed" and not self._settings.include_feeds:
                continue

            if not mark_seen(
                seen,
                url=url,
                kind=kind,
                source_type=source_type,
            ):
                continue

            context = _candidate_context_payload(
                candidate=candidate,
            )

            if kind not in {"page", "feed"}:
                context.setdefault("source_page_depth", parent_depth)
                context.setdefault("source_page_url", task_parent_url)

            tasks.append(
                CrawlTask.build_discovered(
                    source_name=source_name,
                    url=url,
                    kind=kind,
                    parent_depth=parent_depth,
                    source_type=source_type,
                    parent_url=task_parent_url,
                    context=context,
                    id_generator=self._id_generator,
                )
            )

        return tuple(tasks)


def _resolve_source_type(
    *,
    candidate: ExtractionCandidate,
) -> str:
    """Resolve a stable source type for one extraction candidate."""

    configured = _normalize_source_type(candidate.source_type)

    if configured:
        return configured

    if candidate.asset is not None:
        return _ASSET_SOURCE_TYPE

    return "discovered_link"


def _normalize_source_type(
    value: object,
) -> str:
    """Normalize a candidate source-type value."""

    if value is None:
        return ""

    return str(value).strip().lower()


def _is_asset_candidate(
    *,
    candidate: ExtractionCandidate,
    source_type: str,
) -> bool:
    """Return whether a candidate represents an embedded media asset."""

    return candidate.asset is not None or source_type == _ASSET_SOURCE_TYPE


def _resolve_candidate_kind(
    *,
    candidate: ExtractionCandidate,
    source_type: str,
    url: str,
    extension_detector: ExtensionDetector,
) -> str:
    """Resolve candidate kind without overriding explicit sitemap semantics."""

    explicit_kind = candidate.kind

    if explicit_kind is None and candidate.asset is not None:
        explicit_kind = candidate.asset.kind

    normalized_explicit_kind = _normalize_kind(explicit_kind)

    if normalized_explicit_kind:
        return normalized_explicit_kind

    # Sitemap XML structure is authoritative. A nested sitemap may end in
    # .xml, but it must travel through the page/XML pipeline rather than
    # being reclassified as a feed by extension.
    if source_type in _SITEMAP_SOURCE_TYPES:
        return "page"

    detected_kind = extension_detector.detect_kind(url)

    return _normalize_kind(detected_kind) or "page"


def _normalize_kind(
    value: object,
) -> str:
    """Normalize a crawl-kind value."""

    if value is None:
        return ""

    return str(value).strip().lower()


def _candidate_context_payload(
    *,
    candidate: ExtractionCandidate,
) -> dict[str, object]:
    """Build task context from an asset or link candidate."""

    asset = candidate.asset

    if asset is not None:
        metadata = getattr(
            asset,
            "metadata",
            {},
        )

        quality_score = None

        if isinstance(metadata, Mapping):
            raw_quality_score = metadata.get("asset_quality_score")

            if isinstance(
                raw_quality_score,
                (int, float),
            ):
                quality_score = float(raw_quality_score)

        to_task_context = getattr(
            asset,
            "to_task_context",
            None,
        )

        if callable(to_task_context):
            payload = to_task_context(
                quality_score=quality_score,
            )

            return context_payload(payload)

    return context_payload(candidate.context)


def context_payload(
    context: Any | None,
) -> dict[str, object]:
    """Convert a candidate context into a plain task-context dictionary."""

    if context is None:
        return {}

    to_dict = getattr(
        context,
        "to_dict",
        None,
    )

    if callable(to_dict):
        payload = to_dict()

        if isinstance(payload, Mapping):
            return {str(key): value for key, value in payload.items()}

        return {}

    if isinstance(context, Mapping):
        return {str(key): value for key, value in context.items()}

    return {}


def discovered_url_skip_reason(
    url: str,
    *,
    base_url: str | None = None,
) -> str | None:
    """Return a specific reason for skipping an obvious URL artifact."""

    if looks_like_email_artifact_url(url):
        return "email_artifact_url"

    if is_same_page_fragment_url(
        url=url,
        base_url=base_url,
    ):
        return "same_page_fragment"

    if is_same_page_url(
        url=url,
        base_url=base_url,
    ):
        return "same_page_url"

    return None


def looks_like_email_artifact_url(
    url: str,
) -> bool:
    """Return whether a URL contains an accidental email artifact."""

    try:
        parsed = urlsplit(url)
    except ValueError:
        return False

    if parsed.scheme.lower() == "mailto":
        return True

    haystack = " ".join(
        unquote(value or "").lower()
        for value in (
            parsed.netloc,
            parsed.path,
            parsed.query,
        )
    )

    return contains_email_address(haystack)


def contains_email_address(
    value: str,
) -> bool:
    """Return whether text contains an email-address pattern."""

    return bool(_EMAIL_ADDRESS_RE.search(value))


def is_same_page_fragment_url(
    *,
    url: str,
    base_url: str | None,
) -> bool:
    """Return whether a URL is a fragment of the current page."""

    if not base_url:
        return False

    try:
        parsed = urlsplit(url)
        base = urlsplit(base_url)

        if not parsed.fragment:
            return False

        return _without_fragment(parsed) == _without_fragment(base)
    except ValueError:
        return False


def is_same_page_url(
    *,
    url: str,
    base_url: str | None,
) -> bool:
    """Return whether a URL resolves to the current page itself."""

    if not base_url:
        return False

    try:
        parsed = urlsplit(url)
        base = urlsplit(base_url)

        if parsed.fragment:
            return False

        return _without_fragment(parsed) == _without_fragment(base)
    except ValueError:
        return False


def _without_fragment(
    parsed: SplitResult,
) -> str:
    """Return a normalized URL identity without its fragment."""

    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower().rstrip(".")

    port = parsed.port

    if port is not None and not (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        authority = f"{hostname}:{port}"
    else:
        authority = hostname

    normalized_path = unquote(parsed.path or "/").replace(
        "\\",
        "/",
    )

    while "//" in normalized_path:
        normalized_path = normalized_path.replace(
            "//",
            "/",
        )

    path = posixpath.normpath(normalized_path)

    if not path.startswith("/"):
        path = f"/{path}"

    path = path.rstrip("/") or "/"

    query_pairs = parse_qsl(
        parsed.query,
        keep_blank_values=True,
    )
    query_pairs.sort()

    query = urlencode(
        query_pairs,
        doseq=True,
    )

    return urlunsplit(
        (
            scheme,
            authority,
            path,
            query,
            "",
        )
    )


def mark_seen(
    seen: set[str],
    *,
    url: str,
    kind: str,
    source_type: str,
) -> bool:
    """Return whether a task identity is new in this extraction pass."""

    key = discovered_task_identity_from_parts(
        url=url,
        kind=kind,
        source_type=source_type,
    )

    if key in seen:
        return False

    seen.add(key)
    return True


def should_extract_assets_first(
    *,
    focus_kinds: tuple[str, ...],
) -> bool:
    """Return whether focused media assets should precede page links."""

    normalized = {
        str(kind).strip().lower() for kind in focus_kinds if str(kind).strip()
    }

    return bool(normalized & _MEDIA_FOCUS_KINDS)


def ordered_asset_candidates(
    *,
    discovery: Any,
    focus_kinds: tuple[str, ...],
) -> tuple[Any, ...]:
    """Order discovered assets according to requested focus modalities."""

    normalized_focus = tuple(
        str(kind).strip().lower() for kind in focus_kinds if str(kind).strip()
    )

    all_assets = tuple(
        getattr(
            discovery,
            "assets",
            (),
        )
        or ()
    )

    if not normalized_focus:
        return all_assets

    groups = {
        "image": tuple(getattr(discovery, "images", ()) or ()),
        "audio": tuple(getattr(discovery, "audio", ()) or ()),
        "video": tuple(getattr(discovery, "video", ()) or ()),
        "document": tuple(getattr(discovery, "documents", ()) or ()),
    }

    if not any(groups.values()):
        return all_assets

    ordered: list[Any] = []
    used_groups: set[str] = set()
    seen_candidates: set[str] = set()

    def append_unique(
        candidate: Any,
    ) -> None:
        candidate_id = str(
            getattr(
                candidate,
                "candidate_id",
                "",
            )
            or ""
        ).strip()

        candidate_url = str(
            getattr(
                candidate,
                "url",
                "",
            )
            or ""
        ).strip()

        candidate_kind = (
            str(
                getattr(
                    candidate,
                    "kind",
                    "",
                )
                or ""
            )
            .strip()
            .lower()
        )

        key = candidate_id

        if not key:
            key = discovered_task_identity_from_parts(
                url=candidate_url,
                kind=candidate_kind,
                source_type=_ASSET_SOURCE_TYPE,
            )

        if key in seen_candidates:
            return

        seen_candidates.add(key)
        ordered.append(candidate)

    for kind in normalized_focus:
        if kind not in groups:
            continue

        for candidate in groups[kind]:
            append_unique(candidate)

        used_groups.add(kind)

    for kind in _ASSET_GROUP_ORDER:
        if kind in used_groups:
            continue

        for candidate in groups[kind]:
            append_unique(candidate)

    # Preserve any assets that were present only in the aggregate collection.
    for candidate in all_assets:
        append_unique(candidate)

    return tuple(ordered)


__all__ = [
    "DiscoveryTaskBuilder",
    "contains_email_address",
    "context_payload",
    "discovered_url_skip_reason",
    "is_same_page_fragment_url",
    "is_same_page_url",
    "looks_like_email_artifact_url",
    "mark_seen",
    "ordered_asset_candidates",
    "should_extract_assets_first",
]
