"""Canonical URL deduplication for resolved asset candidates."""

from __future__ import annotations

from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit

from crawler.extraction.assets.candidate.asset_extraction_records import (
    AssetCandidate,
)

TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = frozenset({"fbclid", "gclid", "msclkid"})


class AssetCandidateDeduper:
    """Deduplicate asset candidates by canonical URL identity."""

    def add_or_merge(
        self,
        *,
        results: list[AssetCandidate],
        seen: dict[str, int],
        candidate: AssetCandidate,
    ) -> bool:
        identity_key = self._identity_key(candidate)
        existing_index = seen.get(identity_key)

        if existing_index is None:
            seen[identity_key] = len(results)
            results.append(candidate)
            return True

        existing = results[existing_index]
        candidate_preference = self.candidate_preference_score(candidate.url)
        existing_preference = self.candidate_preference_score(existing.url)

        if candidate_preference > existing_preference:
            preferred = candidate
            other = existing
        elif candidate_preference < existing_preference:
            preferred = existing
            other = candidate
        elif self.candidate_context_score(
            candidate
        ) > self.candidate_context_score(existing):
            preferred = candidate
            other = existing
        else:
            preferred = existing
            other = candidate

        merged = self._merge_candidates(
            preferred=preferred,
            other=other,
        )

        if merged == existing:
            return False

        results[existing_index] = merged
        return True

    def _identity_key(
        self,
        candidate: AssetCandidate,
    ) -> str:
        canonical_url = self.candidate_identity_key(candidate.url)
        kind_value = getattr(candidate.kind, "value", candidate.kind)
        return f"{canonical_url}|{kind_value}"

    @staticmethod
    def _merge_candidates(
        *,
        preferred: AssetCandidate,
        other: AssetCandidate,
    ) -> AssetCandidate:
        return AssetCandidate(
            url=preferred.url,
            kind=preferred.kind,
            parent_url=preferred.parent_url,
            source_attribute=preferred.source_attribute,
            source_tag=preferred.source_tag,
            alt_text=preferred.alt_text or other.alt_text,
            caption_text=(preferred.caption_text or other.caption_text),
            surrounding_text=(
                preferred.surrounding_text or other.surrounding_text
            ),
            mime_hint=preferred.mime_hint or other.mime_hint,
            width=(
                preferred.width if preferred.width is not None else other.width
            ),
            height=(
                preferred.height
                if preferred.height is not None
                else other.height
            ),
            duration_seconds=(
                preferred.duration_seconds
                if preferred.duration_seconds is not None
                else other.duration_seconds
            ),
            metadata={
                **other.metadata,
                **preferred.metadata,
            },
        )

    @staticmethod
    def candidate_identity_key(url: str) -> str:
        parsed = urlsplit(_canonical_asset_url(url))
        if _is_iiif_full_url_path(path=parsed.path):
            return _iiif_full_identity_key(parsed_url=parsed)
        return parsed.geturl()

    @staticmethod
    def candidate_preference_score(url: str) -> float:
        parsed = urlsplit(url)
        if not _is_iiif_full_url_path(path=parsed.path):
            return 0.0

        size_segment = _iiif_size_segment(path=parsed.path)
        if size_segment == "full":
            return 10_000.0

        if size_segment.startswith("pct:"):
            try:
                return float(size_segment.split(":", 1)[1])
            except ValueError:
                return 0.0

        numeric_prefix = size_segment.split(",", 1)[0].strip()
        if numeric_prefix.isdigit():
            return float(numeric_prefix)

        return 0.0

    @staticmethod
    def candidate_context_score(candidate: AssetCandidate) -> int:
        score = 0
        for value in (
            candidate.caption_text,
            candidate.alt_text,
            candidate.surrounding_text,
            candidate.mime_hint,
        ):
            if value:
                score += 1
        if candidate.width is not None or candidate.height is not None:
            score += 1
        return score


def _canonical_asset_url(url: str) -> str:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    netloc = _canonical_netloc(parsed_url=parsed, scheme=scheme)
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(
                parsed.query,
                keep_blank_values=True,
            )
            if not _is_tracking_query_key(key=key)
        ),
        doseq=True,
    )
    return parsed._replace(
        scheme=scheme,
        netloc=netloc,
        query=query,
        fragment="",
    ).geturl()


def _canonical_netloc(*, parsed_url: SplitResult, scheme: str) -> str:
    hostname = parsed_url.hostname
    if hostname is None:
        return str(parsed_url.netloc).lower()

    userinfo = ""
    if parsed_url.username:
        userinfo = parsed_url.username
        if parsed_url.password:
            userinfo = f"{userinfo}:{parsed_url.password}"
        userinfo = f"{userinfo}@"

    host = hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    try:
        port = parsed_url.port
    except ValueError:
        return f"{userinfo}{host}"

    if port is None or (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        return f"{userinfo}{host}"
    return f"{userinfo}{host}:{port}"


def _is_tracking_query_key(*, key: str) -> bool:
    lowered = key.strip().lower()
    return lowered in TRACKING_QUERY_KEYS or lowered.startswith(
        TRACKING_QUERY_PREFIXES
    )


def _is_iiif_full_url_path(*, path: str) -> bool:
    lowered_path = path.lower()
    return "/iiif/" in lowered_path and "/full/" in lowered_path


def _iiif_full_identity_key(*, parsed_url: SplitResult) -> str:
    lowered_path = parsed_url.path.lower()
    full_index = lowered_path.find("/full/")
    prefix = parsed_url.path[:full_index]
    return str(
        parsed_url._replace(
            path=f"{prefix}/full/",
            query="",
            fragment="",
        ).geturl()
    )


def _iiif_size_segment(*, path: str) -> str:
    lowered_path = path.lower()
    full_index = lowered_path.find("/full/")
    remainder = path[full_index + len("/full/") :]
    return remainder.split("/", 1)[0].strip().lower()
