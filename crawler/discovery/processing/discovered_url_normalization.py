"""Helpers for normalizing and classifying discovered URLs."""

from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

from crawler.classification.media_kind_registry import match_extension


def infer_discovered_kind(*, url: str) -> str:
    """Infer a crawl kind from a discovered URL suffix."""
    suffixes = extract_suffixes(url=url)

    for start_index in range(len(suffixes)):
        combined_suffix = "".join(suffixes[start_index:])
        matched_kind = match_extension(
            f"https://placeholder.invalid{combined_suffix}"
        )
        if matched_kind is not None:
            return matched_kind.value

    return "page"


def extract_suffixes(*, url: str) -> list[str]:
    """Return lower-cased path suffixes for a URL."""
    parsed = urlparse(url)
    return [
        suffix.lower() for suffix in PurePosixPath(parsed.path or "").suffixes
    ]


def dedupe_url_key(url: str) -> str:
    """Return a structural key for local discovery duplicate detection.

    Used only when no settings-driven UrlNormalizer is available (e.g. direct
    unit-test callers). Deliberately performs no query-parameter removal:
    tracking and media-variant equivalence is owned by UrlNormalizerSettings
    through the UrlNormalizer, never by a second canonicalization layer here.
    """
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    netloc = hostname

    if ":" in hostname and not hostname.startswith("["):
        netloc = f"[{hostname}]"

    if parsed.port is not None:
        is_default_http = scheme == "http" and parsed.port == 80
        is_default_https = scheme == "https" and parsed.port == 443
        if not (is_default_http or is_default_https):
            netloc = f"{netloc}:{parsed.port}"

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query = urlencode(sorted(query_pairs), doseq=True)

    return urlunsplit(
        (
            scheme,
            netloc or parsed.netloc.lower(),
            path,
            query,
            "",
        )
    )
