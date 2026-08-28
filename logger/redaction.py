"""Fail-closed redaction for every structured log value."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"
LOCAL_PATH = "[LOCAL_PATH]"

_SENSITIVE_KEY_PARTS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "password",
        "passwd",
        "private_key",
        "secret",
        "session_token",
        "signature",
        "signed_token",
        "token",
    }
)
_URL_PATTERN = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
_WINDOWS_PATH_PATTERN = re.compile(
    r"(?<![\w])(?:[A-Za-z]:[\\/](?:[^\s,;:'\"<>|]+[\\/]?)+)"
)
_POSIX_PATH_PATTERN = re.compile(
    r"(?<![:\w])/(?:Users|home|mnt|private|root|tmp|var/tmp)/[^\s,;:'\"<>]+"
)


def redact_log_value(value: object, *, field_name: str = "") -> object:
    """Return a recursively redacted value suitable for any log sink."""

    if _is_sensitive_field(field_name):
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(key): redact_log_value(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact_log_value(item) for item in value)
    if isinstance(value, list):
        return [redact_log_value(item) for item in value]
    if isinstance(value, set):
        return {redact_log_value(item) for item in value}
    if isinstance(value, str):
        if _is_path_field(field_name) and _is_absolute_path(value):
            return LOCAL_PATH
        return redact_log_text(value)
    return value


def redact_log_text(value: str) -> str:
    """Redact credentials, query values and absolute local paths in text."""

    protected_urls: dict[str, str] = {}

    def protect_url(match: re.Match[str]) -> str:
        marker = f"__REDACTED_URL_{len(protected_urls)}__"
        protected_urls[marker] = _redact_url(match.group(0))
        return marker

    redacted = _URL_PATTERN.sub(protect_url, value)
    redacted = _WINDOWS_PATH_PATTERN.sub(LOCAL_PATH, redacted)
    redacted = _POSIX_PATH_PATTERN.sub(LOCAL_PATH, redacted)
    for marker, safe_url in protected_urls.items():
        redacted = redacted.replace(marker, safe_url)
    return redacted


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if hostname is None:
            return REDACTED
        host = f"[{hostname}]" if ":" in hostname else hostname
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
        query = urlencode(
            [
                (key, REDACTED)
                for key, _ in parse_qsl(
                    parsed.query,
                    keep_blank_values=True,
                )
            ]
        )
        return urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))
    except ValueError:
        return REDACTED


def _is_sensitive_field(field_name: str) -> bool:
    normalized = field_name.strip().lower().replace("-", "_")
    segments = frozenset(normalized.split("_"))
    return normalized in _SENSITIVE_KEY_PARTS or bool(
        segments
        & {
            "apikey",
            "authorization",
            "cookie",
            "credential",
            "password",
            "passwd",
            "secret",
            "signature",
            "token",
        }
    )


def _is_path_field(field_name: str) -> bool:
    normalized = field_name.strip().lower()
    return normalized in {"path", "directory"} or normalized.endswith(
        ("_path", "_directory")
    )


def _is_absolute_path(value: str) -> bool:
    return (
        PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
    )
