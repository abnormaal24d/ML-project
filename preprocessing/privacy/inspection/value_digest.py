"""One-way normalization used to correlate findings without storing values."""

from __future__ import annotations

import hashlib
import unicodedata


def digest_sensitive_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
