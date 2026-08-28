"""Small fail-closed governance helpers used before curation."""

from __future__ import annotations

from collections.abc import Mapping

_KNOWN_LICENSES = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0",
        "CC0-1.0",
        "CC-BY-1.0",
        "CC-BY-2.0",
        "CC-BY-2.5",
        "CC-BY-3.0",
        "CC-BY-4.0",
        "CC-BY-SA-1.0",
        "CC-BY-SA-2.0",
        "CC-BY-SA-2.5",
        "CC-BY-SA-3.0",
        "CC-BY-SA-4.0",
        "GPL-2.0-only",
        "GPL-2.0-or-later",
        "GPL-3.0-only",
        "GPL-3.0-or-later",
        "LGPL-2.1-only",
        "LGPL-2.1-or-later",
        "LGPL-3.0-only",
        "LGPL-3.0-or-later",
        "MIT",
        "MPL-2.0",
        "Unlicense",
    }
)


def governance_payload(value: object) -> dict[str, object]:
    """Extract canonical governance inputs for the normal privacy scan."""

    if not isinstance(value, Mapping):
        return {}
    license_map = _mapping(value.get("license"))
    training_map = _mapping(value.get("training"))
    access_map = _mapping(value.get("access"))
    result: dict[str, object] = {}
    for name, raw in (
        ("license", license_map.get("expression")),
        ("license_url", license_map.get("evidence_url")),
        ("terms_source", license_map.get("evidence_kind")),
        ("governance_note", training_map.get("reason")),
        ("robots_status", access_map.get("decision")),
        ("usage_rules", value.get("usage_rules")),
    ):
        if isinstance(raw, str) and raw.strip():
            result[name] = raw.strip()
    allowed = training_map.get("allowed")
    if isinstance(allowed, bool):
        result["allow_training"] = allowed
    return result


def safe_license_expression(value: object) -> str | None:
    """Return only a known single SPDX-style license identifier."""

    if value is None:
        return None
    text = str(value).strip()
    return text if text in _KNOWN_LICENSES else None


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


__all__ = ["governance_payload", "safe_license_expression"]
