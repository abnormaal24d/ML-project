"""Expand configured source registry entries into runtime source settings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunsplit

from config.source_catalog.registry_settings import SourceRegistrySettings
from config.source_catalog.registry_source_settings import (
    SourceGovernanceEntry,
    validate_registry_sources,
)

DEFAULT_SOURCE_REGISTRY_PATH = Path(
    "config/files/sources/source_registry.json"
)
SOURCE_REGISTRY_SCHEMA_VERSION = "1.0"
_ALLOWED_SOURCE_PROFILES = frozenset(
    {
        "public_science",
        "public_science_small",
    }
)


@dataclass(frozen=True, slots=True)
class RegistryExpansionResult:
    """Expanded registry payload plus provenance metadata."""

    payload: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)


def apply_source_registry(
    mapping: dict[str, Any],
    *,
    project_root: Path,
    environment: str,
    registry_path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
) -> None:
    """
    Apply seed and governance entries from the configured source registry.
    """

    sources_payload = mapping.get("sources")
    if not isinstance(sources_payload, dict):
        return

    registry_selectors = sources_payload.get("registry_sources")
    if registry_selectors is None:
        return

    registry = _read_registry(
        _resolve_registry_path(
            project_root=project_root,
            registry_path=registry_path,
        )
    )
    environment_payload = _registry_environment(
        registry=registry,
        environment=environment,
    )
    if environment_payload:
        active_profile = environment_payload.get("active_profile")
        if active_profile is not None:
            profile_name = str(active_profile).strip()
            if profile_name not in _ALLOWED_SOURCE_PROFILES:
                raise ValueError(
                    "source registry active_profile must be one of "
                    f"{sorted(_ALLOWED_SOURCE_PROFILES)}: {profile_name!r}"
                )
            sources_payload["active_profile"] = profile_name

    selector_names = tuple(
        selector["name"] for selector in _selector_entries(registry_selectors)
    )
    validate_registry_sources(
        registry=registry,
        source_names=selector_names,
        environment=environment,
    )

    expansion = _expand_registry_sources(
        registry=registry,
        selectors=registry_selectors,
    )
    for key, value in expansion.payload.items():
        sources_payload[key] = value

    sources_payload["registry_sources_provenance"] = {
        "selectors": registry_selectors,
        **expansion.provenance,
    }
    sources_payload["selected_registry_sources"] = registry_selectors
    sources_payload.pop("registry_sources", None)


def _resolve_registry_path(
    *,
    project_root: Path,
    registry_path: str | Path,
) -> Path:
    path = Path(registry_path)
    if path.is_absolute():
        raise ValueError(
            "source registry path must be project-relative under "
            "config/files/sources"
        )
    resolved = (project_root / path).resolve()
    allowed_root = (project_root / "config" / "files" / "sources").resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            f"source registry path must remain under {allowed_root}: {path}"
        ) from exc
    return resolved


def _read_registry(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"source registry not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid source registry JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"source registry root must be an object: {path}")

    schema_version = payload.get("schema_version")
    if schema_version != SOURCE_REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            "source registry schema_version must be "
            f"{SOURCE_REGISTRY_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    return payload


def _registry_environment(
    *,
    registry: dict[str, Any],
    environment: str,
) -> dict[str, Any]:
    environments = registry.get("environments", {})
    if not isinstance(environments, dict):
        raise ValueError("source registry environments must be an object")
    payload = environments.get(environment, {})
    if not isinstance(payload, dict):
        raise ValueError(
            f"source registry environment {environment!r} must be an object"
        )
    return payload


def _expand_registry_sources(
    *,
    registry: dict[str, Any],
    selectors: object,
) -> RegistryExpansionResult:
    source_definitions = registry.get("sources")
    if not isinstance(source_definitions, dict):
        raise ValueError("source registry sources must be an object")

    seed_urls: list[str] = []
    seed_entries: list[dict[str, str]] = []
    seed_owner_by_url: dict[str, str] = {}
    source_scopes: list[dict[str, Any]] = []
    seed_hosts: list[str] = []
    allowed_asset_hosts: list[str] = []
    training_allowed_domains: list[str] = []
    feed_alternate_urls: dict[str, list[str]] = {}
    explicit_media_seeds: list[str] = []
    media_feeds: list[str] = []
    media_api_endpoints: list[str] = []
    governance_by_domain: dict[str, dict[str, Any]] = {}
    retention_days_by_domain: dict[str, int] = {}
    provenance: dict[str, Any] = {"truncated_seed_counts": {}}

    for selector in _selector_entries(selectors):
        name = selector["name"]
        source_payload = source_definitions.get(name)
        if not isinstance(source_payload, dict):
            raise ValueError(f"unknown source registry entry: {name!r}")

        max_seeds = selector.get("max_seeds")
        configured_seed_urls = _strings(
            source_payload.get("seed_urls", []),
            field_name=f"sources.{name}.seed_urls",
            validate_urls=True,
            require_strings=True,
        )
        media_urls: list[str] = []
        media_urls_by_field: dict[str, list[str]] = {}
        for field_name in (
            "explicit_media_seeds",
            "media_feeds",
            "media_api_endpoints",
        ):
            values = _strings(
                source_payload.get(field_name, []),
                field_name=f"sources.{name}.{field_name}",
                validate_urls=True,
                require_strings=True,
            )
            media_urls_by_field[field_name] = values
            _extend_unique(
                media_urls,
                values,
            )
        # Direct media URLs are crawl seeds in their own right. Keep them ahead
        # of ordinary page seeds so a source-specific max_seeds limit cannot
        # silently remove the only audio/video discovery entry points.
        all_seed_urls: list[str] = []
        _extend_unique(all_seed_urls, media_urls)
        _extend_unique(all_seed_urls, configured_seed_urls)
        selected_seed_urls = all_seed_urls
        truncated_count = 0
        if max_seeds is not None:
            truncated_count = max(0, len(all_seed_urls) - int(max_seeds))
            selected_seed_urls = all_seed_urls[: int(max_seeds)]
            if truncated_count:
                provenance["truncated_seed_counts"][name] = truncated_count

        for selected_seed_url in selected_seed_urls:
            normalized_seed_url = _normalize_url_key(selected_seed_url)
            existing_owner = seed_owner_by_url.get(normalized_seed_url)
            if existing_owner is not None and existing_owner != name:
                raise ValueError(
                    "seed URL is assigned to multiple registry sources: "
                    f"{selected_seed_url!r} belongs to both "
                    f"{existing_owner!r} and {name!r}"
                )
            if existing_owner is None:
                seed_owner_by_url[normalized_seed_url] = name
                seed_entries.append(
                    {
                        "source_name": name,
                        "url": selected_seed_url,
                    }
                )

        _extend_unique(seed_urls, selected_seed_urls)
        selected_hosts = _limited_seed_hosts(
            configured_hosts=source_payload.get("seed_hosts", []),
            seed_urls=selected_seed_urls,
            field_name=f"sources.{name}.seed_hosts",
            truncated=max_seeds is not None,
        )
        _extend_unique(seed_hosts, selected_hosts)

        asset_hosts = _strings(
            source_payload.get("allowed_asset_hosts", []),
            field_name=f"sources.{name}.allowed_asset_hosts",
            require_strings=True,
        )
        _extend_unique(allowed_asset_hosts, asset_hosts)

        redirect_hosts = _strings(
            source_payload.get("allowed_redirect_hosts", []),
            field_name=f"sources.{name}.allowed_redirect_hosts",
            require_strings=True,
        )
        allow_subdomains = source_payload.get("allow_subdomains", False)
        if not isinstance(allow_subdomains, bool):
            raise ValueError(
                f"sources.{name}.allow_subdomains must be a boolean"
            )

        source_scopes.append(
            {
                "source_name": name,
                "page_hosts": selected_hosts,
                "asset_hosts": asset_hosts,
                "redirect_hosts": redirect_hosts,
                "allow_subdomains": allow_subdomains,
            }
        )

        training_domains = _strings(
            source_payload.get("training_allowed_domains", []),
            field_name=f"sources.{name}.training_allowed_domains",
            require_strings=True,
        )
        if max_seeds is not None:
            training_domains = [
                domain
                for domain in training_domains
                if domain.lower() in {item.lower() for item in selected_hosts}
            ]
        _extend_unique(training_allowed_domains, training_domains)

        active_domains = {
            *(domain.lower() for domain in training_domains),
            *(host.lower() for host in asset_hosts),
            *(host.lower() for host in selected_hosts),
            *(host.lower() for host in redirect_hosts),
        }
        retention_days = source_payload.get("retention_days")
        if isinstance(retention_days, bool) or not isinstance(
            retention_days, int
        ):
            raise ValueError(
                f"sources.{name}.retention_days must be an integer"
            )
        for domain in active_domains:
            existing_retention = retention_days_by_domain.get(domain)
            retention_days_by_domain[domain] = (
                retention_days
                if existing_retention is None
                else min(existing_retention, retention_days)
            )
        _merge_governance_entries(
            governance_by_domain=governance_by_domain,
            gov_entries=source_payload.get("governance", []),
            field_name=f"sources.{name}.governance",
            max_seeds=max_seeds,
            active_domains=active_domains,
        )

        _merge_feed_alternates(
            target=feed_alternate_urls,
            source_payload=source_payload,
            field_name=f"sources.{name}.feed_alternate_urls",
        )

        selected_seed_keys = {
            _normalize_url_key(url) for url in selected_seed_urls
        }
        for field_name, target in (
            ("explicit_media_seeds", explicit_media_seeds),
            ("media_feeds", media_feeds),
            ("media_api_endpoints", media_api_endpoints),
        ):
            _extend_unique(
                target,
                [
                    url
                    for url in media_urls_by_field[field_name]
                    if _normalize_url_key(url) in selected_seed_keys
                ],
            )

    derived_training_allowed_domains = [
        domain
        for domain, entry in governance_by_domain.items()
        if entry["training"]["allowed"] is True
    ]

    return RegistryExpansionResult(
        payload={
            "seed_urls": seed_urls,
            "seed_entries": seed_entries,
            "source_scopes": source_scopes,
            "seed_hosts": seed_hosts,
            "allowed_asset_hosts": allowed_asset_hosts,
            "training_allowed_domains": derived_training_allowed_domains,
            "feed_alternate_urls": feed_alternate_urls,
            "explicit_media_seeds": tuple(explicit_media_seeds),
            "media_feeds": tuple(media_feeds),
            "media_api_endpoints": tuple(media_api_endpoints),
            "governance": list(governance_by_domain.values()),
            "retention_days_by_domain": retention_days_by_domain,
        },
        provenance=provenance,
    )


def _merge_governance_entries(
    *,
    governance_by_domain: dict[str, dict[str, Any]],
    gov_entries: object,
    field_name: str,
    max_seeds: int | None,
    active_domains: set[str],
) -> None:
    """Merge governance entries with conflict detection. Extracted to shrink caller."""
    for entry in _governance_entries(gov_entries, field_name=field_name):
        rules_entry = SourceGovernanceEntry.model_validate(entry)
        domain = rules_entry.domain.strip().lower()
        if not domain:
            continue
        if max_seeds is not None and domain not in active_domains:
            continue
        # Convert registry flat gov entry to full structured SourceRulesSettings shape
        structured = {
            "domain": domain,
            "source_id": domain,
            "source_name": domain,
            "rules_version": "1",
            "registry_version": "1",
            "collection": {
                "allowed": rules_entry.allow_collection,
                "reason": "from_registry",
            },
            "license": {
                "expression": rules_entry.license or "unknown",
                "evidence_url": rules_entry.license_evidence_url,
                "evidence_kind": rules_entry.license_evidence_kind,
                "reason": "registry",
            },
            "training": {
                "allowed": rules_entry.allow_training,
                "reason": "from_registry",
            },
            "allow_boilerplate_image_caption": (
                rules_entry.allow_boilerplate_image_caption
            ),
        }
        existing = governance_by_domain.get(domain)
        if existing is not None and existing != structured:
            raise ValueError(
                "conflicting governance rules for domain "
                f"{domain!r} across registry sources"
            )
        governance_by_domain[domain] = structured


def _merge_feed_alternates(
    *,
    target: dict[str, list[str]],
    source_payload: dict[str, Any],
    field_name: str,
) -> None:
    raw_alternates = source_payload.get("feed_alternate_urls", {})
    if not isinstance(raw_alternates, dict):
        raise ValueError(f"{field_name} must be an object")
    for primary, alternates in raw_alternates.items():
        primary_normalized = _normalize_url_key(str(primary))
        if not primary_normalized:
            continue
        _validate_absolute_url(
            str(primary), field_name=f"{field_name}.{primary}"
        )
        values = _strings(
            alternates,
            field_name=f"{field_name}.{primary}",
            validate_urls=True,
            require_strings=True,
        )
        if not values:
            continue
        existing = list(target.get(primary_normalized, ()))
        seen = {_normalize_url_key(item) for item in existing}
        for value in values:
            marker = _normalize_url_key(value)
            if marker in seen:
                continue
            seen.add(marker)
            existing.append(value)
        target[primary_normalized] = existing


def _normalize_url_key(value: str) -> str:
    """Return a comparison key with only scheme and host case-folded."""

    text = value.strip()
    parsed = urlsplit(text)
    if not parsed.scheme or not parsed.netloc:
        return text
    try:
        port = parsed.port
    except ValueError:
        return text
    hostname = parsed.hostname
    if not hostname:
        return text

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    netloc = f"{userinfo}{hostname.lower()}"
    if port is not None:
        netloc += f":{port}"
    return urlunsplit(
        (
            parsed.scheme.lower(),
            netloc,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def _selector_entries(selectors: object) -> tuple[dict[str, Any], ...]:
    registry_settings = SourceRegistrySettings.model_validate(
        {"selectors": selectors}
    )
    return registry_settings.selector_dicts()


def _strings(
    value: object,
    *,
    field_name: str,
    validate_urls: bool = False,
    require_strings: bool = False,
) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")
    normalized: list[str] = []
    for index, item in enumerate(value):
        if require_strings and not isinstance(item, str):
            raise ValueError(
                f"{field_name}[{index}] must be a string, got {type(item).__name__}"
            )
        text = str(item).strip()
        if not text:
            continue
        if validate_urls:
            _validate_absolute_url(text, field_name=field_name)
        normalized.append(text)
    return normalized


def _validate_absolute_url(value: str, *, field_name: str) -> None:
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must contain absolute http/https URLs"
        ) from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or not hostname
    ):
        raise ValueError(f"{field_name} must contain absolute http/https URLs")


def _limited_seed_hosts(
    *,
    configured_hosts: object,
    seed_urls: list[str],
    field_name: str,
    truncated: bool = False,
) -> list[str]:
    hosts = _strings(
        configured_hosts,
        field_name=field_name,
        require_strings=True,
    )
    selected_hosts = {
        parsed.hostname.lower()
        for parsed in (urlparse(url) for url in seed_urls)
        if parsed.hostname
    }
    if hosts:
        if truncated:
            return [host for host in hosts if host.lower() in selected_hosts]
        missing = sorted(
            host for host in hosts if host.lower() not in selected_hosts
        )
        if missing:
            raise ValueError(
                f"{field_name} contains hosts without matching seed URLs: "
                f"{missing}"
            )
        return hosts
    return sorted(selected_hosts)


def _governance_entries(
    value: object,
    *,
    field_name: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"{field_name}[{index}] must be an object")
        entries.append(entry)
    return tuple(entries)


def _extend_unique(target: list[str], values: list[str]) -> None:
    seen = {_normalize_url_key(item) for item in target}
    for value in values:
        normalized = value.strip()
        marker = _normalize_url_key(normalized)
        if not normalized or marker in seen:
            continue
        seen.add(marker)
        target.append(normalized)
