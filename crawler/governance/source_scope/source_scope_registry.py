"""Immutable source-bound host scopes used by crawler governance."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast


def _normalize_source_name(value: object) -> str:
    source_name = str(value).strip().lower()
    if not source_name:
        raise ValueError("source scope requires source_name")
    return source_name


def _normalize_hosts(values: object, *, field_name: str) -> frozenset[str]:
    if values is None:
        return frozenset()

    try:
        candidates = tuple(cast(Iterable[object], values))
    except TypeError as exc:
        raise ValueError(f"{field_name} must be iterable") from exc

    normalized: set[str] = set()
    for candidate in candidates:
        host = str(candidate).strip().lower().removesuffix(".")
        if not host:
            raise ValueError(f"{field_name} contains an empty host")
        normalized.add(host)

    return frozenset(normalized)


@dataclass(frozen=True, slots=True)
class SourceScope:
    """Allowed page, asset, and redirect hosts for one source family."""

    source_name: str
    page_hosts: frozenset[str]
    asset_hosts: frozenset[str]
    redirect_hosts: frozenset[str]
    allow_subdomains: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_name",
            _normalize_source_name(self.source_name),
        )
        object.__setattr__(
            self,
            "page_hosts",
            _normalize_hosts(self.page_hosts, field_name="page_hosts"),
        )
        object.__setattr__(
            self,
            "asset_hosts",
            _normalize_hosts(self.asset_hosts, field_name="asset_hosts"),
        )
        object.__setattr__(
            self,
            "redirect_hosts",
            _normalize_hosts(
                self.redirect_hosts,
                field_name="redirect_hosts",
            ),
        )

        if not self.page_hosts and not self.asset_hosts:
            raise ValueError(
                f"source scope {self.source_name!r} has no approved hosts"
            )

    def allows_page_host(self, host: str) -> bool:
        return self._matches(host, self.page_hosts)

    def allows_asset_host(self, host: str) -> bool:
        return self._matches(host, self.asset_hosts)

    def allows_redirect_host(self, host: str) -> bool:
        return self._matches(host, self.redirect_hosts)

    def _matches(self, host: str, allowed_hosts: frozenset[str]) -> bool:
        candidate = str(host).strip().lower().removesuffix(".")
        if not candidate:
            return False
        if candidate in allowed_hosts:
            return True
        if not self.allow_subdomains:
            return False
        return any(
            candidate.endswith(f".{allowed_host}")
            for allowed_host in allowed_hosts
        )


class SourceScopeRegistry:
    """Fail-closed lookup of immutable source host scopes."""

    def __init__(self, scopes: Iterable[SourceScope]) -> None:
        normalized: dict[str, SourceScope] = {}

        for scope in scopes:
            if not isinstance(scope, SourceScope):
                raise TypeError("source scopes must be SourceScope instances")

            source_name = scope.source_name

            if source_name in normalized:
                raise ValueError(f"duplicate source scope: {source_name!r}")

            normalized[source_name] = scope

        if not normalized:
            raise ValueError("source scope registry must not be empty")

        self._scopes: Mapping[str, SourceScope] = MappingProxyType(normalized)

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._scopes))

    def require(self, source_name: str) -> SourceScope:
        normalized_name = _normalize_source_name(source_name)

        try:
            return self._scopes[normalized_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown source scope: {normalized_name!r}"
            ) from exc
