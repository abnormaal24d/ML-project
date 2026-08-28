"""Registry invariants for immutable source host scopes."""

from __future__ import annotations

import pytest

from crawler.governance.source_scope.source_scope_registry import (
    SourceScope,
    SourceScopeRegistry,
)


def _scope(
    source_name: str,
    *,
    page_hosts: frozenset[str] | None = None,
) -> SourceScope:
    return SourceScope(
        source_name=source_name,
        page_hosts=page_hosts or frozenset({"example.com"}),
        asset_hosts=frozenset(),
        redirect_hosts=frozenset(),
    )


def test_registry_rejects_duplicate_source_names() -> None:
    first = _scope("news", page_hosts=frozenset({"example.com"}))
    # SourceScope normalizes "NEWS" to "news", so this collides with the
    # first scope and the registry must fail closed.
    second = SourceScope(
        source_name="NEWS",
        page_hosts=frozenset({"example.org"}),
        asset_hosts=frozenset(),
        redirect_hosts=frozenset(),
    )

    with pytest.raises(ValueError, match="duplicate source scope"):
        SourceScopeRegistry((first, second))


def test_registry_rejects_non_scope_entries() -> None:
    with pytest.raises(TypeError, match="SourceScope instances"):
        SourceScopeRegistry(("news",))  # type: ignore[list-item]


def test_registry_rejects_empty_scope_collection() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SourceScopeRegistry(())


def test_registry_lookup_normalizes_source_name() -> None:
    registry = SourceScopeRegistry((_scope("news"),))

    assert registry.require(" NEWS ").source_name == "news"


def test_registry_exposes_sorted_source_names() -> None:
    registry = SourceScopeRegistry((_scope("video"), _scope("audio")))

    assert registry.source_names == ("audio", "video")


def test_registry_lookup_fails_closed_for_unknown_source() -> None:
    registry = SourceScopeRegistry((_scope("news"),))

    with pytest.raises(ValueError, match="Unknown source scope"):
        registry.require("unknown")
