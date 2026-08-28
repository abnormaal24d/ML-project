"""Composition boundary for building the source scope registry.

Composition owns exactly one responsibility here: adapting validated
config host representations into canonical domain hosts through
``HostNormalizer.require()``. Scope and collection invariants live in the
domain objects.
"""

from __future__ import annotations

import pytest

from config.source_catalog.catalog_settings import SourceScopeSettings
from crawler.governance.domains.host_normalizer import HostNormalizer
from orchestration.composition.runtime.governance import (
    build_source_scope_registry,
)


def test_source_scope_builder_canonicalizes_hosts() -> None:
    registry = build_source_scope_registry(
        source_scopes=(
            SourceScopeSettings(
                source_name="news",
                page_hosts=("BÜCHER.example.",),
            ),
        ),
        host_normalizer=HostNormalizer(),
    )

    scope = registry.require("news")

    assert scope.page_hosts == frozenset({"xn--bcher-kva.example"})


def test_source_scope_builder_rejects_invalid_hosts() -> None:
    settings = SourceScopeSettings(
        source_name="news",
        page_hosts=("invalid host",),
    )

    with pytest.raises(ValueError):
        build_source_scope_registry(
            source_scopes=(settings,),
            host_normalizer=HostNormalizer(),
        )


def test_source_scope_builder_preserves_duplicate_detection() -> None:
    """Duplicate config entries fail closed at the domain boundary.

    ``SourceProfileSettings`` already rejects duplicate names during config
    validation; this test proves the domain registry independently enforces
    the same invariant for directly constructed scope instances.
    """

    duplicated = (
        SourceScopeSettings(
            source_name="news",
            page_hosts=("example.com",),
        ),
        SourceScopeSettings(
            source_name="news",
            page_hosts=("example.org",),
        ),
    )

    with pytest.raises(ValueError, match="duplicate source scope"):
        build_source_scope_registry(
            source_scopes=duplicated,
            host_normalizer=HostNormalizer(),
        )
