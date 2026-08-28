"""Checkpoint restore semantics for the scheduler seen-identity registry."""

from __future__ import annotations

from crawler.scheduling.dedupe import seen_url_registry as registry_module
from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry


def test_restore_consumes_all_entries_then_keeps_latest_occurrences(
    monkeypatch,
) -> None:
    monkeypatch.setattr(registry_module, "time", lambda: 1_000.0)
    registry = SeenUrlRegistry(max_seen=2)

    loaded = registry.replace_entries(
        (
            ("https://example.test/a", 1.0),
            ("https://example.test/b", 2.0),
            ("https://example.test/a", 3.0),
            ("https://example.test/c", 4.0),
        )
    )

    assert loaded == 2
    assert registry.export_entries() == (
        ("https://example.test/a", 3.0),
        ("https://example.test/c", 4.0),
    )


def test_restore_purges_expired_entries_even_when_restore_order_is_not_temporal(
    monkeypatch,
) -> None:
    monkeypatch.setattr(registry_module, "time", lambda: 1_000.0)
    registry = SeenUrlRegistry(max_seen=10, ttl_seconds=10.0)

    loaded = registry.replace_entries(
        (
            ("https://example.test/fresh", 999.0),
            ("https://example.test/stale", 1.0),
        )
    )

    assert loaded == 1
    assert registry.export_entries() == (
        ("https://example.test/fresh", 999.0),
    )
