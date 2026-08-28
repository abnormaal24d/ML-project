"""Regression tests for streaming curated content fingerprints."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from crawler.curation.snapshots.dataset_assembly.curated_snapshot_fingerprint import (
    _FINGERPRINT_RECORD_BATCH_SIZE,
    build_curation_input_fingerprint,
    build_snapshot_fingerprint_payload,
)


def _entry(
    *,
    run_id: str,
    fetch_record_id: str,
    object_id: str,
    kind: str,
    storage_relative_path: str,
    content_sha256: str | None,
    byte_size: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        record=SimpleNamespace(
            run_id=run_id,
            fetch_record_id=fetch_record_id,
            object_id=object_id,
            kind=kind,
            storage_relative_path=storage_relative_path,
            content_sha256=content_sha256,
            byte_size=byte_size,
        )
    )


def _reference_build_curation_input_fingerprint(
    *,
    raw_entries: tuple[Any, ...],
    settings_payload: dict[str, Any],
    relevant_kinds: frozenset[str],
) -> str:
    """Reference serialization algorithm for exact digest verification."""

    payload = {
        "raw_entries": [
            {
                "run_id": entry.record.run_id,
                "fetch_record_id": entry.record.fetch_record_id,
                "object_id": entry.record.object_id,
                "kind": entry.record.kind,
                "storage_relative_path": entry.record.storage_relative_path,
                "content_sha256": getattr(
                    entry.record, "content_sha256", None
                ),
                "byte_size": entry.record.byte_size,
            }
            for entry in sorted(
                raw_entries,
                key=lambda item: (
                    item.record.run_id,
                    item.record.fetch_record_id,
                    item.record.storage_relative_path,
                ),
            )
        ],
        "relevant_kinds": sorted(relevant_kinds),
        "settings": settings_payload,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sample_entries() -> tuple[Any, ...]:
    return (
        _entry(
            run_id="run-b",
            fetch_record_id="fetch-2",
            object_id="obj-2",
            kind="image",
            storage_relative_path="images/b.jpg",
            content_sha256="bbb",
            byte_size=20,
        ),
        _entry(
            run_id="run-a",
            fetch_record_id="fetch-1",
            object_id="obj-1",
            kind="document",
            storage_relative_path="docs/a.txt",
            content_sha256="aaa",
            byte_size=10,
        ),
        _entry(
            run_id="run-a",
            fetch_record_id="fetch-1",
            object_id="obj-1b",
            kind="document",
            storage_relative_path="docs/a2.txt",
            content_sha256=None,
            byte_size=11,
        ),
        _entry(
            run_id="run-c",
            fetch_record_id="fetch-9",
            object_id="obj-ü",
            kind="audio",
            storage_relative_path="audio/ünicode.wav",
            content_sha256="ccc",
            byte_size=30,
        ),
    )


def test_streaming_fingerprint_matches_reference_payload_hash() -> None:
    raw_entries = _sample_entries()
    settings_payload = {
        "nested": {"beta": 2, "alpha": 1},
        "flag": True,
        "threshold": 0.5,
        "path": object(),  # exercises default=str
    }
    relevant_kinds = frozenset({"document", "image", "audio", "video"})

    assert build_curation_input_fingerprint(
        raw_entries=raw_entries,
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    ) == _reference_build_curation_input_fingerprint(
        raw_entries=raw_entries,
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    )


def test_fingerprint_independent_of_manifest_row_order() -> None:
    entries = _sample_entries()
    settings_payload = {"mode": "curated", "version": 3}
    relevant_kinds = frozenset({"document", "image", "audio"})

    forward = build_curation_input_fingerprint(
        raw_entries=entries,
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    )
    reverse = build_curation_input_fingerprint(
        raw_entries=tuple(reversed(entries)),
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    )
    reference_reverse = _reference_build_curation_input_fingerprint(
        raw_entries=tuple(reversed(entries)),
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    )

    assert forward == reverse
    assert reverse == reference_reverse


def test_fingerprint_serialization_is_batched_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = tuple(
        _entry(
            run_id=f"run-{index % 3}",
            fetch_record_id=f"fetch-{index:04d}",
            object_id=f"obj-{index}",
            kind="document",
            storage_relative_path=f"docs/{index:04d}.txt",
            content_sha256=f"hash-{index}",
            byte_size=index,
        )
        for index in range(600)
    )
    settings_payload = {"k": "v"}
    relevant_kinds = frozenset({"document"})

    dumps_calls: list[Any] = []
    original_dumps = json.dumps

    def tracking_dumps(obj: Any, *args: Any, **kwargs: Any) -> str:
        dumps_calls.append(obj)
        return original_dumps(obj, *args, **kwargs)

    # Patch only the module under test so the reference implementation stays clean.
    from crawler.curation.snapshots.dataset_assembly import (
        curated_snapshot_fingerprint,
    )

    monkeypatch.setattr(
        curated_snapshot_fingerprint.json, "dumps", tracking_dumps
    )

    digest = build_curation_input_fingerprint(
        raw_entries=entries,
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    )

    # Never serialize one giant top-level payload with raw_entries.
    for call in dumps_calls:
        if isinstance(call, dict):
            assert "raw_entries" not in call

    # Record batches are lists of projections, each <= batch size.
    record_batch_calls = [
        call
        for call in dumps_calls
        if isinstance(call, list)
        and call
        and isinstance(call[0], dict)
        and "run_id" in call[0]
    ]
    assert record_batch_calls
    total_records = 0
    for batch in record_batch_calls:
        assert len(batch) <= _FINGERPRINT_RECORD_BATCH_SIZE
        total_records += len(batch)
    assert total_records == len(entries)

    # Full dataset must not be serialized as one record list.
    assert all(len(batch) < len(entries) for batch in record_batch_calls)

    # Digest equivalence check after inspecting streaming dumps only.
    assert digest == _reference_build_curation_input_fingerprint(
        raw_entries=entries,
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    )


def test_empty_entries_match_reference() -> None:
    settings_payload = {"empty": True}
    relevant_kinds = frozenset({"document"})
    assert build_curation_input_fingerprint(
        raw_entries=(),
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    ) == _reference_build_curation_input_fingerprint(
        raw_entries=(),
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    )


def test_stable_tie_break_preserves_input_order_for_equal_keys() -> None:
    """Equal sort keys keep original relative order (stable sort)."""

    first = _entry(
        run_id="run-a",
        fetch_record_id="fetch-1",
        object_id="obj-first",
        kind="document",
        storage_relative_path="same/path.txt",
        content_sha256="aaa",
        byte_size=1,
    )
    second = _entry(
        run_id="run-a",
        fetch_record_id="fetch-1",
        object_id="obj-second",
        kind="image",
        storage_relative_path="same/path.txt",
        content_sha256="bbb",
        byte_size=2,
    )
    settings_payload = {}
    relevant_kinds = frozenset({"document", "image"})

    forward = (first, second)
    reverse = (second, first)

    # Different input order with equal sort keys yields different digests
    # under stable sort — both streaming and reference implementations must agree.
    forward_digest = build_curation_input_fingerprint(
        raw_entries=forward,
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    )
    reverse_digest = build_curation_input_fingerprint(
        raw_entries=reverse,
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    )
    assert forward_digest == _reference_build_curation_input_fingerprint(
        raw_entries=forward,
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    )
    assert reverse_digest == _reference_build_curation_input_fingerprint(
        raw_entries=reverse,
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    )
    assert forward_digest != reverse_digest


class _Dumpable:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return dict(self._payload)


def curation_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        near_deduper=_Dumpable({"threshold": 0.92}),
        document_chunker=_Dumpable({"chunk_target_tokens": 768}),
        document_assembler=_Dumpable({"require_allow_training": True}),
        writer=_Dumpable({"write_manifest": True}),
        builder=_Dumpable({"fail_on_empty_snapshot": True}),
    )


def base_settings() -> SimpleNamespace:
    return SimpleNamespace(
        sources=SimpleNamespace(active=_Dumpable({"source": "active"})),
        preprocessing=_Dumpable({"privacy": "strict"}),
        collection=SimpleNamespace(
            processors=_Dumpable({"audio": {"enabled": True}})
        ),
        datasets=SimpleNamespace(
            curation=curation_namespace(),
            paths=SimpleNamespace(
                output_subdirectory="multimodal",
                curated_entities_directory="entities",
                snapshot_manifest_filename="manifest.json",
            ),
        ),
    )


def test_snapshot_fingerprint_payload_projects_output_setting_groups() -> None:
    settings = SimpleNamespace(
        sources=SimpleNamespace(active=_Dumpable({"source": "active"})),
        preprocessing=_Dumpable({"privacy": "strict"}),
        collection=SimpleNamespace(
            processors=_Dumpable({"audio": {"enabled": True}})
        ),
        datasets=SimpleNamespace(
            curation=SimpleNamespace(
                near_deduper=_Dumpable({"threshold": 0.92}),
                document_chunker=_Dumpable({"chunk_target_tokens": 768}),
                document_assembler=_Dumpable({"require_allow_training": True}),
            ),
        ),
    )

    assert build_snapshot_fingerprint_payload(
        source_profile=settings.sources.active,
        preprocessing=settings.preprocessing,
        curation=settings.datasets.curation,
        processors=settings.collection.processors,
    ) == {
        "source_profile": {"source": "active"},
        "preprocessing": {"privacy": "strict"},
        "curation": {
            "near_deduper": {"threshold": 0.92},
            "document_chunker": {"chunk_target_tokens": 768},
            "document_assembler": {"require_allow_training": True},
        },
        "collection_processors": {"audio": {"enabled": True}},
    }


def test_snapshot_fingerprint_payload_ignores_operational_settings() -> None:
    """Operational curation groups must not appear in the digest payload."""

    baseline = _payload(base_settings())
    operational = base_settings()
    operational.datasets.curation.builder = _Dumpable(
        {"fail_on_empty_snapshot": False}
    )
    operational.datasets.curation.writer = _Dumpable({"write_manifest": False})
    changed = _payload(operational)

    assert changed == baseline


def test_snapshot_fingerprint_ignores_storage_layout_settings() -> None:
    baseline = _payload(base_settings())

    changed_settings = base_settings()
    changed_settings.datasets.paths.output_subdirectory = "other-output"
    changed_settings.datasets.paths.curated_entities_directory = "records"
    changed_settings.datasets.paths.snapshot_manifest_filename = "index.json"

    assert _payload(changed_settings) == baseline


def _payload(settings: object) -> dict[str, Any]:
    namespace = settings
    return build_snapshot_fingerprint_payload(
        source_profile=namespace.sources.active,
        preprocessing=namespace.preprocessing,
        curation=namespace.datasets.curation,
        processors=namespace.collection.processors,
    )
