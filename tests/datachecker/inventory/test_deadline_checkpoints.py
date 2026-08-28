"""Deadline checkpoint coverage for concrete artifact inventory scans."""

from __future__ import annotations

import json
from pathlib import Path

from datachecker.fingerprints import (
    DatasetFingerprintCalculator,
    FileFingerprintCalculator,
)
from datachecker.inventory.raw_run_inventory import RawInventoryReader


def _make_valid_record(
    run_id: str, storage_relative_path: str
) -> dict[str, object]:
    # Use the actual SHA256 of "<html></html>"
    content_sha256 = (
        "b633a587c652d02386c4f16f8c6f6aab7352d97f16367c3c40576214372dd628"
    )
    return {
        "schema_version": "3.0",
        "item_id": content_sha256,
        "fetch_record_id": "f" * 24,
        "stable_url_id": "a" * 24,
        "object_id": content_sha256,
        "run_id": run_id,
        "requested_url": "https://example.com/page",
        "final_url": "https://example.com/page",
        "normalized_url": "https://example.com/page",
        "domain": "example.com",
        "path": "/page",
        "query": None,
        "extension": "html",
        "parent_url": None,
        "referrer_url": None,
        "kind": "page",
        "modality": "page",
        "depth": 0,
        "source_type": "page",
        "fetch_attempt": 1,
        "status_code": 200,
        "content_type": "text/html",
        "mime_type": "text/html",
        "encoding": "utf-8",
        "language": "en",
        "content_sha256": content_sha256,
        "byte_size": 13,  # len("<html></html>")
        "storage_relative_path": storage_relative_path,
        "http_etag": None,
        "http_last_modified": None,
        "fetched_at": "2026-01-01T00:00:00Z",
        "governance": {
            "source": {
                "id": "example.com",
                "name": "example.com",
                "rules_version": "1",
                "registry_version": "1",
            },
            "collection": {
                "allowed": True,
                "reason": "rules_collection_allowed",
            },
            "access": {
                "checked": True,
                "decision": "allow",
                "reason": "from_robots_or_context",
                "robots_url": "https://example.com/robots.txt",
                "user_agent": "MultimodalCrawler/1.0",
                "fetched_at": "2026-01-01T00:00:00Z",
                "cache_expires_at": None,
                "crawl_delay_seconds": None,
            },
            "rights": {
                "checked": True,
                "decision": "allow",
                "expression": "cc-by-4.0",
                "evidence_url": "https://creativecommons.org/licenses/by/4.0/",
                "evidence_kind": "source_registry",
                "reason": "rights_evidence_verified",
                "rights_reserved": False,
                "tdm_allowed": True,
                "commercial_use_allowed": True,
                "attribution_required": True,
                "review_expires_at": None,
                "rules_version": "1",
            },
            "privacy": {
                "checked": True,
                "result": "pass",
                "action": "none",
                "reason": "explicit_governance_check",
            },
            "dedupe": {
                "checked": True,
                "result": "pass",
                "content_hash": content_sha256,
                "duplicate_of": None,
            },
            "quality": {
                "checked": True,
                "result": "pass",
                "reason": "explicit_governance_check",
            },
            "lineage": {
                "complete": True,
                "requested_url": "https://example.com/page",
                "final_url": "https://example.com/page",
                "origin": "example.com",
                "fetched_at": "2026-01-01T00:00:00Z",
                "run_id": run_id,
            },
            "training": {
                "allowed": True,
                "reason": "rules",
                "rules_version": "1",
            },
        },
    }


def test_raw_current_objects_scan_checkpoints_every_256_records(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    page_path = run_directory / "page.html"
    page_path.write_text("<html></html>", encoding="utf-8")
    current_objects_path = run_directory / "current_objects.jsonl"
    record = _make_valid_record("test-run", page_path.name)
    current_objects_path.write_text(
        "\n".join(json.dumps(record) for _ in range(256)),
        encoding="utf-8",
    )
    events: list[str] = []

    records = RawInventoryReader.valid_current_records(
        run_directory=run_directory,
        current_objects_path=current_objects_path,
        expected_schema_version="3.0",
        expected_run_id="test-run",
        checkpoint=events.append,
    )

    assert records.valid_count == 1
    assert events == ["raw_current_objects_scan"]


def test_dataset_fingerprints_checkpoint_large_file_and_file_sets(
    tmp_path: Path,
) -> None:
    paths = tuple(tmp_path / f"artifact-{index:02}.txt" for index in range(32))
    for path in paths:
        path.write_text(path.name, encoding="utf-8")
    events: list[str] = []
    calculator = DatasetFingerprintCalculator(
        file_fingerprint_calculator=FileFingerprintCalculator(),
    )

    digest = calculator.calculate(
        paths=paths,
        root=tmp_path,
        checkpoint=events.append,
    )

    assert digest
    assert events == ["dataset_fingerprint_scan"]

    large_file = tmp_path / "large.bin"
    large_file.write_bytes(b"x" * (65_536 * 32))
    events.clear()

    FileFingerprintCalculator().calculate(
        path=large_file,
        checkpoint=events.append,
    )

    assert events == ["file_fingerprint_scan"]
