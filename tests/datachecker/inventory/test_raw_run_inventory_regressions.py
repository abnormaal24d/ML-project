from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from datachecker.inventory.raw_run_inventory import (
    RawInventoryReader,
    ValidCurrentRecords,
)


def _write_summary(path: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "status": "completed",
        "final": True,
        "schema_version": "3.0",
        "lifecycle_stage": "raw",
        "raw_run_id": "test-run",
        "manifest_path": "records/objects.jsonl",
        "manifest_write_count": 1,
        "modality_counts": {"image": 1},
        "object_records_total": 1,
        "failed_url_count": 0,
        "required_records": [
            "records/objects.jsonl",
            "records/errors.jsonl",
            "records/discovered_asset_manifest.jsonl",
            "records/rejected_asset_manifest.jsonl",
            "records/current_objects.jsonl",
            "records/object_metadata.jsonl",
        ],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _current_records() -> ValidCurrentRecords:
    return ValidCurrentRecords(
        records=(
            {
                "object_id": "object-1",
                "stable_url_id": "object-1",
                "fetch_record_id": "fetch-1",
                "schema_version": "3.0",
                "run_id": "test-run",
                "modality": "image",
                "mime_type": "image/jpeg",
                "storage_relative_path": "objects/image.jpg",
                "byte_size": 100,
                "content_sha256": "a" * 64,
            },
        ),
        modality_counts={"image": 1},
        errors=(),
    )


def _make_reader(tmp_path: Path) -> RawInventoryReader:
    dataset_paths = SimpleNamespace(
        raw_sync_directory="records",
        manifest_filename="records/objects.jsonl",
        raw_sync_errors_filename="errors.jsonl",
        raw_sync_discovered_assets_filename="discovered_asset_manifest.jsonl",
        raw_sync_rejected_assets_filename="rejected_asset_manifest.jsonl",
        raw_sync_current_objects_filename="current_objects.jsonl",
        raw_sync_metadata_filename="object_metadata.jsonl",
        raw_sync_summary_filename="run_manifest.json",
    )
    return RawInventoryReader(
        settings=SimpleNamespace(),
        artifact_path_registry=SimpleNamespace(dataset_paths=dataset_paths),
        dataset_fingerprint_calculator=SimpleNamespace(
            calculate=lambda **_: "fingerprint"
        ),
        raw_schema_version="3.0",
    )


def test_raw_inventory_requires_regular_artifact_files(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path)
    records_path = tmp_path / "objects.jsonl"
    records_path.mkdir()
    errors_path = tmp_path / "errors.jsonl"
    errors_path.write_text("", encoding="utf-8")

    reader = _make_reader(tmp_path)
    valid, errors = reader._raw_run_schema_valid(
        run_directory=tmp_path,
        summary_path=summary_path,
        records_path=records_path,
        errors_path=errors_path,
        current_records=_current_records(),
        expected_summary_path=tmp_path / "records" / "run_manifest.json",
        expected_manifest_relative_path="records/objects.jsonl",
        expected_schema_version="3.0",
        expected_run_id="test-run",
    )

    assert valid is False
    assert errors == ("missing_objects_jsonl",)


def test_raw_inventory_rejects_unreconciled_or_unsafe_summary_fields(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    _write_summary(
        summary_path,
        modality_counts={"image": 1, "video": 999},
        failed_url_count="not-a-count",
        required_records=["../outside.jsonl", "directory"],
    )
    records_path = tmp_path / "objects.jsonl"
    records_path.write_text("", encoding="utf-8")
    errors_path = tmp_path / "errors.jsonl"
    errors_path.write_text("", encoding="utf-8")
    (tmp_path / "directory").mkdir()

    reader = _make_reader(tmp_path)
    valid, errors = reader._raw_run_schema_valid(
        run_directory=tmp_path,
        summary_path=summary_path,
        records_path=records_path,
        errors_path=errors_path,
        current_records=_current_records(),
        expected_summary_path=tmp_path / "records" / "run_manifest.json",
        expected_manifest_relative_path="records/objects.jsonl",
        expected_schema_version="3.0",
        expected_run_id="test-run",
    )

    assert valid is False
    assert "modality_mismatch:video:999!=0" in errors
    assert "failed_url_count_invalid" in errors
    assert "required_record_escapes_run:../outside.jsonl" in errors
    assert "missing_required:directory" in errors
    assert RawInventoryReader._summary_count("not-a-count") == 0


def test_raw_inventory_marks_invalid_failed_url_count_without_crashing(
    tmp_path: Path,
) -> None:
    records_directory = tmp_path / "records"
    records_directory.mkdir()
    (records_directory / "objects.jsonl").write_text("", encoding="utf-8")
    (records_directory / "errors.jsonl").write_text("", encoding="utf-8")
    summary_path = tmp_path / "run_manifest.json"
    _write_summary(summary_path, failed_url_count="not-a-count")

    reader = _make_reader(tmp_path)

    inventory = reader.read(
        raw_run_directory=tmp_path,
        run_summary_path=summary_path,
    )

    assert inventory.failed_url_count == 0
    assert inventory.schema_valid is False
    assert "failed_url_count_invalid" in inventory.raw_schema_errors
