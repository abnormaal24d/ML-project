from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from crawler.curation.ingest.curation_input_loader import (
    CurationInputLoader,
)
from crawler.curation.ingest.schema.record import RawManifestRecord


def _v3_row() -> dict[str, object]:
    return {
        "schema_version": "3.0",
        "run_id": "run-1",
        "fetch_record_id": "fetch-1",
        "object_id": "object-1",
        "requested_url": "https://example.test/a",
        "final_url": "https://example.test/a",
        "normalized_url": "https://example.test/a",
        "parent_url": None,
        "kind": "page",
        "modality": "document",
        "depth": 0,
        "source_type": "registry",
        "status_code": 200,
        "content_sha256": "a" * 64,
        "byte_size": 12,
        "observed_bytes": 12,
        "storage_relative_path": "objects/a.bin",
        "domain": "example.test",
        "path": "/a",
        "fetched_at": "2026-07-19T00:00:00+00:00",
    }


def test_v3_raw_manifest_requires_canonical_fields() -> None:
    record = RawManifestRecord.from_payload(_v3_row())
    assert record.fetch_record_id == "fetch-1"
    assert record.modality == "document"

    language_row = _v3_row()
    language_row.update(
        {
            "language": "nl",
            "language_confidence": 0.97,
            "language_source": "fasttext",
            "language_detector_version": "crawler-language-v1",
        }
    )
    language_record = RawManifestRecord.from_payload(language_row)
    assert language_record.language == "nl"
    assert language_record.language_confidence == 0.97
    assert language_record.language_source == "fasttext"
    assert language_record.language_detector_version == "crawler-language-v1"

    row = _v3_row()
    del row["fetch_record_id"]
    with pytest.raises(ValueError, match="fetch_record_id"):
        RawManifestRecord.from_payload(row)


def test_manifest_normalizes_language_code() -> None:
    row = _v3_row()
    row["language"] = " NL "

    record = RawManifestRecord.from_payload(row)

    assert record.language == "nl"


def test_raw_manifest_rejects_invalid_language_confidence() -> None:
    row = _v3_row()
    row["language_confidence"] = 1.5

    with pytest.raises(ValueError, match="language_confidence"):
        RawManifestRecord.from_payload(row)


@pytest.mark.parametrize(
    "storage_relative_path",
    (
        "../../README.md",
        "/absolute/object.bin",
        "C:/absolute/object.bin",
        r"C:\\absolute\\object.bin",
        r"C:drive-relative.bin",
        r"\\\\server\\share\\object.bin",
        r"objects\\alternate-separator.bin",
        "objects/object.bin:alternate-stream",
    ),
)
def test_raw_manifest_rejects_unsafe_storage_paths(
    storage_relative_path: str,
) -> None:
    row = _v3_row()
    row["storage_relative_path"] = storage_relative_path

    with pytest.raises(ValueError, match="storage_relative_path"):
        RawManifestRecord.from_payload(row)


@pytest.mark.parametrize("byte_size", (-1, True, 1.5, "12"))
def test_raw_manifest_rejects_noncanonical_byte_size(
    byte_size: object,
) -> None:
    row = _v3_row()
    row["byte_size"] = byte_size

    with pytest.raises(ValueError, match="byte_size"):
        RawManifestRecord.from_payload(row)


@pytest.mark.parametrize(
    "content_sha256",
    ("", "a" * 63, "g" * 64, "sha256:not-a-digest"),
)
def test_raw_manifest_rejects_invalid_content_sha256(
    content_sha256: str,
) -> None:
    row = _v3_row()
    row["content_sha256"] = content_sha256

    with pytest.raises(ValueError, match="content_sha256"):
        RawManifestRecord.from_payload(row)


def test_v2_rows_are_rejected() -> None:
    row = _v3_row()
    row["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="Unsupported|unsupported|schema"):
        RawManifestRecord.from_payload(row)


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[dict[str, object]] = []

    def warning(self, event: str, **values: object) -> None:
        self.warnings.append({"event": event, **values})


def _loader(*, project_root: Path, logger: _Logger) -> CurationInputLoader:
    return CurationInputLoader(
        settings=SimpleNamespace(
            raw_schema_version="3.0",
            selected_run_ids=(),
        ),
        dataset_paths=_dataset_paths(),
        project_root=project_root,
        logger=logger,
    )


def _dataset_paths() -> SimpleNamespace:
    return SimpleNamespace(
        output_directory="data/datasets",
        output_subdirectory="multimodal",
        raw_output_directory="data/raw",
        raw_sync_directory="records",
        raw_sync_current_objects_filename="current_objects.jsonl",
    )


def test_manifest_reader_skips_only_malformed_rows(tmp_path: Path) -> None:
    manifest_path = tmp_path / "records.jsonl"
    manifest_path.write_text(
        "\n".join(
            (
                "",
                "not-json",
                json.dumps(["not", "an", "object"]),
                json.dumps(_v3_row()),
            )
        ),
        encoding="utf-8",
    )
    logger = _Logger()

    records = tuple(
        _loader(project_root=tmp_path, logger=logger)._iter_manifest_records(
            manifest_path=manifest_path
        )
    )

    assert [record.fetch_record_id for record in records] == ["fetch-1"]
    assert [warning["line"] for warning in logger.warnings] == [2, 3]


def test_manifest_reader_rejects_schema_drift_fail_closed(
    tmp_path: Path,
) -> None:
    row = _v3_row()
    row["schema_version"] = "2.0"
    manifest_path = tmp_path / "records.jsonl"
    manifest_path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(ValueError, match="schema mismatch"):
        tuple(
            _loader(
                project_root=tmp_path,
                logger=_Logger(),
            )._iter_manifest_records(manifest_path=manifest_path)
        )


def test_manifest_kind_count_uses_the_record_iterator(tmp_path: Path) -> None:
    manifest_path = tmp_path / "records.jsonl"
    manifest_path.write_text(
        "\n".join(("not-json", '{"kind": "video"}', json.dumps(_v3_row()))),
        encoding="utf-8",
    )
    loader = _loader(project_root=tmp_path, logger=_Logger())

    assert loader._count_manifest_kinds(manifest_path=manifest_path) == {
        "page": 1
    }
    assert (
        len(tuple(loader._iter_manifest_records(manifest_path=manifest_path)))
        == 1
    )


def _selection_loader(
    *,
    project_root: Path,
    logger: _Logger,
    selected_run_ids: tuple[str, ...] = (),
) -> CurationInputLoader:
    return CurationInputLoader(
        settings=SimpleNamespace(
            raw_schema_version="3.0",
            run_selection_mode="all",
            selected_run_ids=selected_run_ids,
            coverage_selection_max_runs=2,
        ),
        dataset_paths=_dataset_paths(),
        project_root=project_root,
        logger=logger,
    )


def _coverage_loader(
    *,
    project_root: Path,
    logger: _Logger,
    minimums: dict[str, int],
    max_runs: int = 4,
) -> CurationInputLoader:
    return CurationInputLoader(
        settings=SimpleNamespace(
            raw_schema_version="3.0",
            run_selection_mode="coverage_combined",
            selected_run_ids=(),
            coverage_selection_max_runs=max_runs,
        ),
        dataset_paths=_dataset_paths(),
        project_root=project_root,
        logger=logger,
        minimum_modality_counts=minimums,
    )


def _write_run(
    *,
    project_root: Path,
    run_id: str,
    rows: list[dict[str, object]],
    final: bool = True,
    output_readiness: dict[str, object] | None = None,
) -> None:
    run_directory = project_root / "data" / "raw" / "multimodal" / run_id
    records_directory = run_directory / "records"
    records_directory.mkdir(parents=True, exist_ok=True)
    (records_directory / "current_objects.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    summary = {
        "status": "completed" if final else "incomplete",
        "final": final,
    }
    if output_readiness is not None:
        summary["output_readiness"] = output_readiness
    (records_directory / "run_manifest.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


def _kind_row(*, kind: str) -> dict[str, object]:
    row = _v3_row()
    row["kind"] = kind
    return row


def test_selected_modality_counts_aggregates_final_runs(
    tmp_path: Path,
) -> None:
    _write_run(
        project_root=tmp_path,
        run_id="run-a",
        rows=[_kind_row(kind="document"), _kind_row(kind="image")],
    )
    _write_run(
        project_root=tmp_path,
        run_id="run-b",
        rows=[_kind_row(kind="audio"), _kind_row(kind="audio")],
    )

    counts = _selection_loader(
        project_root=tmp_path,
        logger=_Logger(),
    ).selected_modality_counts()

    assert counts == {"document": 1, "image": 1, "audio": 2}


def test_selected_modality_counts_excludes_non_final_runs(
    tmp_path: Path,
) -> None:
    _write_run(
        project_root=tmp_path,
        run_id="run-final",
        rows=[_kind_row(kind="video")],
    )
    _write_run(
        project_root=tmp_path,
        run_id="run-interrupted",
        rows=[_kind_row(kind="audio")],
        final=False,
    )

    counts = _selection_loader(
        project_root=tmp_path,
        logger=_Logger(),
    ).selected_modality_counts()

    assert counts == {"video": 1}


def test_selected_record_count_sums_selected_runs(tmp_path: Path) -> None:
    _write_run(
        project_root=tmp_path,
        run_id="run-a",
        rows=[_kind_row(kind="document"), _kind_row(kind="document")],
    )
    _write_run(
        project_root=tmp_path,
        run_id="run-b",
        rows=[_kind_row(kind="audio")],
    )

    loader = _selection_loader(project_root=tmp_path, logger=_Logger())

    assert loader.selected_record_count() == 3


def test_coverage_combined_combines_runs_across_deficits(
    tmp_path: Path,
) -> None:
    _write_run(
        project_root=tmp_path,
        run_id="run-a",
        rows=(
            [_kind_row(kind="document") for _ in range(10)]
            + [_kind_row(kind="image") for _ in range(13)]
            + [_kind_row(kind="audio") for _ in range(1)]
            + [_kind_row(kind="video") for _ in range(1)]
        ),
    )
    _write_run(
        project_root=tmp_path,
        run_id="run-b",
        rows=(
            [_kind_row(kind="document") for _ in range(5)]
            + [_kind_row(kind="image") for _ in range(7)]
            + [_kind_row(kind="audio") for _ in range(4)]
            + [_kind_row(kind="video") for _ in range(4)]
        ),
    )

    loader = _coverage_loader(
        project_root=tmp_path,
        logger=_Logger(),
        minimums={
            "document": 15,
            "image": 20,
            "audio": 5,
            "video": 5,
        },
    )

    specs = loader._selected_manifest_specs()
    run_ids = {
        loader._resolve_run_directory(
            manifest_path=path,
            manifest_name=name,
        ).name
        for path, name in specs
    }
    assert run_ids == {"run-a", "run-b"}
    assert loader.selected_modality_counts() == {
        "document": 15,
        "image": 20,
        "audio": 5,
        "video": 5,
    }


def test_coverage_combined_stops_once_minima_are_satisfied(
    tmp_path: Path,
) -> None:
    _write_run(
        project_root=tmp_path,
        run_id="run-alone",
        rows=(
            [_kind_row(kind="document") for _ in range(15)]
            + [_kind_row(kind="image") for _ in range(20)]
            + [_kind_row(kind="audio") for _ in range(5)]
            + [_kind_row(kind="video") for _ in range(5)]
        ),
    )
    _write_run(
        project_root=tmp_path,
        run_id="run-sparse",
        rows=[_kind_row(kind="video")],
    )

    loader = _coverage_loader(
        project_root=tmp_path,
        logger=_Logger(),
        minimums={
            "document": 15,
            "image": 20,
            "audio": 5,
            "video": 5,
        },
        max_runs=4,
    )

    specs = loader._selected_manifest_specs()
    run_ids = {
        loader._resolve_run_directory(
            manifest_path=path,
            manifest_name=name,
        ).name
        for path, name in specs
    }
    assert run_ids == {"run-alone"}


def test_coverage_combined_falls_back_to_best_run_when_no_combination_fits(
    tmp_path: Path,
) -> None:
    _write_run(
        project_root=tmp_path,
        run_id="run-doc",
        rows=[_kind_row(kind="document") for _ in range(16)],
    )
    _write_run(
        project_root=tmp_path,
        run_id="run-audio",
        rows=[_kind_row(kind="audio") for _ in range(6)],
    )
    _write_run(
        project_root=tmp_path,
        run_id="run-video",
        rows=[_kind_row(kind="video") for _ in range(6)],
    )

    loader = _coverage_loader(
        project_root=tmp_path,
        logger=_Logger(),
        minimums={"document": 15, "audio": 5, "video": 5},
        max_runs=2,
    )

    specs = loader._selected_manifest_specs()
    run_ids = {
        loader._resolve_run_directory(
            manifest_path=path,
            manifest_name=name,
        ).name
        for path, name in specs
    }
    assert run_ids == {"run-video"}


def test_coverage_combined_prefers_satisfying_combination_over_score_order(
    tmp_path: Path,
) -> None:
    _write_run(
        project_root=tmp_path,
        run_id="run-a",
        rows=(
            [_kind_row(kind="audio")]
            + [_kind_row(kind="document") for _ in range(25)]
            + [_kind_row(kind="image") for _ in range(25)]
        ),
    )
    _write_run(
        project_root=tmp_path,
        run_id="run-c",
        rows=[_kind_row(kind="video") for _ in range(5)],
    )
    _write_run(
        project_root=tmp_path,
        run_id="run-b",
        rows=[_kind_row(kind="audio") for _ in range(5)],
    )

    loader = _coverage_loader(
        project_root=tmp_path,
        logger=_Logger(),
        minimums={"audio": 5, "video": 5},
        max_runs=2,
    )

    specs = loader._selected_manifest_specs()
    run_ids = {
        loader._resolve_run_directory(
            manifest_path=path,
            manifest_name=name,
        ).name
        for path, name in specs
    }
    assert run_ids == {"run-b", "run-c"}
    assert loader.selected_modality_counts() == {"audio": 5, "video": 5}


def test_selected_crawl_evidence_aggregates_readiness_across_runs(
    tmp_path: Path,
) -> None:
    _write_run(
        project_root=tmp_path,
        run_id="run-a",
        rows=[_kind_row(kind="document") for _ in range(10)],
        output_readiness={
            "object_records_total": 10,
            "successful_requests_total": 20,
            "quality_score": 0.5,
        },
    )
    _write_run(
        project_root=tmp_path,
        run_id="run-b",
        rows=[_kind_row(kind="document") for _ in range(30)],
        output_readiness={
            "object_records_total": 30,
            "successful_requests_total": 40,
            "quality_score": 0.9,
        },
    )

    evidence = _selection_loader(
        project_root=tmp_path,
        logger=_Logger(),
    ).selected_crawl_evidence()

    assert evidence is not None
    assert evidence.object_records_total == 40
    assert evidence.successful_requests_total == 60
    assert evidence.quality_score == pytest.approx(0.8)


def test_selected_crawl_evidence_none_without_readiness_reports(
    tmp_path: Path,
) -> None:
    _write_run(
        project_root=tmp_path,
        run_id="run-a",
        rows=[_kind_row(kind="document")],
    )

    evidence = _selection_loader(
        project_root=tmp_path,
        logger=_Logger(),
    ).selected_crawl_evidence()

    assert evidence is None


def test_selected_crawl_evidence_fail_closed_when_any_run_lacks_readiness(
    tmp_path: Path,
) -> None:
    _write_run(
        project_root=tmp_path,
        run_id="run-a",
        rows=[_kind_row(kind="document")],
        output_readiness={
            "object_records_total": 10,
            "successful_requests_total": 20,
            "quality_score": 0.5,
        },
    )
    _write_run(
        project_root=tmp_path,
        run_id="run-b",
        rows=[_kind_row(kind="document")],
    )

    evidence = _selection_loader(
        project_root=tmp_path,
        logger=_Logger(),
    ).selected_crawl_evidence()

    assert evidence is None
