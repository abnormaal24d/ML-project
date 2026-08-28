from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.load import load_settings
from config.settings.datasets import DatasetPathSettings
from mmcrawler_datasets.dataset import iter_raw_records, resolve_split_paths
from mmcrawler_datasets.schema import DatasetSplit
from mmcrawler_datasets.snapshots.output_writer import (
    SnapshotOutputSettings,
    SnapshotShardWriteError,
    write_shard_index,
    write_snapshot_rows,
    write_webdataset_shards,
)
from mmcrawler_datasets.snapshots.publication import staged_snapshot
from mmcrawler_datasets.snapshots.rejected_sample_reports import (
    ensure_rejected_rows_report,
    write_pair_rejections,
)
from mmcrawler_datasets.snapshots.validation import validate_snapshot


def test_shard_only_output_round_trips_through_loader(
    tmp_path: Path,
) -> None:
    snapshot_root, paths, rows = _write_snapshot(
        tmp_path=tmp_path,
        write_jsonl=False,
        write_shards=True,
        max_samples_per_shard=2,
    )

    assert validate_snapshot(
        training_directory=snapshot_root,
        dataset_paths=paths,
        write_jsonl=False,
        write_shards=True,
        shard_index_filename="shard_index.json",
    )

    shard_paths = resolve_split_paths(
        dataset_root=snapshot_root,
        split=DatasetSplit.TRAIN,
    )
    assert [path.name for path in shard_paths] == [
        "train-00000.tar",
        "train-00001.tar",
    ]
    assert tuple(iter_raw_records(paths=shard_paths)) == rows["train"]


@pytest.mark.parametrize(
    ("write_jsonl", "write_shards"),
    ((True, False), (False, True)),
)
def test_each_configured_output_mode_is_structurally_valid(
    tmp_path: Path,
    write_jsonl: bool,
    write_shards: bool,
) -> None:
    snapshot_root, paths, _rows = _write_snapshot(
        tmp_path=tmp_path,
        write_jsonl=write_jsonl,
        write_shards=write_shards,
        max_samples_per_shard=2,
    )

    assert validate_snapshot(
        training_directory=snapshot_root,
        dataset_paths=paths,
        write_jsonl=write_jsonl,
        write_shards=write_shards,
        shard_index_filename="shard_index.json",
    )


def test_corrupt_shard_fails_validation_and_loader_closed(
    tmp_path: Path,
) -> None:
    snapshot_root, paths, _rows = _write_snapshot(
        tmp_path=tmp_path,
        write_jsonl=False,
        write_shards=True,
        max_samples_per_shard=2,
    )
    shard_path = snapshot_root / "shards" / "train-00000.tar"
    with shard_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"x")

    assert not validate_snapshot(
        training_directory=snapshot_root,
        dataset_paths=paths,
        write_jsonl=False,
        write_shards=True,
        shard_index_filename="shard_index.json",
    )
    with pytest.raises(ValueError, match="checksum"):
        resolve_split_paths(
            dataset_root=snapshot_root,
            split=DatasetSplit.TRAIN,
        )


def test_corrupt_declared_shard_fails_closed_even_with_jsonl_available(
    tmp_path: Path,
) -> None:
    snapshot_root, _paths, _rows = _write_snapshot(
        tmp_path=tmp_path,
        write_jsonl=True,
        write_shards=True,
        max_samples_per_shard=2,
    )
    shard_path = snapshot_root / "shards" / "train-00000.tar"
    with shard_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"x")

    with pytest.raises(ValueError, match="checksum"):
        resolve_split_paths(
            dataset_root=snapshot_root,
            split=DatasetSplit.TRAIN,
        )


@pytest.mark.usefixtures("production_whisper_env")
def test_production_writer_settings_publish_webdataset_shards(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    settings = load_settings(
        "prod",
        project_root=project_root,
        environment="prod",
    )
    writer = settings.datasets.training.writer
    output_settings = SnapshotOutputSettings(
        write_jsonl=writer.write_jsonl,
        write_shards=writer.write_shards,
        shard_format=writer.shard_format,
        max_samples_per_shard=writer.shard_max_samples,
        max_bytes_per_shard=writer.shard_max_bytes,
        shards_directory=writer.training_shards_directory,
        shard_index_filename=writer.shard_index_filename,
    )

    snapshot_root, paths, rows = _write_snapshot(
        tmp_path=tmp_path,
        write_jsonl=output_settings.write_jsonl,
        write_shards=output_settings.write_shards,
        max_samples_per_shard=output_settings.max_samples_per_shard,
        output_settings=output_settings,
    )

    assert validate_snapshot(
        training_directory=snapshot_root,
        dataset_paths=paths,
        write_jsonl=output_settings.write_jsonl,
        write_shards=output_settings.write_shards,
        shard_index_filename=output_settings.shard_index_filename,
    )
    shard_paths = tuple(
        (snapshot_root / output_settings.shards_directory).glob("*.tar")
    )
    assert shard_paths
    assert (snapshot_root / output_settings.shard_index_filename).is_file()
    assert (
        tuple(
            iter_raw_records(
                paths=(
                    snapshot_root
                    / output_settings.shards_directory
                    / "train-00000.tar",
                )
            )
        )
        == rows["train"]
    )


def test_shard_byte_limit_fails_before_an_output_is_published(
    tmp_path: Path,
) -> None:
    output_settings = _output_settings(
        write_jsonl=False,
        write_shards=True,
        max_samples_per_shard=1,
        max_bytes_per_shard=10_239,
    )

    with pytest.raises(SnapshotShardWriteError, match="byte limit"):
        write_webdataset_shards(
            training_directory=tmp_path,
            rows_by_split={
                "train": ({"sample_id": "sample-1", "split": "train"},),
                "val": (),
                "test": (),
            },
            output_settings=output_settings,
        )

    assert not (tmp_path / "shards" / "train-00000.tar").exists()


def test_failed_staged_build_preserves_the_previously_published_snapshot(
    tmp_path: Path,
) -> None:
    target = tmp_path / "snapshot"
    target.mkdir()
    marker = target / "published.txt"
    marker.write_text("previous", encoding="utf-8")

    with pytest.raises(RuntimeError, match="interrupted"):
        with staged_snapshot(final_snapshot_root=target) as staging:
            (staging / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("interrupted shard write")

    assert marker.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".staging-snapshot-*"))


def _write_snapshot(
    *,
    tmp_path: Path,
    write_jsonl: bool,
    write_shards: bool,
    max_samples_per_shard: int,
    output_settings: SnapshotOutputSettings | None = None,
) -> tuple[
    Path,
    DatasetPathSettings,
    dict[str, tuple[dict[str, object], ...]],
]:
    paths = DatasetPathSettings()
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    rows = {
        "train": (
            {"sample_id": "train-1", "split": "train"},
            {"sample_id": "train-2", "split": "train"},
            {"sample_id": "train-3", "split": "train"},
        ),
        "val": ({"sample_id": "val-1", "split": "val"},),
        "test": (),
    }
    if output_settings is None:
        output_settings = _output_settings(
            write_jsonl=write_jsonl,
            write_shards=write_shards,
            max_samples_per_shard=max_samples_per_shard,
            max_bytes_per_shard=None,
        )

    output_paths: dict[str, object] = {}
    if write_jsonl:
        splits_root = snapshot_root / paths.training_splits_directory
        write_snapshot_rows(
            splits_root / paths.training_train_filename,
            rows["train"],
        )
        write_snapshot_rows(
            splits_root / paths.training_val_filename,
            rows["val"],
        )
        write_snapshot_rows(
            splits_root / paths.training_test_filename,
            rows["test"],
        )
        output_paths["splits"] = {
            "train": (
                Path(paths.training_splits_directory)
                / paths.training_train_filename
            ).as_posix(),
            "val": (
                Path(paths.training_splits_directory)
                / paths.training_val_filename
            ).as_posix(),
            "test": (
                Path(paths.training_splits_directory)
                / paths.training_test_filename
            ).as_posix(),
        }
    if write_shards:
        entries = write_webdataset_shards(
            training_directory=snapshot_root,
            rows_by_split=rows,
            output_settings=output_settings,
        )
        write_shard_index(
            path=snapshot_root / output_settings.shard_index_filename,
            entries_by_split=entries,
        )
        output_paths["shard_index"] = output_settings.shard_index_filename

    write_pair_rejections(training_directory=snapshot_root, rows=())
    ensure_rejected_rows_report(training_directory=snapshot_root)
    (snapshot_root / paths.dataset_manifest_filename).write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "splits": {
                    split: len(split_rows)
                    for split, split_rows in rows.items()
                },
                "paths": output_paths,
                "outputs": {
                    "jsonl": write_jsonl,
                    "shards": write_shards,
                    "shard_format": (
                        "webdataset_tar" if write_shards else None
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    (snapshot_root / paths.stats_filename).write_text(
        json.dumps({"schema_version": "3.0"}),
        encoding="utf-8",
    )
    return snapshot_root, paths, rows


def _output_settings(
    *,
    write_jsonl: bool,
    write_shards: bool,
    max_samples_per_shard: int,
    max_bytes_per_shard: int | None,
) -> SnapshotOutputSettings:
    return SnapshotOutputSettings(
        write_jsonl=write_jsonl,
        write_shards=write_shards,
        shard_format="webdataset_tar",
        max_samples_per_shard=max_samples_per_shard,
        max_bytes_per_shard=max_bytes_per_shard,
        shards_directory="shards",
        shard_index_filename="shard_index.json",
    )
