"""Canonical atomic writers for persisted training snapshot outputs."""

from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from mmcrawler_datasets.snapshots.shards import (
    SHARD_SPLITS,
    WEB_DATASET_TAR_FORMAT,
    ShardIndexEntry,
    build_shard_index_payload,
    sha256_file,
    shard_filename,
    shard_record_member_name,
)


@dataclass(frozen=True, slots=True)
class SnapshotOutputSettings:
    """Explicit output dependency for a training snapshot build."""

    write_jsonl: bool
    write_shards: bool
    shard_format: Literal["webdataset_tar"]
    max_samples_per_shard: int
    max_bytes_per_shard: int | None
    shards_directory: str
    shard_index_filename: str

    def __post_init__(self) -> None:
        if not self.write_jsonl and not self.write_shards:
            raise ValueError(
                "at least one training snapshot output must be enabled"
            )
        if self.shard_format != WEB_DATASET_TAR_FORMAT:
            raise ValueError(
                f"unsupported shard format: {self.shard_format!r}"
            )
        if self.max_samples_per_shard < 1:
            raise ValueError("max_samples_per_shard must be positive")
        if (
            self.max_bytes_per_shard is not None
            and self.max_bytes_per_shard < 1
        ):
            raise ValueError("max_bytes_per_shard must be positive when set")
        _require_relative_output_path(
            self.shards_directory,
            label="shards_directory",
        )
        _require_relative_output_path(
            self.shard_index_filename,
            label="shard_index_filename",
        )
        if not self.shard_index_filename.endswith(".json"):
            raise ValueError("shard_index_filename must end with '.json'")


class SnapshotShardWriteError(RuntimeError):
    """Raised when a requested WebDataset shard output cannot be produced."""


def write_snapshot_metadata(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Write one formatted snapshot metadata object atomically."""

    _write_json(path=path, payload=payload, indent=2)


def write_snapshot_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Write snapshot samples as newline-delimited records atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(path)
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_webdataset_shards(
    *,
    training_directory: Path,
    rows_by_split: Mapping[str, Iterable[Mapping[str, object]]],
    output_settings: SnapshotOutputSettings,
) -> dict[str, tuple[ShardIndexEntry, ...]]:
    """Write immutable WebDataset tar shards and return their index entries."""

    if not output_settings.write_shards:
        raise SnapshotShardWriteError(
            "WebDataset shard writer was invoked while shard output is disabled"
        )
    if output_settings.shard_format != WEB_DATASET_TAR_FORMAT:
        raise SnapshotShardWriteError(
            f"unsupported shard format: {output_settings.shard_format!r}"
        )

    missing_splits = sorted(set(SHARD_SPLITS) - set(rows_by_split))
    unknown_splits = sorted(set(rows_by_split) - set(SHARD_SPLITS))
    if missing_splits or unknown_splits:
        raise SnapshotShardWriteError(
            "shard rows must contain exactly train, val, and test: "
            f"missing={missing_splits} unknown={unknown_splits}"
        )

    shards_directory = training_directory / output_settings.shards_directory
    _remove_previous_shards(
        training_directory=training_directory,
        shards_directory=shards_directory,
    )
    shards_directory.mkdir(parents=True, exist_ok=True)
    entries_by_split: dict[str, tuple[ShardIndexEntry, ...]] = {}

    for split in SHARD_SPLITS:
        serialized_rows = _serialize_split_rows(
            split=split,
            rows=rows_by_split[split],
        )
        entries: list[ShardIndexEntry] = []
        for shard_number, records in enumerate(
            _split_shard_records(
                records=serialized_rows,
                max_samples=output_settings.max_samples_per_shard,
                max_bytes=output_settings.max_bytes_per_shard,
            )
        ):
            shard_path = shards_directory / shard_filename(
                split=split,
                shard_number=shard_number,
            )
            _write_tar_shard(path=shard_path, records=records)
            byte_size = shard_path.stat().st_size
            if (
                output_settings.max_bytes_per_shard is not None
                and byte_size > output_settings.max_bytes_per_shard
            ):
                raise SnapshotShardWriteError(
                    "written shard exceeds configured maximum byte size: "
                    f"{shard_path}"
                )
            relative_path = shard_path.relative_to(
                training_directory
            ).as_posix()
            entry: ShardIndexEntry = {
                "path": relative_path,
                "sample_count": len(records),
                "byte_size": byte_size,
                "sha256": sha256_file(shard_path),
            }
            entries.append(entry)
        entries_by_split[split] = tuple(entries)

    return entries_by_split


def write_shard_index(
    *,
    path: Path,
    entries_by_split: dict[str, tuple[ShardIndexEntry, ...]],
) -> None:
    """Write the checksum-bearing index consumed by the dataset loader."""

    _write_json(
        path=path,
        payload=build_shard_index_payload(entries_by_split=entries_by_split),
        indent=2,
    )


@dataclass(frozen=True, slots=True)
class _SerializedShardRecord:
    member_name: str
    payload: bytes


def _serialize_split_rows(
    *,
    split: str,
    rows: Iterable[Mapping[str, object]],
) -> tuple[_SerializedShardRecord, ...]:
    serialized: list[_SerializedShardRecord] = []
    for sample_number, row in enumerate(rows):
        try:
            payload = json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SnapshotShardWriteError(
                f"cannot serialize {split} sample {sample_number} into a shard"
            ) from exc
        serialized.append(
            _SerializedShardRecord(
                member_name=shard_record_member_name(
                    split=split,
                    sample_number=sample_number,
                ),
                payload=payload,
            )
        )
    return tuple(serialized)


def _split_shard_records(
    *,
    records: tuple[_SerializedShardRecord, ...],
    max_samples: int,
    max_bytes: int | None,
) -> tuple[tuple[_SerializedShardRecord, ...], ...]:
    shards: list[tuple[_SerializedShardRecord, ...]] = []
    current: list[_SerializedShardRecord] = []

    for record in records:
        proposed = [*current, record]
        if current and (
            len(proposed) > max_samples
            or _exceeds_max_bytes(records=proposed, max_bytes=max_bytes)
        ):
            shards.append(tuple(current))
            current = [record]
        else:
            current = proposed

        if _exceeds_max_bytes(records=current, max_bytes=max_bytes):
            raise SnapshotShardWriteError(
                "one serialized sample exceeds the configured shard byte limit"
            )

    if current:
        shards.append(tuple(current))
    return tuple(shards)


def _exceeds_max_bytes(
    *,
    records: list[_SerializedShardRecord],
    max_bytes: int | None,
) -> bool:
    if max_bytes is None:
        return False
    return _estimated_tar_size(records=records) > max_bytes


def _estimated_tar_size(*, records: list[_SerializedShardRecord]) -> int:
    """Predict ``tarfile``'s USTAR output size before publishing a shard."""

    member_bytes = sum(
        tarfile.BLOCKSIZE
        + _round_up(value=len(record.payload), multiple=tarfile.BLOCKSIZE)
        for record in records
    )
    raw_size = member_bytes + (2 * tarfile.BLOCKSIZE)
    return _round_up(value=raw_size, multiple=tarfile.RECORDSIZE)


def _write_tar_shard(
    *,
    path: Path,
    records: tuple[_SerializedShardRecord, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(path)
    try:
        with tarfile.open(
            temporary_path,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        ) as archive:
            for record in records:
                info = tarfile.TarInfo(name=record.member_name)
                info.size = len(record.payload)
                info.mode = 0o644
                info.mtime = 0
                archive.addfile(info, io.BytesIO(record.payload))
        os.replace(temporary_path, path)
    except (OSError, tarfile.TarError) as exc:
        raise SnapshotShardWriteError(
            f"failed to write WebDataset shard: {path}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(
    *,
    path: Path,
    payload: Mapping[str, Any],
    indent: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(path)
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=indent,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _remove_previous_shards(
    *,
    training_directory: Path,
    shards_directory: Path,
) -> None:
    """Remove only the prior generated shard directory under this snapshot."""

    if not shards_directory.exists():
        return
    resolved_training_directory = training_directory.resolve()
    resolved_shards_directory = shards_directory.resolve()
    try:
        resolved_shards_directory.relative_to(resolved_training_directory)
    except ValueError as exc:
        raise SnapshotShardWriteError(
            "configured shard directory escapes the training snapshot: "
            f"{shards_directory}"
        ) from exc
    if not resolved_shards_directory.is_dir():
        raise SnapshotShardWriteError(
            f"configured shard path is not a directory: {shards_directory}"
        )
    shutil.rmtree(resolved_shards_directory)


def _round_up(*, value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _require_relative_output_path(
    value: str,
    *,
    label: str,
) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a relative path string")
    raw_value = value.strip()
    candidate = Path(raw_value)
    windows_candidate = PureWindowsPath(raw_value)
    if (
        not raw_value
        or candidate.is_absolute()
        or windows_candidate.is_absolute()
        or candidate.drive
        or windows_candidate.drive
        or not candidate.parts
        or any(
            part in {"", ".", ".."}
            for path_parts in (candidate.parts, windows_candidate.parts)
            for part in path_parts
        )
    ):
        raise ValueError(f"{label} must be a non-empty relative path")


__all__ = [
    "SnapshotOutputSettings",
    "SnapshotShardWriteError",
    "write_shard_index",
    "write_snapshot_metadata",
    "write_snapshot_rows",
    "write_webdataset_shards",
]
