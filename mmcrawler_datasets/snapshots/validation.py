"""Structural validation for persisted training snapshots."""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING

from mmcrawler_datasets.safe_io import (
    MAX_RECORD_BYTES,
    MAX_TAR_MEMBERS,
    iter_bounded_jsonl,
    read_bounded_text,
    resolve_dataset_reference,
)
from mmcrawler_datasets.snapshots.rejected_sample_reports import (
    REQUIRED_REJECTED_SAMPLE_REPORTS,
)
from mmcrawler_datasets.snapshots.shards import (
    WEB_DATASET_TAR_FORMAT,
    load_and_validate_shard_index,
    validate_shard_record_member_name,
)
from schemas.versions import is_supported_training_schema_version

if TYPE_CHECKING:
    from config.settings.datasets import DatasetPathSettings


@dataclass(frozen=True, slots=True)
class TrainingDatasetValidationError(Exception):
    """Raised when a training snapshot fails post-build validation."""

    snapshot_id: str
    validation_errors: tuple[str, ...]
    validation_payload: dict[str, object]
    validation_report_path: Path
    training_directory: Path
    validation_remediation: tuple[str, ...] = ()

    def __str__(self) -> str:
        joined = "; ".join(self.validation_errors)
        return joined or "training snapshot validation failed"


def validate_snapshot(
    *,
    training_directory: Path,
    dataset_paths: DatasetPathSettings,
    write_jsonl: bool,
    write_shards: bool,
    shard_index_filename: str,
) -> bool:
    """Validate every requested output before a snapshot is published.

    Shards are validated through their index, actual tar members, sample
    counts, sizes, and checksums. A requested shard output never degrades to
    JSONL-only validation.
    """

    if not training_directory.is_dir() or not write_jsonl and not write_shards:
        return False
    manifest = _read_object(
        training_directory / dataset_paths.dataset_manifest_filename
    )
    stats = _read_object(training_directory / dataset_paths.stats_filename)
    if manifest is None or stats is None:
        return False
    if not is_supported_training_schema_version(
        str(manifest.get("schema_version") or "")
    ):
        return False
    if not is_supported_training_schema_version(
        str(stats.get("schema_version") or "")
    ):
        return False
    if not _validate_declared_outputs(
        manifest=manifest,
        write_jsonl=write_jsonl,
        write_shards=write_shards,
        shard_index_filename=shard_index_filename,
    ):
        return False

    jsonl_rows_by_split = (
        _read_jsonl_splits(
            training_directory=training_directory,
            manifest=manifest,
        )
        if write_jsonl
        else None
    )
    if write_jsonl and jsonl_rows_by_split is None:
        return False

    shard_rows_by_split = (
        _read_shard_splits(
            training_directory=training_directory,
            shard_index_filename=shard_index_filename,
        )
        if write_shards
        else None
    )
    if write_shards and shard_rows_by_split is None:
        return False

    if (
        jsonl_rows_by_split is not None
        and shard_rows_by_split is not None
        and not _same_rows_by_split(
            left=jsonl_rows_by_split,
            right=shard_rows_by_split,
        )
    ):
        return False

    rows_by_split = (
        jsonl_rows_by_split
        if jsonl_rows_by_split is not None
        else shard_rows_by_split
    )
    if rows_by_split is None:
        return False
    if any(
        not (training_directory / filename).is_file()
        for filename in REQUIRED_REJECTED_SAMPLE_REPORTS
    ):
        return False
    sample_ids: set[str] = set()
    observed_counts: dict[str, int] = {}
    for split, rows in rows_by_split.items():
        observed_counts[split] = len(rows)
        for row in rows:
            sample_id = str(row.get("sample_id") or "").strip()
            if not sample_id or sample_id in sample_ids:
                return False
            if str(row.get("split") or "") != split:
                return False
            sample_ids.add(sample_id)
    manifest_counts = manifest.get("splits")
    if not isinstance(manifest_counts, dict):
        return False
    return all(
        _exact_int(manifest_counts.get(name)) == count
        for name, count in observed_counts.items()
    )


def _validate_declared_outputs(
    *,
    manifest: dict[str, object],
    write_jsonl: bool,
    write_shards: bool,
    shard_index_filename: str,
) -> bool:
    """Bind requested runtime outputs to the completed snapshot manifest."""

    paths = manifest.get("paths")
    outputs = manifest.get("outputs")
    if not isinstance(paths, dict) or not isinstance(outputs, dict):
        return False

    if outputs.get("jsonl") is not write_jsonl:
        return False
    if outputs.get("shards") is not write_shards:
        return False
    if write_jsonl:
        split_paths = paths.get("splits")
        if not isinstance(split_paths, dict):
            return False
        if any(
            not isinstance(split_paths.get(split), str)
            or not split_paths[split].strip()
            for split in ("train", "val", "test")
        ):
            return False
    if write_shards:
        if paths.get("shard_index") != shard_index_filename:
            return False
        if outputs.get("shard_format") != WEB_DATASET_TAR_FORMAT:
            return False
    elif outputs.get("shard_format") is not None:
        return False
    return True


def _read_jsonl_splits(
    *,
    training_directory: Path,
    manifest: dict[str, object],
) -> dict[str, tuple[dict[str, object], ...]] | None:
    raw_paths = manifest.get("paths")
    if not isinstance(raw_paths, dict):
        return None
    raw_split_paths = raw_paths.get("splits")
    if not isinstance(raw_split_paths, dict):
        return None
    try:
        split_paths = {
            split: resolve_dataset_reference(
                dataset_root=training_directory,
                reference=_required_path(
                    raw_split_paths.get(split),
                    label=f"manifest paths.splits.{split}",
                ),
                label=f"manifest paths.splits.{split}",
            )
            for split in ("train", "val", "test")
        }
    except (TypeError, ValueError):
        return None
    rows_by_split = {
        name: _read_jsonl(path) for name, path in split_paths.items()
    }
    if any(rows is None for rows in rows_by_split.values()):
        return None
    return {
        split: rows
        for split, rows in rows_by_split.items()
        if rows is not None
    }


def _required_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    return value


def _read_shard_splits(
    *,
    training_directory: Path,
    shard_index_filename: str,
) -> dict[str, tuple[dict[str, object], ...]] | None:
    try:
        index_path = resolve_dataset_reference(
            dataset_root=training_directory,
            reference=shard_index_filename,
            label="configured shard index",
        )
        if not index_path.is_file():
            return None
        entries_by_split = load_and_validate_shard_index(
            dataset_root=training_directory,
            index_path=index_path,
        )
        rows_by_split: dict[str, tuple[dict[str, object], ...]] = {}
        for split, entries in entries_by_split.items():
            rows: list[dict[str, object]] = []
            for entry in entries:
                shard_rows = _read_tar_rows(path=entry.path)
                if len(shard_rows) != entry.sample_count:
                    return None
                rows.extend(shard_rows)
            rows_by_split[split] = tuple(rows)
        return rows_by_split
    except (OSError, TypeError, ValueError, tarfile.TarError):
        return None


def _read_tar_rows(*, path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    member_names: set[str] = set()
    with tarfile.open(path, mode="r") as archive:
        for member_number, member in enumerate(archive, start=1):
            if member_number > MAX_TAR_MEMBERS:
                raise ValueError(f"tar shard exceeds member limit: {path}")
            if not member.isfile():
                raise ValueError(
                    f"tar shard contains a non-file member: {member.name}"
                )
            validate_shard_record_member_name(member.name)
            if member.name in member_names:
                raise ValueError(
                    f"tar shard contains duplicate member: {member.name}"
                )
            member_names.add(member.name)
            if member.size > MAX_RECORD_BYTES:
                raise ValueError(
                    f"tar record exceeds byte limit: {member.name}"
                )
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"tar record cannot be read: {member.name}")
            line = read_bounded_text(
                handle=handle,
                max_bytes=MAX_RECORD_BYTES,
                label=f"tar record {member.name!r}",
            )
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"tar record must contain a JSON object: {member.name}"
                )
            rows.append({str(key): value for key, value in payload.items()})
    return tuple(rows)


def _same_rows_by_split(
    *,
    left: dict[str, tuple[dict[str, object], ...]],
    right: dict[str, tuple[dict[str, object], ...]],
) -> bool:
    """Require dual JSONL/shard outputs to contain exactly the same records."""

    if left.keys() != right.keys():
        return False
    for split in left:
        left_rows = left[split]
        right_rows = right[split]
        if len(left_rows) != len(right_rows):
            return False
        for left_row, right_row in zip(left_rows, right_rows, strict=True):
            if left_row != right_row:
                return False
    return True


def _read_object(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return {str(key): value for key, value in payload.items()}


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...] | None:
    if not path.is_file():
        return None
    rows: list[dict[str, object]] = []
    try:
        for payload in iter_bounded_jsonl(
            path=path, max_bytes=MAX_RECORD_BYTES
        ):
            rows.append({str(key): value for key, value in payload.items()})
    except (ValueError, OSError):
        return None
    return tuple(rows)


def _exact_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


__all__ = ["TrainingDatasetValidationError", "validate_snapshot"]
