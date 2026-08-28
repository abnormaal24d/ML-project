"""Shared WebDataset tar-shard and shard-index contract.

The training snapshot writer and the dataset loader use this module rather
than maintaining independent assumptions about shard file names, record
members, checksums, or index payloads.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, TypedDict

from mmcrawler_datasets.safe_io import (
    load_bounded_json_object,
    resolve_dataset_reference,
)
from schemas.versions import TRAINING_DATASET_SCHEMA_VERSION

WEB_DATASET_TAR_FORMAT: Final = "webdataset_tar"
SHARD_INDEX_SCHEMA_VERSION: Final = "webdataset_shard_index_v1"
SHARD_CHECKSUM_ALGORITHM: Final = "sha256"
SHARD_COMPRESSION: Final = "none"
SHARD_FILE_EXTENSION: Final = ".tar"
SHARD_RECORD_EXTENSION: Final = ".json"
SHARD_SPLITS: Final = ("train", "val", "test")

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_SAMPLE_KEY_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ShardIndexEntry(TypedDict):
    """Serialized description of one immutable shard."""

    path: str
    sample_count: int
    byte_size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ResolvedShardIndexEntry:
    """A verified shard-index entry with its dataset-root-confined path."""

    path: Path
    sample_count: int
    byte_size: int
    sha256: str


def shard_filename(*, split: str, shard_number: int) -> str:
    """Return the canonical, deterministic name for one tar shard."""

    _require_known_split(split)
    if shard_number < 0:
        raise ValueError("shard number must not be negative")
    return f"{split}-{shard_number:05d}{SHARD_FILE_EXTENSION}"


def shard_record_member_name(*, split: str, sample_number: int) -> str:
    """Return the canonical WebDataset member name for one JSON sample."""

    _require_known_split(split)
    if sample_number < 0:
        raise ValueError("sample number must not be negative")
    sample_key = f"{split}-{sample_number:012d}"
    return f"{sample_key}{SHARD_RECORD_EXTENSION}"


def build_shard_index_payload(
    *,
    entries_by_split: dict[str, tuple[ShardIndexEntry, ...]],
) -> dict[str, object]:
    """Build the single canonical serialized shard-index payload."""

    unknown_splits = sorted(set(entries_by_split) - set(SHARD_SPLITS))
    if unknown_splits:
        raise ValueError(f"unknown shard-index splits: {unknown_splits}")
    missing_splits = sorted(set(SHARD_SPLITS) - set(entries_by_split))
    if missing_splits:
        raise ValueError(f"shard index lacks splits: {missing_splits}")

    return {
        "schema_version": SHARD_INDEX_SCHEMA_VERSION,
        "format": WEB_DATASET_TAR_FORMAT,
        "compression": SHARD_COMPRESSION,
        "checksum_algorithm": SHARD_CHECKSUM_ALGORITHM,
        "dataset_schema_version": TRAINING_DATASET_SCHEMA_VERSION,
        "record_extension": SHARD_RECORD_EXTENSION,
        "splits": {
            split: [dict(entry) for entry in entries_by_split[split]]
            for split in SHARD_SPLITS
        },
    }


def load_and_validate_shard_index(
    *,
    dataset_root: Path,
    index_path: Path,
) -> dict[str, tuple[ResolvedShardIndexEntry, ...]]:
    """Load and fail closed on an invalid or tampered shard index."""

    payload = load_bounded_json_object(path=index_path)
    return validate_shard_index_payload(
        dataset_root=dataset_root,
        payload=payload,
    )


def validate_shard_index_payload(
    *,
    dataset_root: Path,
    payload: dict[str, object],
) -> dict[str, tuple[ResolvedShardIndexEntry, ...]]:
    """Validate every index entry and its on-disk checksum.

    Returning only verified entries prevents callers from silently skipping a
    bad entry and accidentally training from an incomplete dataset.
    """

    if payload.get("schema_version") != SHARD_INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported shard-index schema version")
    if payload.get("format") != WEB_DATASET_TAR_FORMAT:
        raise ValueError("unsupported shard format")
    if payload.get("compression") != SHARD_COMPRESSION:
        raise ValueError("unsupported shard compression")
    if payload.get("checksum_algorithm") != SHARD_CHECKSUM_ALGORITHM:
        raise ValueError("unsupported shard checksum algorithm")
    if (
        payload.get("dataset_schema_version")
        != TRAINING_DATASET_SCHEMA_VERSION
    ):
        raise ValueError("unsupported shard dataset schema version")
    if payload.get("record_extension") != SHARD_RECORD_EXTENSION:
        raise ValueError("unsupported shard record extension")

    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("shard index splits must be an object")

    unknown_splits = sorted(set(raw_splits) - set(SHARD_SPLITS))
    if unknown_splits:
        raise ValueError(
            f"shard index contains unknown splits: {unknown_splits}"
        )

    resolved: dict[str, tuple[ResolvedShardIndexEntry, ...]] = {}
    all_paths: set[Path] = set()
    for split in SHARD_SPLITS:
        raw_entries = raw_splits.get(split)
        if not isinstance(raw_entries, list):
            raise ValueError(f"shard index has no entry list for {split}")

        entries: list[ResolvedShardIndexEntry] = []
        for position, raw_entry in enumerate(raw_entries):
            entry = _validate_index_entry(
                dataset_root=dataset_root,
                raw_entry=raw_entry,
                split=split,
                position=position,
            )
            if entry.path in all_paths:
                raise ValueError(
                    f"shard index references a shard more than once: "
                    f"{entry.path}"
                )
            all_paths.add(entry.path)
            entries.append(entry)
        resolved[split] = tuple(entries)
    return resolved


def validate_shard_record_member_name(name: str) -> None:
    """Ensure a tar member is the one JSON component in this shard contract."""

    path = PurePosixPath(name)
    if path.name != name or path.suffix != SHARD_RECORD_EXTENSION:
        raise ValueError(f"invalid WebDataset record member name: {name!r}")
    sample_key = path.stem
    if not _SAMPLE_KEY_PATTERN.fullmatch(sample_key):
        raise ValueError(f"invalid WebDataset sample key: {sample_key!r}")


def sha256_file(path: Path) -> str:
    """Return the digest of a shard without loading it wholly into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_index_entry(
    *,
    dataset_root: Path,
    raw_entry: object,
    split: str,
    position: int,
) -> ResolvedShardIndexEntry:
    if not isinstance(raw_entry, dict):
        raise ValueError(
            f"shard index entry {split}[{position}] must be an object"
        )

    raw_path = raw_entry.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"shard index entry {split}[{position}] lacks a path")
    path = resolve_dataset_reference(
        dataset_root=dataset_root,
        reference=raw_path,
        label=f"shard index entry {split}[{position}] path",
    )
    if path.suffix != SHARD_FILE_EXTENSION:
        raise ValueError(
            f"shard index entry {split}[{position}] must reference a tar file"
        )
    if not path.is_file():
        raise ValueError(
            f"indexed shard does not exist for {split}[{position}]: {path}"
        )

    sample_count = _require_non_negative_int(
        raw_entry.get("sample_count"),
        label=f"shard index entry {split}[{position}].sample_count",
    )
    byte_size = _require_non_negative_int(
        raw_entry.get("byte_size"),
        label=f"shard index entry {split}[{position}].byte_size",
    )
    digest = raw_entry.get("sha256")
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(
            f"shard index entry {split}[{position}].sha256 is invalid"
        )
    if path.stat().st_size != byte_size:
        raise ValueError(
            f"indexed shard byte size does not match for {split}[{position}]"
        )
    if sha256_file(path) != digest:
        raise ValueError(
            f"indexed shard checksum does not match for {split}[{position}]"
        )

    return ResolvedShardIndexEntry(
        path=path,
        sample_count=sample_count,
        byte_size=byte_size,
        sha256=digest,
    )


def _require_known_split(split: str) -> None:
    if split not in SHARD_SPLITS:
        raise ValueError(f"unknown shard split: {split!r}")


def _require_non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


__all__ = [
    "SHARD_CHECKSUM_ALGORITHM",
    "SHARD_COMPRESSION",
    "SHARD_FILE_EXTENSION",
    "SHARD_INDEX_SCHEMA_VERSION",
    "SHARD_RECORD_EXTENSION",
    "SHARD_SPLITS",
    "ShardIndexEntry",
    "ResolvedShardIndexEntry",
    "WEB_DATASET_TAR_FORMAT",
    "build_shard_index_payload",
    "load_and_validate_shard_index",
    "sha256_file",
    "shard_filename",
    "shard_record_member_name",
    "validate_shard_index_payload",
    "validate_shard_record_member_name",
]
