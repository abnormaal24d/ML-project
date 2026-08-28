"""JSONL dataset indexing and lazy sample loading."""

from __future__ import annotations

import json
import os
import tarfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from filelock import FileLock
from torch.utils.data import Dataset

from config.environment.default_values import (
    DEFAULT_DATASET_MANIFEST_FILENAME,
)
from mmcrawler_datasets.record_components.parsing import parse_record
from mmcrawler_datasets.record_components.validation import (
    assert_object_paths_exist,
    validate_sample,
)
from mmcrawler_datasets.safe_io import (
    MAX_RECORD_BYTES,
    MAX_TAR_MEMBERS,
    load_bounded_json_object,
    read_bounded_text,
    resolve_dataset_reference,
)
from mmcrawler_datasets.schema import DatasetSplit, MultimodalSample
from mmcrawler_datasets.snapshots.rejected_sample_reports import (
    REJECTED_ROWS_FILENAME,
)
from mmcrawler_datasets.snapshots.shards import (
    WEB_DATASET_TAR_FORMAT,
    load_and_validate_shard_index,
    validate_shard_record_member_name,
)
from schemas.multimodal_tasks import canonical_task_name

if TYPE_CHECKING:
    from logger.project_logger import ProjectLogger

_MANAGED_DATASET_ROOT_PREFIXES = (".staging-", ".replaced-")


@dataclass(frozen=True, slots=True)
class RecordRef:
    path: Path
    line_number: int
    offset: int
    sample_id: str
    record_id: str
    task_type: str
    positive_id: str | None
    negative_ids: tuple[str, ...]
    modality_signature: tuple[str, ...]
    member_name: str | None = None


DatasetStats = dict[str, int]


def resolve_split_paths(
    *,
    dataset_root: Path,
    split: DatasetSplit,
) -> tuple[Path, ...]:
    """Resolve canonical JSONL/tar paths for one dataset split."""

    if not isinstance(split, DatasetSplit):
        raise TypeError(f"split must be DatasetSplit, got {type(split)!r}")

    resolved_root = Path(dataset_root).resolve()
    if resolved_root.name.startswith(_MANAGED_DATASET_ROOT_PREFIXES):
        raise FileNotFoundError(
            "managed temporary dataset directory is not trainable: "
            f"{resolved_root}"
        )
    return _manifest_output_paths(
        dataset_root=resolved_root,
        split=split,
    )


def iter_raw_records(
    *, paths: tuple[Path, ...]
) -> Iterator[dict[str, object]]:
    """Yield bounded raw records from validated JSONL or tar paths."""

    for path in paths:
        for _ref, record in _iter_records(path=path):
            yield record


def read_record(*, ref: RecordRef) -> dict[str, object]:
    if ref.member_name is not None:
        with tarfile.open(ref.path, mode="r") as archive:
            line = _read_indexed_tar_record(archive=archive, ref=ref)
        return _load_record_from_line(line=line, ref=ref)

    with ref.path.open("r", encoding="utf-8") as handle:
        handle.seek(ref.offset)
        line = handle.readline().strip()
    return _load_record_from_line(line=line, ref=ref)


def build_dataset_index(
    *,
    dataset_root: Path,
    paths: tuple[Path, ...],
    enabled_tasks: frozenset[str],
    minimums_by_task: dict[str, int],
    rejected_rows_path: Path,
    split_value: str,
    max_samples: int | None,
    drop_samples_with_invalid_targets: bool,
    require_materialized_tensors: bool,
) -> tuple[tuple[RecordRef, ...], DatasetStats]:
    """Parse each record once while building a validated index."""

    refs: list[RecordRef] = []
    record_ids: set[str] = set()
    seen = 0
    rejected = 0
    duplicates = 0

    rejected_rows_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_rows_path.touch(exist_ok=True)

    for path in paths:
        for provisional_ref, record in _iter_records(path=path):
            seen += 1
            try:
                indexed_ref = _index_record(
                    ref=provisional_ref,
                    record=record,
                    dataset_root=dataset_root,
                    enabled_tasks=enabled_tasks,
                    drop_samples_with_invalid_targets=(
                        drop_samples_with_invalid_targets
                    ),
                    require_materialized_tensors=require_materialized_tensors,
                )
            except (OSError, TypeError, ValueError) as exc:
                rejected += 1
                _append_rejection(
                    path=rejected_rows_path,
                    ref=provisional_ref,
                    error=f"invalid_record:{type(exc).__name__}",
                    split_value=split_value,
                )
                continue

            if indexed_ref.record_id in record_ids:
                duplicates += 1
                rejected += 1
                _append_rejection(
                    path=rejected_rows_path,
                    ref=indexed_ref,
                    error="duplicate_record_id",
                    split_value=split_value,
                )
                continue

            record_ids.add(indexed_ref.record_id)
            refs.append(indexed_ref)
            if max_samples is not None and len(refs) >= max_samples:
                break
        if max_samples is not None and len(refs) >= max_samples:
            break

    filtered_refs, minimum_rejected = _apply_task_minimums(
        refs=tuple(refs),
        minimums_by_task=minimums_by_task,
        rejected_rows_path=rejected_rows_path,
        split_value=split_value,
    )
    rejected += minimum_rejected

    return filtered_refs, {
        "total": seen,
        "valid": len(filtered_refs),
        "invalid": rejected,
        "duplicates": duplicates,
    }


class MultimodalJsonlDataset(Dataset[MultimodalSample]):
    """Index one multimodal JSONL split; load samples lazily by offset."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        split: DatasetSplit,
        record_paths: tuple[Path, ...],
        max_samples: int | None,
        logger: ProjectLogger,
        enabled_tasks: frozenset[str] = frozenset(),
        minimums_by_task: dict[str, int] | None = None,
        drop_samples_with_invalid_targets: bool = True,
        require_materialized_tensors: bool = False,
    ) -> None:
        if not isinstance(split, DatasetSplit):
            raise TypeError(f"split must be DatasetSplit, got {type(split)!r}")

        self.dataset_root = Path(dataset_root).resolve()
        self.split = split
        self.records_paths = record_paths
        self.rejected_rows_path = self.dataset_root / REJECTED_ROWS_FILENAME

        self.refs, self.stats = build_dataset_index(
            dataset_root=self.dataset_root,
            paths=self.records_paths,
            enabled_tasks=enabled_tasks,
            minimums_by_task=minimums_by_task or {},
            rejected_rows_path=self.rejected_rows_path,
            split_value=self.split.value,
            max_samples=max_samples,
            drop_samples_with_invalid_targets=(
                drop_samples_with_invalid_targets
            ),
            require_materialized_tensors=require_materialized_tensors,
        )

        logger.info(
            "multimodal_dataset_loaded",
            dataset_root=str(self.dataset_root),
            split=self.split.value,
            records_paths=[str(path) for path in self.records_paths],
            sample_count=len(self.refs),
            stats=self.stats,
            rejected_rows_path=(
                self.rejected_rows_path.as_posix()
                if self.rejected_rows_path.exists()
                else None
            ),
        )

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> MultimodalSample:
        ref = self.refs[index]
        record = read_record(ref=ref)
        sample = parse_record(record=record, dataset_root=self.dataset_root)
        assert_object_paths_exist(
            sample=sample,
            ref_path=ref.path,
            line_number=ref.line_number,
        )
        return sample

    def task_type_at(self, index: int) -> str:
        """Return the validated task identity without reparsing a record."""

        return self.refs[index].task_type

    def modality_signature_at(self, index: int) -> tuple[str, ...]:
        """Return the validated input modalities without reparsing a record."""

        return self.refs[index].modality_signature


def _index_record(
    *,
    ref: RecordRef,
    record: dict[str, object],
    dataset_root: Path,
    enabled_tasks: frozenset[str],
    drop_samples_with_invalid_targets: bool,
    require_materialized_tensors: bool,
) -> RecordRef:
    sample = parse_record(record=record, dataset_root=dataset_root)
    task_type = canonical_task_name(sample.task_type)

    if enabled_tasks and task_type not in enabled_tasks:
        raise ValueError(f"task_type_disabled:{task_type}")

    if not _sample_has_usable_content(sample):
        raise ValueError("sample_has_no_text_or_objects")

    if drop_samples_with_invalid_targets:
        reasons = list(validate_sample(sample=sample))
        if require_materialized_tensors:
            if sample.has_text and sample.text_tokens_path is None:
                reasons.append("missing_materialized_tensor:text_tokens_path")
            if sample.has_image and sample.image_tensor_path is None:
                reasons.append("missing_materialized_tensor:image_tensor_path")
            if sample.has_audio and sample.audio_tensor_path is None:
                reasons.append("missing_materialized_tensor:audio_tensor_path")
            if sample.has_video and sample.video_tensor_path is None:
                reasons.append("missing_materialized_tensor:video_tensor_path")
            output_modalities = frozenset(sample.output_modalities)
            if (
                "audio" in output_modalities
                and sample.target_audio_tokens_path is None
            ):
                reasons.append(
                    "missing_materialized_tensor:target_audio_tokens_path"
                )
            if (
                "image" in output_modalities
                and sample.target_image_tensor_path is None
            ):
                reasons.append(
                    "missing_materialized_tensor:target_image_tensor_path"
                )
            if (
                "video" in output_modalities
                and sample.target_video_tokens_path is None
            ):
                reasons.append(
                    "missing_materialized_tensor:target_video_tokens_path"
                )
            if (
                sample.source_image_path is not None
                and sample.source_image_tensor_path is None
            ):
                reasons.append(
                    "missing_materialized_tensor:source_image_tensor_path"
                )
            if (
                sample.edit_mask_path is not None
                and sample.edit_mask_tensor_path is None
            ):
                reasons.append(
                    "missing_materialized_tensor:edit_mask_tensor_path"
                )
        if reasons:
            raise ValueError(";".join(str(reason) for reason in reasons))

    return RecordRef(
        path=ref.path,
        line_number=ref.line_number,
        offset=ref.offset,
        sample_id=sample.sample_id,
        record_id=sample.record_id,
        task_type=task_type,
        positive_id=sample.positive_id,
        negative_ids=sample.negative_ids,
        modality_signature=_modality_signature_for_sample(sample),
        member_name=ref.member_name,
    )


def _modality_signature_for_sample(
    sample: MultimodalSample,
) -> tuple[str, ...]:
    modalities: list[str] = []
    for name in (
        "text",
        "image",
        "audio",
        "video",
        "document",
        "layout",
        "mask",
        "code",
        "json",
    ):
        if bool(getattr(sample, f"has_{name}", False)):
            modalities.append(name)
    return tuple(modalities)


def _sample_has_usable_content(sample: MultimodalSample) -> bool:
    if sample.has_text or sample.generative_target_text:
        return True
    return any(
        (
            sample.has_image,
            sample.has_audio,
            sample.has_video,
            sample.has_document,
        )
    )


def _apply_task_minimums(
    *,
    refs: tuple[RecordRef, ...],
    minimums_by_task: dict[str, int],
    rejected_rows_path: Path,
    split_value: str,
) -> tuple[tuple[RecordRef, ...], int]:
    if not minimums_by_task:
        return refs, 0

    counts = Counter(canonical_task_name(ref.task_type) for ref in refs)
    undercovered = {
        task_type: {
            "samples": counts.get(task_type, 0),
            "minimum": minimum,
        }
        for task_type, minimum in minimums_by_task.items()
        if minimum > 0 and counts.get(task_type, 0) < minimum
    }
    if not undercovered:
        return refs, 0

    kept: list[RecordRef] = []
    rejected = 0
    for ref in refs:
        task_type = canonical_task_name(ref.task_type)
        if task_type not in undercovered:
            kept.append(ref)
            continue
        gate = undercovered[task_type]
        rejected += 1
        _append_rejection(
            path=rejected_rows_path,
            ref=ref,
            error=(
                "task_under_minimum_samples:"
                f"{task_type}:{gate['samples']}/{gate['minimum']}"
            ),
            split_value=split_value,
        )
    return tuple(kept), rejected


def _append_rejection(
    *,
    path: Path,
    ref: RecordRef,
    error: str,
    split_value: str,
) -> None:
    payload: dict[str, Any] = {
        "split": split_value,
        "records_path": ref.path.as_posix(),
        "line_number": ref.line_number,
        "member_name": ref.member_name,
        "record_id": ref.record_id or None,
        "sample_id": ref.sample_id or None,
        "task_type": ref.task_type or None,
        "reason": error,
        "error": error,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with FileLock(f"{path}.lock"):
        descriptor = os.open(
            path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
        )
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _iter_records(
    *,
    path: Path,
) -> Iterator[tuple[RecordRef, dict[str, object]]]:
    if path.suffix == ".tar":
        with tarfile.open(path, mode="r") as archive:
            member_names: set[str] = set()
            for member_index, member in enumerate(archive, start=1):
                if member_index > MAX_TAR_MEMBERS:
                    raise ValueError(
                        f"tar shard exceeds {MAX_TAR_MEMBERS} members: {path}"
                    )
                if not member.isfile():
                    raise ValueError(
                        f"non-file tar member in shard {path}: {member.name}"
                    )
                validate_shard_record_member_name(member.name)
                if member.name in member_names:
                    raise ValueError(
                        f"duplicate tar member name {member.name!r} in {path}"
                    )
                member_names.add(member.name)
                if member.size > MAX_RECORD_BYTES:
                    raise ValueError(
                        f"tar record exceeds {MAX_RECORD_BYTES} bytes: "
                        f"{member.name}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(
                        f"missing shard member {member.name} in {path}"
                    )
                line = read_bounded_text(
                    handle=extracted,
                    max_bytes=MAX_RECORD_BYTES,
                    label=f"tar record {member.name!r}",
                )
                ref = RecordRef(
                    path=path,
                    line_number=member_index,
                    offset=0,
                    sample_id="",
                    record_id="",
                    task_type="",
                    positive_id=None,
                    negative_ids=(),
                    modality_signature=(),
                    member_name=member.name,
                )
                yield ref, _load_record_from_line(line=line, ref=ref)
        return

    with path.open("r", encoding="utf-8") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            line = handle.readline(MAX_RECORD_BYTES + 1)
            if not line:
                break
            line_number += 1
            if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
                raise ValueError(
                    f"dataset record exceeds {MAX_RECORD_BYTES} bytes in "
                    f"{path} at line {line_number}"
                )
            stripped = line.strip()
            if not stripped:
                continue
            ref = RecordRef(
                path=path,
                line_number=line_number,
                offset=offset,
                sample_id="",
                record_id="",
                task_type="",
                positive_id=None,
                negative_ids=(),
                modality_signature=(),
            )
            yield ref, _load_record_from_line(line=stripped, ref=ref)


def _read_indexed_tar_record(
    *,
    archive: tarfile.TarFile,
    ref: RecordRef,
) -> str:
    """Read the exact tar entry indexed earlier, never a name alias."""

    for member_index, member in enumerate(archive, start=1):
        if member_index > MAX_TAR_MEMBERS:
            break
        if member_index != ref.line_number:
            continue
        if (
            not member.isfile()
            or member.name != ref.member_name
            or member.size > MAX_RECORD_BYTES
        ):
            break
        validate_shard_record_member_name(member.name)
        extracted = archive.extractfile(member)
        if extracted is None:
            break
        return read_bounded_text(
            handle=extracted,
            max_bytes=MAX_RECORD_BYTES,
            label=f"tar record {member.name!r}",
        )
    raise ValueError(
        f"indexed shard member changed or is missing in {ref.path} at "
        f"position {ref.line_number}"
    )


def _load_record_from_line(
    *,
    line: str,
    ref: RecordRef,
) -> dict[str, object]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {ref.path} at line {ref.line_number}"
        ) from exc
    if not isinstance(record, dict):
        raise ValueError(
            f"dataset record must be an object in {ref.path} "
            f"at line {ref.line_number}"
        )
    return record


def _manifest_output_paths(
    *,
    dataset_root: Path,
    split: DatasetSplit,
) -> tuple[Path, ...]:
    manifest_path = dataset_root / DEFAULT_DATASET_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"dataset manifest is required: {manifest_path}"
        )
    payload = load_bounded_json_object(path=manifest_path)
    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, dict):
        raise ValueError("dataset manifest paths must be an object")
    raw_outputs = payload.get("outputs")
    if not isinstance(raw_outputs, dict):
        raise ValueError("dataset manifest outputs must be an object")

    writes_jsonl = raw_outputs.get("jsonl")
    writes_shards = raw_outputs.get("shards")
    if not isinstance(writes_jsonl, bool) or not isinstance(
        writes_shards,
        bool,
    ):
        raise ValueError("dataset manifest output flags must be booleans")
    entries_by_split = None
    if writes_shards:
        if raw_outputs.get("shard_format") != WEB_DATASET_TAR_FORMAT:
            raise ValueError(
                "dataset manifest has an unsupported shard format"
            )
        shard_index = raw_paths.get("shard_index")
        if not isinstance(shard_index, str) or not shard_index.strip():
            raise ValueError("manifest paths.shard_index must be a path")
        index_path = resolve_dataset_reference(
            dataset_root=dataset_root,
            reference=shard_index,
            label="manifest paths.shard_index",
        )
        if not index_path.is_file():
            raise FileNotFoundError(
                "manifest declares shard output but the shard index is "
                f"missing: {index_path}"
            )
        entries_by_split = load_and_validate_shard_index(
            dataset_root=dataset_root,
            index_path=index_path,
        )
    if writes_jsonl:
        raw_splits = raw_paths.get("splits")
        if not isinstance(raw_splits, dict):
            raise ValueError("manifest paths.splits must be an object")
        raw_path = raw_splits.get(split.value)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                f"manifest paths.splits.{split.value} must be a path"
            )
        split_path = resolve_dataset_reference(
            dataset_root=dataset_root,
            reference=raw_path,
            label=f"manifest split {split.value}",
        )
        if not split_path.is_file():
            raise FileNotFoundError(
                f"manifest-declared JSONL split is missing: {split_path}"
            )
        return (split_path,)

    if writes_shards:
        if entries_by_split is None:
            raise ValueError("shard output was declared but not resolved")
        return tuple(entry.path for entry in entries_by_split[split.value])

    raise ValueError("dataset manifest enables no supported output format")
