"""Read strict dataset-manifest evidence for release validation."""

from __future__ import annotations

import json
from pathlib import Path

from config.environment.default_values import (
    DEFAULT_DATASET_MANIFEST_FILENAME,
    DEFAULT_DATASET_SPLIT_NAMES,
    DEFAULT_TEST_SPLIT_NAME,
    DEFAULT_TRAIN_SPLIT_NAME,
    DEFAULT_VAL_SPLIT_NAME,
)
from schemas.multimodal_tasks import canonical_task_name
from schemas.versions import (
    is_supported_training_schema_version,
    training_schema_error,
)


class DatasetManifestError(RuntimeError):
    """Raised when a dataset manifest is missing, malformed, or incompatible."""


def read_dataset_counts(*, dataset_root: Path) -> dict[str, object]:
    """Read canonical dataset counts and validation evidence, fail-closed."""

    manifest_path = dataset_root / DEFAULT_DATASET_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise DatasetManifestError(
            f"required dataset manifest missing: {manifest_path}"
        )

    payload = _read_json_object(path=manifest_path)
    schema_version = _required_text(payload, "schema_version")
    if not is_supported_training_schema_version(schema_version):
        raise DatasetManifestError(training_schema_error(schema_version))

    splits = _required_split_counts(payload.get("splits"))
    total = _required_nonnegative_int(payload, "samples_total")
    if total != sum(splits.values()):
        raise DatasetManifestError(
            "samples_total must equal the sum of canonical split counts"
        )

    validation_valid = payload.get("valid")
    if not isinstance(validation_valid, bool):
        raise DatasetManifestError("manifest field 'valid' must be boolean")

    tasks = _required_task_counts(payload.get("tasks"), field="tasks")
    tasks_by_split = _required_task_counts_by_split(
        payload.get("tasks_by_split")
    )
    modalities = _required_count_mapping(
        payload.get("modalities"), field="modalities"
    )
    validation_errors = _error_messages(payload.get("errors"))

    return {
        "total": total,
        "splits": splits,
        "tasks": tasks,
        "tasks_by_split": tasks_by_split,
        "modalities": modalities,
        "validation_valid": validation_valid,
        "validation_errors": validation_errors,
    }


def _required_split_counts(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise DatasetManifestError("manifest field 'splits' must be an object")
    splits: dict[str, int] = {}
    for split_name in DEFAULT_DATASET_SPLIT_NAMES:
        splits[split_name] = _exact_nonnegative_int(
            raw.get(split_name), field=f"splits.{split_name}"
        )
    unexpected = set(raw).difference(DEFAULT_DATASET_SPLIT_NAMES)
    if unexpected:
        raise DatasetManifestError(
            f"unexpected split names: {sorted(str(item) for item in unexpected)}"
        )
    return splits


def _required_task_counts(raw: object, *, field: str) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise DatasetManifestError(
            f"manifest field {field!r} must be an object"
        )
    counts: dict[str, int] = {}
    for key, value in raw.items():
        task_type = canonical_task_name(key)
        count = _exact_nonnegative_int(value, field=f"{field}.{key}")
        if count > 0:
            counts[task_type] = counts.get(task_type, 0) + count
    return counts


def _required_count_mapping(raw: object, *, field: str) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise DatasetManifestError(
            f"manifest field {field!r} must be an object"
        )
    counts: dict[str, int] = {}
    for key, value in raw.items():
        name = str(key).strip().lower()
        if not name:
            raise DatasetManifestError(f"{field} contains an empty key")
        count = _exact_nonnegative_int(value, field=f"{field}.{key}")
        if count > 0:
            counts[name] = counts.get(name, 0) + count
    return counts


def _required_task_counts_by_split(raw: object) -> dict[str, dict[str, int]]:
    if not isinstance(raw, dict):
        raise DatasetManifestError(
            "manifest field 'tasks_by_split' must be an object"
        )
    unexpected = set(raw).difference(DEFAULT_DATASET_SPLIT_NAMES)
    if unexpected:
        raise DatasetManifestError(
            "tasks_by_split contains unexpected splits: "
            f"{sorted(str(item) for item in unexpected)}"
        )
    return {
        split_name: _required_task_counts(
            raw.get(split_name, {}), field=f"tasks_by_split.{split_name}"
        )
        for split_name in DEFAULT_DATASET_SPLIT_NAMES
    }


def release_counts(
    payload: dict[str, object],
) -> tuple[int, int, int, int, dict[str, int], dict[str, dict[str, int]]]:
    splits = _required_split_counts(payload.get("splits"))
    return (
        _exact_nonnegative_int(payload.get("total"), field="total"),
        splits[DEFAULT_TRAIN_SPLIT_NAME],
        splits[DEFAULT_VAL_SPLIT_NAME],
        splits[DEFAULT_TEST_SPLIT_NAME],
        _required_task_counts(payload.get("tasks"), field="tasks"),
        _required_task_counts_by_split(payload.get("tasks_by_split")),
    )


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise DatasetManifestError(f"manifest field {key!r} is required")
    return text


def _nonnegative_int(value: object) -> int:
    """Coerce non-manifest runtime counters used by acceptance checks."""

    if value is None or isinstance(value, bool):
        return 0
    try:
        return max(0, int(str(value)))
    except (TypeError, ValueError):
        return 0


def _required_nonnegative_int(payload: dict[str, object], key: str) -> int:
    return _exact_nonnegative_int(payload.get(key), field=key)


def _exact_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetManifestError(
            f"manifest field {field!r} must be a non-negative integer"
        )
    return value


def _error_messages(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(
        not isinstance(item, str) for item in raw
    ):
        raise DatasetManifestError(
            "manifest field 'errors' must be a string list"
        )
    return tuple(dict.fromkeys(raw))


def _read_json_object(*, path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetManifestError(
            f"dataset manifest is unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetManifestError("dataset manifest root must be an object")
    return {str(key): value for key, value in payload.items()}
