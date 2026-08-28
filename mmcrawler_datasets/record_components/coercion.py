"""Primitive coercion and safe dataset path resolution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mmcrawler_datasets.safe_io import resolve_dataset_reference


def optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def optional_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def optional_table(value: object) -> str | dict[str, object] | None:
    if isinstance(value, dict):
        return dict(value)
    return optional_string(value)


def str_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = optional_string(value)
        return (text,) if text is not None else ()
    if not isinstance(value, (list, tuple)):
        return ()
    values: list[str] = []
    for item in value:
        text = optional_string(item)
        if text is not None:
            values.append(text)
    return tuple(values)


def require_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def require_mapping(
    mapping: Mapping[str, object],
    key: str,
) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return dict(value)


def optional_path(*, dataset_root: Path, value: object) -> Path | None:
    text = optional_string(value)
    if text is None:
        return None
    return resolve_path(dataset_root=dataset_root, raw_path=text)


def resolve_path(*, dataset_root: Path, raw_path: str) -> Path:
    return resolve_dataset_reference(
        dataset_root=dataset_root,
        reference=raw_path,
        label="dataset record path",
        allow_absolute=True,
    )
