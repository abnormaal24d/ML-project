"""Shared serialized vocabulary for multimodal tasks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Final, Literal, TypeVar

TaskModality = Literal[
    "text",
    "image",
    "audio",
    "video",
    "document",
    "structured",
    "layout",
    "mask",
]

OutputModality = Literal[
    "text",
    "class",
    "json",
    "image",
    "audio",
    "video",
    "embedding",
    "code",
]

TASK_NAME_PATTERN: Final = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

_TaskMapValue = TypeVar("_TaskMapValue")


def canonical_task_name(value: object) -> str:
    """Return one non-empty syntactically canonical task name."""

    raw_value = str(value).strip()
    if not raw_value:
        raise ValueError("task name must not be empty")

    task_name = raw_value.lower().replace("-", "_")
    if not TASK_NAME_PATTERN.fullmatch(task_name):
        raise ValueError(f"task name is not canonical: {value!r}")

    return task_name


def canonical_task_names(
    values: Iterable[object],
    *,
    field_name: str,
) -> set[str]:
    """Return unique task names or raise with the configuration field."""

    names: set[str] = set()
    for value in values:
        name = _required_task_name(value, field_name=field_name)
        if name in names:
            raise ValueError(
                f"{field_name} contains duplicate canonical task {name!r}"
            )
        names.add(name)
    return names


def canonical_task_mapping(
    values: Mapping[str, _TaskMapValue],
    *,
    field_name: str,
) -> dict[str, _TaskMapValue]:
    """Return a task-keyed mapping with canonical, unique keys."""

    normalized: dict[str, _TaskMapValue] = {}
    for raw_name, value in values.items():
        name = _required_task_name(raw_name, field_name=field_name)
        if name in normalized:
            raise ValueError(
                f"{field_name} contains duplicate canonical task {name!r}"
            )
        normalized[name] = value
    return normalized


def normalized_modalities(
    values: Iterable[object],
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Return non-empty, unique modality names in canonical case."""

    modalities: list[str] = []
    encountered: set[str] = set()
    for value in values:
        modality = str(value).strip().lower()
        if not modality:
            raise ValueError(f"{field_name} contains an empty modality")
        if modality in encountered:
            raise ValueError(
                f"{field_name} contains duplicate modality {modality!r}"
            )
        encountered.add(modality)
        modalities.append(modality)
    return tuple(modalities)


def _required_task_name(value: object, *, field_name: str) -> str:
    try:
        return canonical_task_name(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} contains an invalid task name: {value!r}"
        ) from exc
