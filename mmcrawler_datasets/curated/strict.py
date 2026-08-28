"""Strict validation machinery for persisted curated contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)

NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
Sha256Text = Annotated[
    str,
    StringConstraints(pattern=r"^[a-f0-9]{64}$"),
]
Score = Annotated[
    float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
]
NonNegativeFloat = Annotated[
    float,
    Field(strict=True, ge=0.0, allow_inf_nan=False),
]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class StrictContractModel(BaseModel):
    """Immutable, non-coercing JSON contract model."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    def to_dict(self) -> dict[str, object]:
        """Serialize the exact canonical JSON wire representation."""

        payload = self.model_dump(mode="json")
        return {str(key): value for key, value in payload.items()}


def relative_path(value: object) -> str:
    """Validate one persisted project-relative path (exact canonical form required)."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("curated path must be non-empty")
    if value != value.strip():
        raise ValueError(
            "curated path must not have leading or trailing whitespace"
        )
    candidate = PurePosixPath(value)
    windows_candidate = PureWindowsPath(value)
    if (
        candidate.is_absolute()
        or windows_candidate.is_absolute()
        or bool(windows_candidate.drive)
        or ".." in candidate.parts
        or ".." in windows_candidate.parts
        or "\\" in value
        or "." in candidate.parts
    ):
        raise ValueError("curated path must be project-relative")
    if "//" in value:
        raise ValueError("curated path must not contain duplicate separators")
    if value.endswith("/"):
        raise ValueError("curated path must not have trailing slash")
    canonical = candidate.as_posix()
    if value != canonical:
        raise ValueError("curated path must be in canonical POSIX form")
    return value


def require_exact_dataclass_fields(
    row: Mapping[str, object],
    *,
    record_type: type[Any],
    label: str,
) -> None:
    """Reject missing and unknown persisted dataclass fields."""

    expected = {item.name for item in fields(record_type)}
    observed = set(row)
    unknown = sorted(observed.difference(expected))
    missing = sorted(expected.difference(observed))
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")


def require_text_value(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def optional_text_value(
    value: object,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return require_text_value(value, field_name=field_name)


def require_int_value(
    value: object,
    *,
    field_name: str,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return value


def optional_int_value(
    value: object,
    *,
    field_name: str,
    minimum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return require_int_value(
        value,
        field_name=field_name,
        minimum=minimum,
    )


def require_float_value(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    candidate = float(value)
    if not math.isfinite(candidate):
        raise ValueError(f"{field_name} must be finite")
    return candidate


def optional_float_value(
    value: object,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return require_float_value(value, field_name=field_name)


def require_bool_value(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def optional_bool_value(
    value: object,
    *,
    field_name: str,
) -> bool | None:
    if value is None:
        return None
    return require_bool_value(value, field_name=field_name)


def string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(
            require_text_value(
                item,
                field_name=f"{field_name}[{index}]",
            )
        )
    return tuple(result)


def mapping_tuple(
    value: object,
    *,
    field_name: str,
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an array")
    result: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{index}] must be an object")
        if not all(isinstance(key, str) for key in item):
            raise ValueError(f"{field_name}[{index}] keys must be strings")
        result.append(dict(item))
    return tuple(result)


__all__ = [
    "FiniteFloat",
    "NonEmptyText",
    "NonNegativeFloat",
    "NonNegativeInt",
    "PositiveInt",
    "Score",
    "Sha256Text",
    "StrictContractModel",
    "mapping_tuple",
    "optional_bool_value",
    "optional_float_value",
    "optional_int_value",
    "optional_text_value",
    "relative_path",
    "require_bool_value",
    "require_exact_dataclass_fields",
    "require_float_value",
    "require_int_value",
    "require_text_value",
    "string_tuple",
]
