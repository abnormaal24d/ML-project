"""Shared augmentation settings validation utilities."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast

from config.path_resolution.project_paths import validate_safe_relative_path

AugmentationModality: TypeAlias = Literal[
    "text",
    "document",
    "image",
    "audio",
    "video",
]
AUGMENTATION_MODALITIES = frozenset(
    {"text", "document", "image", "audio", "video"}
)


def validate_output_directory(value: str) -> None:
    """Require a non-empty relative path without parent traversal."""

    try:
        validate_safe_relative_path(value, field_name="output directory")
    except ValueError as exc:
        raise ValueError(
            "output directory must be a non-empty relative path "
            "without parent traversal"
        ) from exc


def validate_operation_names(
    *,
    operations: tuple[str, ...],
    allowed: frozenset[str],
    media_type: str,
) -> None:
    """Reject unknown, blank, or duplicate operation names."""

    normalized = tuple(operation.strip() for operation in operations)

    if any(not operation for operation in normalized):
        raise ValueError(
            f"{media_type} augmentation operations cannot contain blank values"
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"{media_type} augmentation operations cannot contain duplicates"
        )

    unknown = sorted(set(normalized) - allowed)

    if unknown:
        raise ValueError(
            f"unknown {media_type} augmentation operations: "
            f"{', '.join(unknown)}"
        )


def normalize_nonempty_strings(
    *,
    values: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    """Normalize a tuple of required, unique string values."""

    normalized = tuple(value.strip() for value in values)

    if any(not value for value in normalized):
        raise ValueError(f"{field_name} cannot contain blank values")

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} cannot contain duplicate values")

    return normalized


def normalize_mime_types(values: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize, validate, and de-duplicate MIME-type values."""

    normalized: list[str] = []
    encountered: set[str] = set()

    for value in values:
        mime_type = value.strip().lower()

        if not mime_type:
            raise ValueError(
                "allowed_output_mime_types cannot contain blank values"
            )

        if "/" not in mime_type:
            raise ValueError(f"invalid output MIME type: {value!r}")

        major_type, subtype = mime_type.split("/", maxsplit=1)

        if not major_type or not subtype:
            raise ValueError(f"invalid output MIME type: {value!r}")

        if mime_type in encountered:
            continue

        encountered.add(mime_type)
        normalized.append(mime_type)

    return tuple(normalized)


def normalize_modalities(
    modalities: tuple[str, ...],
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[AugmentationModality, ...]:
    """Normalize and validate augmentation modality names."""

    normalized = tuple(
        modality.strip().lower() for modality in modalities if modality.strip()
    )

    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} must contain at least one modality")

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} cannot contain duplicate modalities")

    unknown = sorted(set(normalized) - AUGMENTATION_MODALITIES)

    if unknown:
        raise ValueError(
            f"unknown modalities in {field_name}: {', '.join(unknown)}"
        )

    return cast(tuple[AugmentationModality, ...], normalized)
