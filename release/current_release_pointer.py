"""Current release pointer resolution and validation."""

from __future__ import annotations

from pathlib import Path

from release.release_directory_validation import _validate_release_directory
from release.release_manifest import (
    _reproducibility_requirements_from_manifest,
)
from release.release_utilities import (
    CURRENT_POINTER,
    POINTER_SCHEMA,
    RELEASES_DIRECTORY,
    ProductionPromotionValidationError,
    contained_relative_path,
    read_json_object,
    required_sha256,
    required_string,
    sha256,
)


def resolve_current_release(production_directory: Path) -> Path:
    """Resolve and validate the currently published production release."""

    production_directory = production_directory.resolve(strict=True)
    pointer_path = production_directory / CURRENT_POINTER
    pointer = read_json_object(pointer_path)
    if set(pointer) != {
        "schema_version",
        "release_id",
        "release_directory",
        "release_manifest",
        "release_manifest_sha256",
    }:
        raise ProductionPromotionValidationError(
            "production release pointer fields are invalid"
        )
    if pointer.get("schema_version") != POINTER_SCHEMA:
        raise ProductionPromotionValidationError(
            "production release pointer schema is invalid"
        )

    release_id = required_string(pointer, "release_id")
    release_directory = contained_relative_path(
        root=production_directory,
        relative=required_string(pointer, "release_directory"),
    )
    expected_parent = (production_directory / RELEASES_DIRECTORY).resolve(
        strict=True
    )
    if release_directory.parent != expected_parent:
        raise ProductionPromotionValidationError(
            "production release pointer escapes the releases directory"
        )
    if release_directory.name != release_id or not release_directory.is_dir():
        raise ProductionPromotionValidationError(
            "production release pointer target is unavailable"
        )

    manifest_path = contained_relative_path(
        root=release_directory,
        relative=required_string(pointer, "release_manifest"),
    )
    expected_manifest_sha256 = required_sha256(
        pointer,
        "release_manifest_sha256",
    )
    if not manifest_path.is_file():
        raise ProductionPromotionValidationError(
            "production release manifest is unavailable"
        )
    if sha256(manifest_path) != expected_manifest_sha256:
        raise ProductionPromotionValidationError(
            "production release manifest digest mismatch"
        )

    manifest = read_json_object(manifest_path)
    reproducibility_requirements = _reproducibility_requirements_from_manifest(
        manifest
    )
    _validate_release_directory(
        release_directory=release_directory,
        expected_release_id=release_id,
        reproducibility_requirements=reproducibility_requirements,
    )
    return release_directory


__all__ = ["resolve_current_release"]
