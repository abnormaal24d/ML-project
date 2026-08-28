"""Canonical project path definitions and resolution."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

DATA_ROOT = "data"
RUNTIME_ROOT = "runtime"
LOGS_ROOT = f"{RUNTIME_ROOT}/logs"
CACHE_ROOT = f"{RUNTIME_ROOT}/cache"
ARTIFACTS_ROOT = "artifacts"
SNAPSHOTS_ROOT = f"{DATA_ROOT}/interim"
REGISTRY_ROOT = f"{DATA_ROOT}/registry"
CONFIG_FILES_ROOT = "config/files"
WORKFLOW_ARTIFACTS_ROOT = f"{REGISTRY_ROOT}/workflow_artifacts"
RAW_RUNS_ROOT = f"{DATA_ROOT}/raw/runs"
CURATED_ROOT = f"{DATA_ROOT}/curated"
TRAINING_SETS_ROOT = f"{DATA_ROOT}/interim/training_sets"
AUGMENTED_TRAINING_SETS_ROOT = f"{DATA_ROOT}/interim/augmented_training_sets"
TRAINING_CHECKPOINTS_ROOT = f"{RUNTIME_ROOT}/training/checkpoints"
QUARANTINE_ROOT = f"{DATA_ROOT}/quarantine"

_IS_WINDOWS = sys.platform == "win32"
_WINDOWS_EXTENDED_PATH_PREFIX = "\\\\?\\"


def normalize_project_path(value: str | Path) -> Path:
    """Resolve one project-owned path in canonical Windows long-path form."""

    path = Path(value).expanduser()
    if not _IS_WINDOWS:
        return path.resolve()

    if not path.is_absolute():
        path = Path.cwd() / path

    text = str(path)
    if text.startswith(_WINDOWS_EXTENDED_PATH_PREFIX):
        return path.resolve()
    if text.startswith("\\\\"):
        path = Path(_WINDOWS_EXTENDED_PATH_PREFIX + "UNC\\" + text[2:])
    else:
        path = Path(_WINDOWS_EXTENDED_PATH_PREFIX + text)
    return path.resolve()


def validate_safe_relative_path(
    value: str | Path,
    *,
    field_name: str = "path",
    single_segment: bool = False,
) -> str:
    """Validate and normalize a relative path for safety.

    Args:
        value: The path value to validate.
        field_name: Name of the field for error messages.
        single_segment: If True, the path must be a single segment (no slashes).

    Returns:
        The normalized path string.

    Raises:
        ValueError: If the path is invalid, absolute, contains traversal,
            or (if single_segment) contains path separators.
    """
    text = str(value).strip()

    if not text:
        raise ValueError(f"{field_name} must be a non-empty relative path")

    if "\x00" in text:
        raise ValueError(f"{field_name} contains a NUL byte")

    normalized = text.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(text)

    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {".", ".."} for part in normalized.split("/"))
    ):
        raise ValueError(
            f"{field_name} must be a relative path without traversal: "
            f"{value!r}"
        )

    if single_segment and "/" in normalized:
        raise ValueError(
            f"{field_name} must be a single path segment: {value!r}"
        )

    return text


def join_contained_path(
    base: str | Path,
    relative_path: str | Path,
    *,
    field_name: str = "path",
    single_segment: bool = False,
) -> Path:
    """Join a base path with a validated relative path, ensuring containment.

    Args:
        base: The base directory path.
        relative_path: The relative path to join.
        field_name: Name of the field for error messages.
        single_segment: If True, the relative path must be a single segment.

    Returns:
        The resolved Path joined from base and relative_path.

    Raises:
        ValueError: If the relative path is invalid or the result escapes base.
    """
    safe_relative = validate_safe_relative_path(
        relative_path,
        field_name=field_name,
        single_segment=single_segment,
    )
    base_path = normalize_project_path(base)
    candidate = (base_path / safe_relative).resolve()

    try:
        candidate.relative_to(base_path)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} escapes the base directory: {relative_path!r}"
        ) from exc

    return candidate


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolve and contain paths within one canonical project root."""

    project_root: Path

    def __post_init__(self) -> None:
        """Normalize the project root once during construction."""

        object.__setattr__(
            self,
            "project_root",
            normalize_project_path(self.project_root),
        )

    @property
    def data(self) -> Path:
        return self.resolve(DATA_ROOT)

    @property
    def runtime(self) -> Path:
        return self.resolve(RUNTIME_ROOT)

    @property
    def logs(self) -> Path:
        return self.resolve(LOGS_ROOT)

    @property
    def cache(self) -> Path:
        return self.resolve(CACHE_ROOT)

    @property
    def artifacts(self) -> Path:
        return self.resolve(ARTIFACTS_ROOT)

    @property
    def snapshots(self) -> Path:
        return self.resolve(SNAPSHOTS_ROOT)

    @property
    def registry(self) -> Path:
        return self.resolve(REGISTRY_ROOT)

    @property
    def config_files(self) -> Path:
        return self.resolve(CONFIG_FILES_ROOT)

    @property
    def workflow_artifacts(self) -> Path:
        return self.resolve(WORKFLOW_ARTIFACTS_ROOT)

    def resolve(
        self,
        path: str | Path,
        *,
        allow_absolute: bool = False,
    ) -> Path:
        """Resolve a path while keeping it inside the project root."""

        text = str(path).strip()
        candidate = Path(path).expanduser()
        posix_path = PurePosixPath(text.replace("\\", "/"))
        windows_path = PureWindowsPath(text)

        absolute_input = (
            candidate.is_absolute()
            or bool(posix_path.root)
            or bool(windows_path.root)
            or bool(windows_path.drive)
        )

        if absolute_input:
            if not allow_absolute:
                raise ValueError(f"Absolute paths are not allowed: {path!r}")

            if not candidate.is_absolute():
                raise ValueError(
                    "Absolute path syntax is not supported on the "
                    f"current platform: {path!r}"
                )

            resolved = normalize_project_path(candidate)
        else:
            resolved = (self.project_root / candidate).resolve()

        try:
            resolved.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(
                f"Path escapes the project root: {path!r} -> {resolved}"
            ) from exc

        return resolved

    def resolve_optional(
        self,
        path: str | Path | None,
        *,
        allow_absolute: bool = False,
    ) -> Path | None:
        """Resolve an optional path while preserving None."""

        if path is None:
            return None

        return self.resolve(
            path,
            allow_absolute=allow_absolute,
        )

    def resolve_workflow_path(
        self,
        *additional: str | Path,
        workflow_id: str | None = None,
        generation_id: str | None = None,
        run_id: str | None = None,
    ) -> Path:
        """Resolve a contained path from safe workflow segments."""

        relative_path = Path()

        for field_name, value in (
            ("workflow_id", workflow_id),
            ("generation_id", generation_id),
            ("run_id", run_id),
        ):
            if value is None:
                continue

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

            posix_segment = PurePosixPath(value.replace("\\", "/"))
            windows_segment = PureWindowsPath(value)

            if (
                posix_segment.is_absolute()
                or windows_segment.is_absolute()
                or bool(windows_segment.drive)
                or len(posix_segment.parts) != 1
                or len(windows_segment.parts) != 1
                or value in {".", ".."}
            ):
                raise ValueError(
                    f"{field_name} is not a safe path segment: {value!r}"
                )

            relative_path /= value

        for additional_path in additional:
            text = str(additional_path).strip()

            if not text:
                raise ValueError("Additional workflow paths must not be empty")

            posix_path = PurePosixPath(text.replace("\\", "/"))
            windows_path = PureWindowsPath(text)

            if (
                posix_path.is_absolute()
                or windows_path.is_absolute()
                or bool(windows_path.drive)
                or ".." in posix_path.parts
                or ".." in windows_path.parts
            ):
                raise ValueError(f"Unsafe workflow path: {additional_path!r}")

            relative_path = relative_path.joinpath(
                *(part for part in posix_path.parts if part not in {"", "."})
            )

        return self.resolve(relative_path)
