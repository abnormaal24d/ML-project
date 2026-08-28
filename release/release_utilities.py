"""Shared constants, errors, and filesystem helpers for release publishing."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

POINTER_SCHEMA = "production_release_pointer.v1"
RELEASE_SCHEMA = "production_release.v1"
RELEASES_DIRECTORY = "releases"
CURRENT_POINTER = "current.json"
PROMOTION_LOCK = ".promotion.lock"
STAGING_PREFIX = ".staging-"


class ProductionPromotionLockError(RuntimeError):
    """Raised when another process already owns the promotion lock."""


class ProductionPromotionValidationError(RuntimeError):
    """Raised when a staged or published release is incomplete."""


def atomic_write_json(*, path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def cleanup_staging_directories(releases_directory: Path) -> None:
    for path in releases_directory.glob(f"{STAGING_PREFIX}*"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def current_pointer_references(
    *,
    production_directory: Path,
    release_directory: Path,
) -> bool:
    try:
        pointer = read_json_object(production_directory / CURRENT_POINTER)
        relative = required_string(pointer, "release_directory")
        return contained_relative_path(
            root=production_directory,
            relative=relative,
        ) == release_directory.resolve(strict=True)
    except (OSError, ValueError, ProductionPromotionValidationError):
        return False


def require_separate_roots(
    *,
    candidate_directory: Path,
    production_directory: Path,
) -> None:
    if (
        candidate_directory == production_directory
        or candidate_directory.is_relative_to(production_directory)
        or production_directory.is_relative_to(candidate_directory)
    ):
        raise ValueError(
            "candidate and production directories must not overlap"
        )


def require_directory(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


def safe_segment(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or bool(PureWindowsPath(value).drive)
    ):
        raise ValueError("release artifact name is unsafe")
    return value


def contained_relative_path(*, root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ProductionPromotionValidationError(
            "production release path is unsafe"
        )
    windows_path = PureWindowsPath(relative)
    posix_path = PurePosixPath(relative)
    if (
        "\\" in relative
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.is_absolute()
    ):
        raise ProductionPromotionValidationError(
            "production release path is unsafe"
        )
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ProductionPromotionValidationError(
            "production release path is unsafe"
        )

    root = root.resolve(strict=True)
    candidate = root.joinpath(*parts).resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ProductionPromotionValidationError(
            "production release path escapes its root"
        )
    return candidate


def read_json_object(path: Path) -> dict[str, object]:
    try:
        if path.stat().st_size > 1024 * 1024:
            raise ProductionPromotionValidationError(
                f"JSON artifact exceeds size limit: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionPromotionValidationError(
            f"invalid JSON artifact: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ProductionPromotionValidationError(
            f"JSON artifact root must be an object: {path}"
        )
    return payload


def required_string(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ProductionPromotionValidationError(
            f"production release {name} must be non-empty"
        )
    return value


def required_sha256(payload: dict[str, object], name: str) -> str:
    value = required_string(payload, name).lower()
    if len(value) != 64:
        raise ProductionPromotionValidationError(
            f"production release {name} must be SHA-256"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProductionPromotionValidationError(
            f"production release {name} must be SHA-256"
        ) from exc
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() and not path.is_symlink():
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir() and not path.is_symlink():
            fsync_directory(path)
    fsync_directory(root)


def fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "POINTER_SCHEMA",
    "RELEASE_SCHEMA",
    "RELEASES_DIRECTORY",
    "CURRENT_POINTER",
    "PROMOTION_LOCK",
    "STAGING_PREFIX",
    "ProductionPromotionLockError",
    "ProductionPromotionValidationError",
    "atomic_write_json",
    "cleanup_staging_directories",
    "current_pointer_references",
    "require_separate_roots",
    "require_directory",
    "same_file",
    "safe_segment",
    "contained_relative_path",
    "read_json_object",
    "required_string",
    "required_sha256",
    "sha256",
    "fsync_tree",
    "fsync_directory",
]
