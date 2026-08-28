"""Keyframe and visual-proxy path resolution for training samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ValidatedArtifactPath",
    "relative_dataset_path",
]


def relative_dataset_path(
    *,
    dataset_root: Path,
    output_path: Path,
) -> str:
    """Return a stable POSIX path relative to the dataset root when possible."""

    try:
        return output_path.relative_to(dataset_root).as_posix()
    except ValueError:
        return output_path.as_posix()


@dataclass(frozen=True, slots=True)
class ValidatedArtifactPath:
    """A contained project-relative artifact path and its resolved source."""

    relative_path: str
    resolved_path: Path
    project_root: Path

    def __post_init__(self) -> None:
        relative = _relative_path(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                "validated artifact path must be project-relative"
            )
        root = self.project_root.resolve(strict=True)
        resolved = self.resolved_path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "validated artifact escapes project root"
            ) from exc
        if resolved != (root / relative).resolve(strict=True):
            raise ValueError("artifact relative and resolved paths disagree")
        if not resolved.is_file():
            raise ValueError("validated artifact must be a file")


def _resolve_artifact_path(
    *,
    raw_path: object,
    project_root: Path,
    curated_snapshot_directory: Path | None,
) -> ValidatedArtifactPath | None:
    path = _optional_path(raw_path)
    if path is None:
        return None
    candidate = _relative_path(path)
    root = project_root.resolve(strict=True)
    search_roots = tuple(
        value
        for value in (curated_snapshot_directory, project_root)
        if value is not None
    )
    for search_root in search_roots:
        try:
            approved_root = search_root.resolve(strict=True)
            approved_root.relative_to(root)
            resolved = (approved_root / candidate).resolve(strict=True)
            resolved.relative_to(approved_root)
            relative = resolved.relative_to(root)
        except FileNotFoundError:
            continue
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                "artifact path escapes its approved root"
            ) from exc
        if not resolved.is_file():
            continue
        return ValidatedArtifactPath(
            relative_path=relative.as_posix(),
            resolved_path=resolved,
            project_root=root,
        )
    return None


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path must be a contained relative path")
    return path


def _optional_path(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("artifact path must be text")
    path = value.strip()
    return path or None
