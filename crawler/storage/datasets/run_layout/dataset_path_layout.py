"""Dataset output and run directory path layout."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings.datasets import DatasetPathSettings

from config.path_resolution.project_paths import join_contained_path

_CANONICAL_OUTPUT_SUBDIRECTORY = "multimodal"

_CANONICAL_MODALITY_FILENAME_ATTRIBUTES: dict[str, str] = {
    "page": "raw_pages_filename",
    "feed": "raw_feeds_filename",
    "document": "raw_documents_filename",
    "image": "raw_images_filename",
    "audio": "raw_audio_filename",
    "video": "raw_video_filename",
}

CANONICAL_MODALITIES: tuple[str, ...] = tuple(
    _CANONICAL_MODALITY_FILENAME_ATTRIBUTES.keys()
)


def canonical_modality_filename(
    *,
    dataset_paths: DatasetPathSettings,
    modality: str,
) -> str | None:
    """Return the configured canonical filename for a modality."""

    attribute_name = _CANONICAL_MODALITY_FILENAME_ATTRIBUTES.get(modality)
    if attribute_name is None:
        return None

    filename = getattr(dataset_paths, attribute_name, None)
    if filename is None:
        return None

    return str(filename)


def output_subdirectory(
    *,
    configured_subdirectory: str | None,
) -> str:
    """
    Return the configured output subdirectory or the canonical default.
    """

    if configured_subdirectory is None:
        return _CANONICAL_OUTPUT_SUBDIRECTORY

    stripped = configured_subdirectory.strip()
    if stripped:
        return stripped

    return _CANONICAL_OUTPUT_SUBDIRECTORY


def output_directory(
    *,
    root: str | Path,
    configured_subdirectory: str | None,
) -> Path:
    """Return the output directory for the single supported workflow."""

    return Path(root) / output_subdirectory(
        configured_subdirectory=configured_subdirectory,
    )


def run_directory(
    *,
    root: str | Path,
    configured_subdirectory: str | None,
    run_id: str,
) -> Path:
    """Return the run directory for the single supported workflow."""

    return join_contained_path(
        output_directory(
            root=root,
            configured_subdirectory=configured_subdirectory,
        ),
        run_id,
        field_name="run_id",
        single_segment=True,
    )


def build_run_directory(
    *,
    project_root: Path,
    base_output_directory: str,
    configured_subdirectory: str | None,
    run_id: str,
) -> Path:
    """Build and create a run directory for runtime callers."""

    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        raise ValueError("run_id must not be empty")

    directory = run_directory(
        root=project_root / base_output_directory,
        configured_subdirectory=configured_subdirectory,
        run_id=normalized_run_id,
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def snapshot_directory(
    *,
    project_root: Path,
    base_output_directory: str,
    configured_subdirectory: str | None,
    snapshot_id: str,
) -> Path:
    """Resolve a snapshot directory without creating it."""

    base_root = join_contained_path(
        project_root,
        base_output_directory,
        field_name="base_output_directory",
    )

    return join_contained_path(
        output_directory(
            root=base_root,
            configured_subdirectory=configured_subdirectory,
        ),
        snapshot_id,
        field_name="snapshot_id",
        single_segment=True,
    )


def build_snapshot_directory(
    *,
    project_root: Path,
    base_output_directory: str,
    configured_subdirectory: str | None,
    snapshot_id: str,
) -> Path:
    """Resolve and create a snapshot directory for runtime callers."""

    directory = snapshot_directory(
        project_root=project_root,
        base_output_directory=base_output_directory,
        configured_subdirectory=configured_subdirectory,
        snapshot_id=snapshot_id,
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory
