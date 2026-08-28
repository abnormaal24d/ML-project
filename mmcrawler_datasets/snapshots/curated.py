"""Read one complete curated snapshot as a typed read-model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from mmcrawler_datasets.curated.document import (
    ChunkRecord,
    CuratedDocumentRecord,
)
from mmcrawler_datasets.curated.image import CuratedImageRecord
from mmcrawler_datasets.curated.timed_media import (
    CURATED_AUDIO_CONTRACT_SHA256,
    CURATED_VIDEO_CONTRACT_SHA256,
    CuratedAudioRecord,
    CuratedVideoRecord,
)
from mmcrawler_datasets.safe_io import iter_jsonl, load_bounded_json_object
from schemas.versions import CURATED_DATASET_SCHEMA_VERSION

if TYPE_CHECKING:
    from config.settings.datasets import DatasetPathSettings

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CuratedSnapshot:
    """Typed read-model of one complete curated snapshot."""

    documents: tuple[CuratedDocumentRecord, ...]
    chunks: tuple[ChunkRecord, ...]
    images: tuple[CuratedImageRecord, ...]
    audio: tuple[CuratedAudioRecord, ...]
    video: tuple[CuratedVideoRecord, ...]

    @property
    def documents_by_id(self) -> dict[str, CuratedDocumentRecord]:
        """Index documents by their stable identity."""

        return {document.document_id: document for document in self.documents}


class SnapshotContractError(RuntimeError):
    """The curated snapshot violates its persisted contract."""


def read_snapshot(
    *,
    dataset_paths: DatasetPathSettings,
    snapshot_directory: Path,
) -> CuratedSnapshot:
    """Read one complete snapshot or reject it without partial results."""

    manifest_path = (
        snapshot_directory / dataset_paths.snapshot_manifest_filename
    )
    _validate_manifest(manifest_path=manifest_path)
    entities = snapshot_directory / dataset_paths.curated_entities_directory
    return CuratedSnapshot(
        documents=_read_entity(
            path=entities / dataset_paths.curated_documents_filename,
            decoder=CuratedDocumentRecord.from_dict,
            entity="document",
        ),
        chunks=_read_entity(
            path=entities / dataset_paths.curated_chunks_filename,
            decoder=ChunkRecord.from_dict,
            entity="chunk",
        ),
        images=_read_entity(
            path=entities / dataset_paths.curated_images_filename,
            decoder=CuratedImageRecord.from_dict,
            entity="image",
        ),
        audio=_read_entity(
            path=entities / dataset_paths.curated_audio_filename,
            decoder=CuratedAudioRecord.model_validate,
            entity="audio",
        ),
        video=_read_entity(
            path=entities / dataset_paths.curated_video_filename,
            decoder=CuratedVideoRecord.model_validate,
            entity="video",
        ),
    )


def _validate_manifest(*, manifest_path: Path) -> None:
    try:
        manifest = load_bounded_json_object(path=manifest_path)
    except ValueError as exc:
        raise SnapshotContractError(str(exc)) from exc
    observed_version = manifest.get("schema_version")
    if observed_version != CURATED_DATASET_SCHEMA_VERSION:
        raise SnapshotContractError(
            f"curated snapshot schema mismatch at {manifest_path}: "
            f"expected={CURATED_DATASET_SCHEMA_VERSION!r}, "
            f"observed={observed_version!r}"
        )
    _require_digest(
        manifest=manifest,
        manifest_path=manifest_path,
        key="curated_audio_contract_sha256",
        expected=CURATED_AUDIO_CONTRACT_SHA256,
    )
    _require_digest(
        manifest=manifest,
        manifest_path=manifest_path,
        key="curated_video_contract_sha256",
        expected=CURATED_VIDEO_CONTRACT_SHA256,
    )


def _require_digest(
    *,
    manifest: Mapping[str, object],
    manifest_path: Path,
    key: str,
    expected: str,
) -> None:
    observed = manifest.get(key)
    if observed != expected:
        raise SnapshotContractError(
            f"curated contract digest mismatch at {manifest_path}: "
            f"field={key!r}, expected={expected!r}, observed={observed!r}"
        )


def _read_entity(
    *,
    path: Path,
    decoder: Callable[[Mapping[str, object]], T],
    entity: str,
) -> tuple[T, ...]:
    decoded: list[T] = []
    try:
        for line_number, row in enumerate(iter_jsonl(path=path), start=1):
            observed = row.get("schema_version")
            if observed != CURATED_DATASET_SCHEMA_VERSION:
                raise SnapshotContractError(
                    f"curated {entity} schema mismatch at {path}:{line_number}: "
                    f"expected={CURATED_DATASET_SCHEMA_VERSION!r}, "
                    f"observed={observed!r}"
                )
            try:
                decoded.append(decoder(row))
            except (KeyError, TypeError, ValueError) as exc:
                raise SnapshotContractError(
                    f"invalid curated {entity} record at {path}:{line_number}"
                ) from exc
    except ValueError as exc:
        raise SnapshotContractError(str(exc)) from exc
    return tuple(decoded)


__all__ = ["CuratedSnapshot", "SnapshotContractError", "read_snapshot"]
