"""Configuration and result models for curated snapshot assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from logger.project_logger import ProjectLogger
from mmcrawler_datasets.curated.timed_media import TimedMediaRecord

_TimedMediaRow = TypeVar("_TimedMediaRow", bound=TimedMediaRecord)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from config.settings.datasets import (
        CuratedDatasetAssemblerSettings,
        DatasetPathSettings,
    )
    from crawler.curation.documents.assembler import CuratedDocumentAssembler
    from crawler.curation.ingest.curation_input_loader import (
        CurationInputLoader,
    )
    from crawler.curation.ingest.schema.entry import RawManifestEntry
    from crawler.curation.publishing.dataset_export.curated_dataset_writer import (
        CuratedDatasetWriter,
    )
    from mmcrawler_datasets.assembly.text_chunk_splitter import (
        TextChunkSplitter,
    )
    from mmcrawler_datasets.curated.document import CuratedDocumentRecord
    from mmcrawler_datasets.curated.image import CuratedImageRecord
    from preprocessing.multimodal_preprocessor import MultimodalPreprocessor
    from preprocessing.preprocessed_document import PreprocessedDocument
    from preprocessing.preprocessing_input import PreprocessingInput


class SnapshotDirectoryResolver(Protocol):
    """Resolve a snapshot path without creating filesystem state."""

    def __call__(
        self,
        *,
        project_root: Path,
        base_output_directory: str,
        configured_subdirectory: str | None,
        snapshot_id: str,
    ) -> Path: ...


class DocumentCuratorFactory(Protocol):
    """Build the document curator for one snapshot directory."""

    def __call__(
        self,
        *,
        snapshot_directory: Path,
    ) -> CuratedDocumentAssembler: ...


class PreprocessingInputBuilder(Protocol):
    """Build preprocessing inputs from raw manifest entries."""

    def __call__(
        self,
        *,
        raw_entries: tuple[RawManifestEntry, ...],
    ) -> tuple[PreprocessingInput, ...]: ...


class DatasetWriterFactory(Protocol):
    """Build the dataset writer for one snapshot directory."""

    def __call__(
        self,
        *,
        snapshot_directory: Path,
    ) -> CuratedDatasetWriter: ...


class TimedMediaRowDeduper(Protocol):
    """Deduplicate one homogeneous timed-media record collection."""

    def __call__(
        self,
        *,
        rows: tuple[_TimedMediaRow, ...],
    ) -> tuple[_TimedMediaRow, ...]: ...


@dataclass(frozen=True, slots=True)
class CuratedValidationReport:
    """Structured curated validation result."""

    valid: bool
    errors: tuple[str, ...]
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class CuratedAssemblyResult:
    """Result metadata for a completed curated snapshot build."""

    snapshot_id: str
    snapshot_directory: Path
    documents: int
    chunks: int
    images: int
    audio: int
    video: int
    alignments: int
    source_run_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CuratedDatasetAssemblerConfig:
    """Static curated snapshot build settings and path context."""

    settings: CuratedDatasetAssemblerSettings
    dataset_paths: DatasetPathSettings
    project_root: Path
    relevant_kinds: frozenset[str]
    snapshot_fingerprint_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CuratedDatasetAssemblerDependencies:
    """Collaborators used to assemble and publish curated snapshots."""

    logger: ProjectLogger
    raw_manifest_reader: CurationInputLoader
    snapshot_id_factory: Callable[[], str]
    snapshot_directory_resolver: SnapshotDirectoryResolver
    document_curator_factory: DocumentCuratorFactory
    preprocessing_input_builder: PreprocessingInputBuilder
    preprocessing_phase_runner: MultimodalPreprocessor
    dataset_writer_factory: DatasetWriterFactory
    chunker: TextChunkSplitter
    document_deduper: Callable[
        ...,
        tuple[
            tuple[CuratedDocumentRecord, ...],
            dict[str, PreprocessedDocument],
        ],
    ]
    image_deduper: Callable[..., tuple[CuratedImageRecord, ...]]
    media_row_deduper: TimedMediaRowDeduper
    sync_row_assembler: Callable[..., tuple[dict[str, Any], ...]]
    sync_row_deduper: Callable[..., tuple[dict[str, Any], ...]]
    snapshot_validator: Callable[..., CuratedValidationReport]
    snapshot_manifest_writer: Callable[..., None]
