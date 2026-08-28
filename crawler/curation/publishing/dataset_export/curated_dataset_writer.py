"""Writer for curated layer record files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mmcrawler_datasets.curated.document import (
    ChunkRecord,
    CuratedDocumentRecord,
)
from mmcrawler_datasets.curated.image import CuratedImageRecord
from mmcrawler_datasets.curated.timed_media import (
    CuratedAudioRecord,
    CuratedVideoRecord,
)

if TYPE_CHECKING:
    from pathlib import Path

    from config.settings.datasets import (
        CuratedDatasetWriterSettings,
        DatasetPathSettings,
    )
    from crawler.curation.publishing.dataset_export.jsonl_writer import (
        JsonlWriter,
    )


class CuratedDatasetWriter:
    """Persist curated entities, views, and alignments under a snapshot."""

    def __init__(
        self,
        *,
        settings: CuratedDatasetWriterSettings,
        dataset_paths: DatasetPathSettings,
        snapshot_directory: Path,
        jsonl_writer: JsonlWriter,
    ) -> None:
        self._settings = settings
        self._dataset_paths = dataset_paths
        self._snapshot_directory = snapshot_directory
        self._jsonl_writer = jsonl_writer
        self._entities_root = (
            snapshot_directory / self._dataset_paths.curated_entities_directory
        )
        self._views_root = (
            snapshot_directory / self._dataset_paths.curated_views_directory
        )
        self._alignments_root = (
            snapshot_directory
            / self._dataset_paths.curated_alignments_directory
        )

    def write_documents(
        self,
        *,
        documents: tuple[CuratedDocumentRecord, ...],
    ) -> Path:
        """Write curated document records and return output path."""

        if not all(
            isinstance(document, CuratedDocumentRecord)
            for document in documents
        ):
            raise TypeError(
                "write_documents requires CuratedDocumentRecord values"
            )
        payloads = [document.to_dict() for document in documents]
        output_path = (
            self._entities_root
            / self._dataset_paths.curated_documents_filename
        )
        self._jsonl_writer.write(path=output_path, rows=payloads)
        self._jsonl_writer.write(
            path=(
                self._views_root
                / "text"
                / self._dataset_paths.curated_documents_filename
            ),
            rows=payloads,
        )
        return output_path

    def write_chunks(self, *, chunks: tuple[ChunkRecord, ...]) -> Path:
        """Write curated text segment records and return output path."""

        if not all(isinstance(chunk, ChunkRecord) for chunk in chunks):
            raise TypeError("write_chunks requires ChunkRecord values")
        payloads = [chunk.to_dict() for chunk in chunks]
        output_path = (
            self._entities_root / self._dataset_paths.curated_chunks_filename
        )
        self._jsonl_writer.write(path=output_path, rows=payloads)
        self._jsonl_writer.write(
            path=(
                self._views_root
                / "text"
                / self._dataset_paths.curated_chunks_filename
            ),
            rows=payloads,
        )
        return output_path

    def write_images(
        self,
        *,
        images: tuple[CuratedImageRecord, ...],
    ) -> Path:
        """Write curated image records and return output path."""

        if not all(isinstance(image, CuratedImageRecord) for image in images):
            raise TypeError("write_images requires CuratedImageRecord values")
        payloads = [image.to_dict() for image in images]
        output_path = (
            self._entities_root / self._dataset_paths.curated_images_filename
        )
        self._jsonl_writer.write(path=output_path, rows=payloads)
        self._jsonl_writer.write(
            path=(
                self._views_root
                / "image"
                / self._dataset_paths.curated_images_filename
            ),
            rows=payloads,
        )
        return output_path

    def write_audio(
        self,
        *,
        records: tuple[CuratedAudioRecord, ...],
    ) -> Path:
        """Write canonical curated audio records and return output path."""

        if not all(
            isinstance(record, CuratedAudioRecord) for record in records
        ):
            raise TypeError("write_audio requires CuratedAudioRecord values")
        payloads = [record.to_dict() for record in records]
        output_path = (
            self._entities_root / self._dataset_paths.curated_audio_filename
        )
        self._jsonl_writer.write(path=output_path, rows=payloads)
        self._jsonl_writer.write(
            path=(
                self._views_root
                / "audio"
                / self._dataset_paths.curated_audio_filename
            ),
            rows=payloads,
        )
        return output_path

    def write_video(
        self,
        *,
        records: tuple[CuratedVideoRecord, ...],
    ) -> Path:
        """Write canonical curated video records and return output path."""

        if not all(
            isinstance(record, CuratedVideoRecord) for record in records
        ):
            raise TypeError("write_video requires CuratedVideoRecord values")
        payloads = [record.to_dict() for record in records]
        output_path = (
            self._entities_root / self._dataset_paths.curated_video_filename
        )
        self._jsonl_writer.write(path=output_path, rows=payloads)
        self._jsonl_writer.write(
            path=(
                self._views_root
                / "video"
                / self._dataset_paths.curated_video_filename
            ),
            rows=payloads,
        )
        return output_path

    def write_cross_modal_alignments(
        self,
        *,
        rows: tuple[dict[str, Any], ...],
    ) -> Path:
        """Write cross-modal alignment rows for curated lineage."""

        payloads = list(rows)
        output_path = (
            self._alignments_root
            / self._dataset_paths.curated_sync_links_filename
        )
        self._jsonl_writer.write(path=output_path, rows=payloads)
        self._write_alignment_views(rows=payloads)
        return output_path

    def _write_alignment_views(self, *, rows: list[dict[str, Any]]) -> None:
        """Persist first-class alignment views by target modality."""

        image_rows = self._filter_alignment_rows(rows=rows, modality="image")
        audio_rows = self._filter_alignment_rows(rows=rows, modality="audio")
        video_rows = self._filter_alignment_rows(rows=rows, modality="video")

        self._jsonl_writer.write(
            path=(
                self._alignments_root
                / self._dataset_paths.curated_text_image_alignments_filename
            ),
            rows=image_rows,
        )
        self._jsonl_writer.write(
            path=(
                self._alignments_root
                / self._dataset_paths.curated_text_audio_alignments_filename
            ),
            rows=audio_rows,
        )
        self._jsonl_writer.write(
            path=(
                self._alignments_root
                / self._dataset_paths.curated_text_video_alignments_filename
            ),
            rows=video_rows,
        )
        self._jsonl_writer.write(
            path=(
                self._alignments_root
                / self._dataset_paths.curated_image_audio_alignments_filename
            ),
            rows=self._filter_image_audio_alignment_rows(rows=rows),
        )

    @staticmethod
    def _filter_alignment_rows(
        *,
        rows: list[dict[str, Any]],
        modality: str,
    ) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if str(row.get("object_modality") or "").strip().lower()
            == modality
        ]

    @staticmethod
    def _filter_image_audio_alignment_rows(
        *,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            row for row in rows if row.get("image_id") and row.get("audio_id")
        ]
