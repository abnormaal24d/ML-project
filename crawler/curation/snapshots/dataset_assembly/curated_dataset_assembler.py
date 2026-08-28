"""High-level curated snapshot builder orchestration."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

from crawler.curation.snapshots.dataset_assembly.curated_assembly_types import (
    CuratedAssemblyResult,
    CuratedDatasetAssemblerConfig,
    CuratedDatasetAssemblerDependencies,
)
from crawler.curation.snapshots.dataset_assembly.curated_processor_profile import (
    resolve_curated_processor_profile,
)
from crawler.curation.snapshots.dataset_assembly.curated_record_loader import (
    build_reused_assembly_result,
    count_raw_kinds,
    find_reusable_snapshot_id,
    load_curated_raw_records,
    resolve_snapshot_directory,
    resolve_snapshot_root,
)
from crawler.curation.snapshots.dataset_assembly.curated_snapshot_build_pipeline import (
    build_filtered_curated_bundle,
    write_curated_snapshot,
)
from mmcrawler_datasets.snapshots.publication import staged_snapshot

if TYPE_CHECKING:
    from pathlib import Path


class CuratedDatasetAssembler:
    """Build curated snapshot outputs from raw crawl artifacts."""

    def __init__(
        self,
        *,
        config: CuratedDatasetAssemblerConfig,
        dependencies: CuratedDatasetAssemblerDependencies,
    ) -> None:
        self._config = config
        self._dependencies = dependencies

    async def build(
        self, *, snapshot_id: str | None = None
    ) -> CuratedAssemblyResult:
        """Build a curated snapshot (async: awaits multimodal preprocessing)."""

        record_set = load_curated_raw_records(
            raw_manifest_reader=self._dependencies.raw_manifest_reader,
            relevant_kinds=self._config.relevant_kinds,
            settings_payload=self._config.snapshot_fingerprint_payload,
        )
        if snapshot_id is None:
            reusable_snapshot_id = find_reusable_snapshot_id(
                snapshot_root=self._snapshot_root(),
                snapshot_manifest_filename=(
                    self._config.dataset_paths.snapshot_manifest_filename
                ),
                content_fingerprint=record_set.content_fingerprint,
            )
            if reusable_snapshot_id is not None:
                return cast(
                    CuratedAssemblyResult,
                    build_reused_assembly_result(
                        logger=self._dependencies.logger,
                        snapshot_directory_resolver=(
                            self._dependencies.snapshot_directory_resolver
                        ),
                        project_root=self._config.project_root,
                        dataset_paths=self._config.dataset_paths,
                        snapshot_id=reusable_snapshot_id,
                        content_fingerprint=record_set.content_fingerprint,
                    ),
                )
            resolved_snapshot_id = (
                f"{self._dependencies.snapshot_id_factory()}_"
                f"{record_set.content_fingerprint[:12]}"
            )
        else:
            resolved_snapshot_id = snapshot_id

        final_snapshot_directory = resolve_snapshot_directory(
            snapshot_directory_resolver=(
                self._dependencies.snapshot_directory_resolver
            ),
            project_root=self._config.project_root,
            dataset_paths=self._config.dataset_paths,
            snapshot_id=resolved_snapshot_id,
        )

        self._dependencies.logger.info(
            "curated_snapshot_build_started",
            snapshot_id=resolved_snapshot_id,
            snapshot_directory=final_snapshot_directory.as_posix(),
            relevant_kinds=sorted(self._config.relevant_kinds),
            content_fingerprint=record_set.content_fingerprint,
        )

        raw_kind_counts = count_raw_kinds(raw_entries=record_set.raw_entries)
        processors_payload = self._config.snapshot_fingerprint_payload.get(
            "collection_processors",
            {},
        )
        profile = resolve_curated_processor_profile(
            processors_payload=processors_payload,
            raw_kind_counts=raw_kind_counts,
        )
        self._dependencies.logger.info(
            "multimodal_profile_active",
            dataset_subdirectory=self._config.dataset_paths.output_subdirectory,
            configured_image_ocr=profile.configured_image_ocr,
            configured_audio_transcription=profile.configured_audio_transcription,
            configured_video_transcription=profile.configured_video_transcription,
            configured_document_ocr=profile.configured_document_ocr,
            effective_image_ocr=profile.effective_image_ocr,
            effective_audio_transcription=profile.effective_audio_transcription,
            effective_video_transcription=profile.effective_video_transcription,
            effective_document_ocr=profile.effective_document_ocr,
            text_document_only=not any(
                raw_kind_counts.get(kind, 0) > 0
                for kind in ("image", "audio", "video", "document")
            ),
            raw_kind_counts=raw_kind_counts,
        )

        with staged_snapshot(
            final_snapshot_root=final_snapshot_directory,
            logger=self._dependencies.logger,
            replace_existing=False,
        ) as staging_directory:
            filtered_bundle = await build_filtered_curated_bundle(
                dependencies=self._dependencies,
                snapshot_id=resolved_snapshot_id,
                snapshot_directory=staging_directory,
                project_root=self._config.project_root,
                schema_version=self._config.settings.curated_schema_version,
                record_set=record_set,
            )
            staged_result = write_curated_snapshot(
                logger=self._dependencies.logger,
                dependencies=self._dependencies,
                dataset_paths=self._config.dataset_paths,
                schema_version=self._config.settings.curated_schema_version,
                snapshot_id=resolved_snapshot_id,
                snapshot_directory=staging_directory,
                record_set=record_set,
                filtered_bundle=filtered_bundle,
            )

        self._dependencies.logger.info(
            "curated_snapshot_built",
            snapshot_id=resolved_snapshot_id,
            snapshot_directory=final_snapshot_directory.as_posix(),
            raw_entries=len(record_set.raw_entries),
            documents=staged_result.documents,
            chunks=staged_result.chunks,
            images=staged_result.images,
            audio=staged_result.audio,
            video=staged_result.video,
            alignments=staged_result.alignments,
            source_runs=staged_result.source_run_ids,
        )

        return replace(
            staged_result,
            snapshot_directory=final_snapshot_directory,
        )

    def _snapshot_root(self) -> Path:
        return resolve_snapshot_root(
            project_root=self._config.project_root,
            base_output_directory=self._config.dataset_paths.curated_output_directory,
            configured_subdirectory=self._config.dataset_paths.output_subdirectory,
        )
