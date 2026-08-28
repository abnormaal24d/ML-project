"""Writer for preprocessing manifests from native workflow inventories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from datachecker.manifests.artifact_manifest import format_manifest_path
from datachecker.manifests.crawl_manifest import CrawlManifest
from datachecker.manifests.manifest_file_writer import ManifestWriterBase, Now
from datachecker.manifests.preprocessing_manifest import PreprocessingManifest
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus

if TYPE_CHECKING:
    from config.path_resolution.workflow_artifact_paths import (
        ArtifactPathRegistry,
    )
    from datachecker.fingerprints import (
        FileFingerprintCalculator,
        SettingsFingerprintCalculator,
    )
    from datachecker.inventory.curated_snapshot_inventory import (
        CuratedInventoryReader,
    )
    from datachecker.inventory.raw_run_inventory import RawInventoryReader
    from datachecker.inventory.training_snapshot_inventory import (
        TrainingInventoryReader,
    )
    from datachecker.manifests.artifact_manifest import RunArtifactIdentity
    from datachecker.manifests.manifest_file_writer import ManifestFileWriter
    from logger.project_logger import ProjectLogger


class PreprocessingManifestWriter(ManifestWriterBase):
    """Persist preprocessing evidence from the three native inventories."""

    def __init__(
        self,
        *,
        artifact_path_registry: ArtifactPathRegistry,
        raw_inventory_reader: RawInventoryReader,
        curated_inventory_reader: CuratedInventoryReader,
        training_inventory_reader: TrainingInventoryReader,
        settings_fingerprint_calculator: SettingsFingerprintCalculator,
        file_fingerprint_calculator: FileFingerprintCalculator,
        preprocessing_settings_payload: dict[str, object],
        normalization_settings_payload: dict[str, object],
        deduplication_settings_payload: dict[str, object],
        splitting_settings_payload: dict[str, object],
        validation_settings_payload: dict[str, object],
        logger: ProjectLogger,
        project_root: Path | None,
        file_writer: ManifestFileWriter,
        artifact_identity: RunArtifactIdentity,
        now: Now,
    ) -> None:
        super().__init__(
            artifact_path_registry=artifact_path_registry,
            logger=logger,
            project_root=project_root,
            file_writer=file_writer,
            artifact_identity=artifact_identity,
            now=now,
        )
        self._raw_inventory_reader = raw_inventory_reader
        self._curated_inventory_reader = curated_inventory_reader
        self._training_inventory_reader = training_inventory_reader
        self._settings_fingerprint_calculator = settings_fingerprint_calculator
        self._file_fingerprint_calculator = file_fingerprint_calculator
        self._preprocessing_settings_payload = preprocessing_settings_payload
        self._normalization_settings_payload = normalization_settings_payload
        self._deduplication_settings_payload = deduplication_settings_payload
        self._splitting_settings_payload = splitting_settings_payload
        self._validation_settings_payload = validation_settings_payload

    def write_preprocessing_manifest(self) -> PreprocessingManifest:
        """Validate current output and persist preprocessing provenance."""

        crawl_manifest = self._require_crawl_manifest()
        raw_run_directory = self._require_path(
            crawl_manifest.raw_run_directory,
            field="raw run directory",
        )
        run_summary_path = self._require_path(
            crawl_manifest.run_summary_path,
            field="run summary path",
        )
        crawl_output_fingerprint = self._require_text(
            crawl_manifest.output_fingerprint,
            field="crawl output fingerprint",
        )

        raw_inventory = self._raw_inventory_reader.read(
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
        )
        if not raw_inventory.schema_valid:
            raise RuntimeError(
                "cannot write preprocessing manifest: raw crawl schema is invalid"
            )
        if raw_inventory.fingerprint != crawl_output_fingerprint:
            raise RuntimeError(
                "cannot write preprocessing manifest: raw crawl output changed "
                "after crawl manifest promotion"
            )

        curated_inventory = self._curated_inventory_reader.read()
        training_inventory = self._training_inventory_reader.read_training()
        curated_snapshot_directory = self._require_path(
            curated_inventory.directory,
            field="curated snapshot directory",
        )
        curated_snapshot_manifest_path = self._require_path(
            curated_inventory.manifest_path,
            field="curated snapshot manifest",
        )
        training_snapshot_directory = self._require_path(
            training_inventory.directory,
            field="training snapshot directory",
        )
        training_dataset_manifest_path = self._require_path(
            training_inventory.manifest_path,
            field="training dataset manifest",
        )
        output_fingerprint = self._require_text(
            training_inventory.fingerprint,
            field="training output fingerprint",
        )
        if not curated_inventory.schema_valid:
            raise RuntimeError(
                "cannot write preprocessing manifest: curated snapshot schema "
                "is invalid"
            )
        if not training_inventory.schema_valid:
            raise RuntimeError(
                "cannot write preprocessing manifest: training snapshot schema "
                "is invalid"
            )

        manifest = PreprocessingManifest(
            **self._identity_fields(),
            crawl_manifest_hash=self._settings_fingerprint_calculator.calculate(
                payload=crawl_manifest.to_payload(),
            ),
            crawl_output_fingerprint=crawl_output_fingerprint,
            preprocessing_settings_hash=(
                self._settings_fingerprint_calculator.calculate(
                    payload=self._preprocessing_settings_payload,
                )
            ),
            normalization_settings_hash=(
                self._settings_fingerprint_calculator.calculate(
                    payload=self._normalization_settings_payload,
                )
            ),
            deduplication_settings_hash=(
                self._settings_fingerprint_calculator.calculate(
                    payload=self._deduplication_settings_payload,
                )
            ),
            splitting_settings_hash=(
                self._settings_fingerprint_calculator.calculate(
                    payload=self._splitting_settings_payload,
                )
            ),
            validation_settings_hash=(
                self._settings_fingerprint_calculator.calculate(
                    payload=self._validation_settings_payload,
                )
            ),
            output_fingerprint=output_fingerprint,
            curated_snapshot_directory=curated_snapshot_directory,
            curated_snapshot_manifest_path=curated_snapshot_manifest_path,
            training_snapshot_directory=training_snapshot_directory,
            training_dataset_manifest_path=training_dataset_manifest_path,
            training_dataset_manifest_hash=(
                self._file_fingerprint_calculator.calculate(
                    path=training_dataset_manifest_path,
                )
            ),
            input_document_count=raw_inventory.fetched_url_count,
            output_document_count=curated_inventory.document_count,
            output_chunk_count=curated_inventory.chunk_count,
            output_image_count=curated_inventory.image_count,
            output_audio_count=curated_inventory.audio_count,
            output_video_count=curated_inventory.video_count,
            output_alignment_count=curated_inventory.alignment_count,
            training_sample_count=training_inventory.sample_count,
            rejected_document_count=curated_inventory.rejected_document_count,
            rejected_image_count=curated_inventory.rejected_image_count,
            rejected_audio_count=curated_inventory.rejected_audio_count,
            rejected_video_count=curated_inventory.rejected_video_count,
            image_coverage=curated_inventory.image_coverage,
            audio_coverage=curated_inventory.audio_coverage,
            video_coverage=curated_inventory.video_coverage,
            built_at=self._utc_now_iso(),
            lifecycle_stage="preprocessed",
            status=WorkflowLifecycleStatus.COMPLETED,
            final=True,
        )

        manifest_path = (
            self._artifact_path_registry.preprocessing_manifest_path()
        )
        self._write_manifest(path=manifest_path, payload=manifest.to_payload())
        self._logger.info(
            "workflow_preprocessing_manifest_written",
            manifest_path=format_manifest_path(manifest_path),
            raw_run_directory=format_manifest_path(raw_run_directory),
            run_summary_path=format_manifest_path(run_summary_path),
            curated_snapshot_directory=format_manifest_path(
                curated_snapshot_directory
            ),
            curated_snapshot_manifest_path=format_manifest_path(
                curated_snapshot_manifest_path
            ),
            training_snapshot_directory=format_manifest_path(
                training_snapshot_directory
            ),
            training_dataset_manifest_path=format_manifest_path(
                training_dataset_manifest_path
            ),
            input_document_count=manifest.input_document_count,
            output_document_count=manifest.output_document_count,
            output_chunk_count=manifest.output_chunk_count,
            training_sample_count=manifest.training_sample_count,
            output_fingerprint=manifest.output_fingerprint,
        )
        return manifest

    def _require_crawl_manifest(self) -> CrawlManifest:
        """Return the current generation's valid crawl manifest."""

        manifest_path = self._artifact_path_registry.crawl_manifest_path()
        if not manifest_path.is_file():
            raise RuntimeError(
                "crawl manifest must exist before writing the preprocessing "
                "manifest"
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "cannot parse crawl manifest before preprocessing"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("crawl manifest must be a JSON object")
        try:
            manifest = CrawlManifest.from_payload(payload)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("crawl manifest is invalid") from exc
        self._require_same_generation(manifest)
        return manifest

    @staticmethod
    def _require_path(path: Path | None, *, field: str) -> Path:
        if path is None:
            raise RuntimeError(
                f"cannot write preprocessing manifest: {field} is missing"
            )
        return path

    @staticmethod
    def _require_text(value: str | None, *, field: str) -> str:
        if value is None or not value.strip():
            raise RuntimeError(
                f"cannot write preprocessing manifest: {field} is missing"
            )
        return value
