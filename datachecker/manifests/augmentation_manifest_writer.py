"""Writer for augmentation manifests from native workflow inventories."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from datachecker.manifests.artifact_manifest import (
    ArtifactManifest,
    format_manifest_path,
)
from datachecker.manifests.augmentation_manifest import AugmentationManifest
from datachecker.manifests.crawl_manifest import CrawlManifest
from datachecker.manifests.manifest_file_writer import ManifestWriterBase, Now
from datachecker.manifests.preprocessing_manifest import PreprocessingManifest
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus

_Manifest = TypeVar("_Manifest", bound=ArtifactManifest)

if TYPE_CHECKING:
    from config.path_resolution.workflow_artifact_paths import (
        ArtifactPathRegistry,
    )
    from datachecker.fingerprints import SettingsFingerprintCalculator
    from datachecker.inventory.curated_snapshot_inventory import (
        CuratedInventory,
        CuratedInventoryReader,
    )
    from datachecker.inventory.training_snapshot_inventory import (
        TrainingInventory,
        TrainingInventoryReader,
    )
    from datachecker.manifests.artifact_manifest import RunArtifactIdentity
    from datachecker.manifests.manifest_file_writer import ManifestFileWriter
    from logger.project_logger import ProjectLogger


class AugmentationManifestWriteError(RuntimeError):
    """Raised when current augmentation output cannot be promoted."""


class AugmentationManifestWriter(ManifestWriterBase):
    """Persist augmentation evidence from the native output inventories."""

    def __init__(
        self,
        *,
        artifact_path_registry: ArtifactPathRegistry,
        curated_inventory_reader: CuratedInventoryReader,
        training_inventory_reader: TrainingInventoryReader,
        settings_fingerprint_calculator: SettingsFingerprintCalculator,
        augmentation_settings_payload: dict[str, object],
        augmentation_strategy_payload: dict[str, object],
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
        self._curated_inventory_reader = curated_inventory_reader
        self._training_inventory_reader = training_inventory_reader
        self._settings_fingerprint_calculator = settings_fingerprint_calculator
        self._augmentation_settings_payload = augmentation_settings_payload
        self._augmentation_strategy_payload = augmentation_strategy_payload

    def write_augmentation_manifest(self) -> AugmentationManifest:
        """Validate current augmentation output and persist provenance."""

        (
            curated_inventory,
            training_inventory,
            augmented_inventory,
            output_fingerprint,
        ) = self._read_and_validate_inventories()
        preprocessing_manifest = self._require_preprocessing_manifest()
        manifest = self._build_manifest(
            curated_inventory=curated_inventory,
            training_inventory=training_inventory,
            augmented_inventory=augmented_inventory,
            preprocessing_manifest=preprocessing_manifest,
            output_fingerprint=output_fingerprint,
        )
        manifest_path = (
            self._artifact_path_registry.augmentation_manifest_path()
        )
        self._write_manifest(path=manifest_path, payload=manifest.to_payload())
        self._logger.info(
            "workflow_augmentation_manifest_written",
            manifest_path=format_manifest_path(manifest_path),
            augmented_training_directory=format_manifest_path(
                manifest.augmented_training_directory
            ),
        )
        return manifest

    def _read_and_validate_inventories(
        self,
    ) -> tuple[CuratedInventory, TrainingInventory, TrainingInventory, str]:
        curated_inventory = self._curated_inventory_reader.read()
        training_inventory = self._training_inventory_reader.read_training()
        augmented_inventory = self._training_inventory_reader.read_augmented()

        if not curated_inventory.schema_valid:
            raise AugmentationManifestWriteError(
                "cannot write augmentation manifest: curated input schema is "
                "invalid"
            )
        if not training_inventory.schema_valid:
            raise AugmentationManifestWriteError(
                "cannot write augmentation manifest: training input schema is "
                "invalid"
            )
        if not augmented_inventory.schema_valid:
            raise AugmentationManifestWriteError(
                "cannot write augmentation manifest: augmented output schema "
                "is invalid"
            )

        self._require_path(
            training_inventory.directory,
            field="training snapshot directory",
        )
        self._require_path(
            augmented_inventory.directory,
            field="augmented training directory",
        )
        self._require_path(
            augmented_inventory.manifest_path,
            field="augmented dataset manifest",
        )
        output_fingerprint = self._require_text(
            augmented_inventory.fingerprint,
            field="augmented output fingerprint",
        )
        return (
            curated_inventory,
            training_inventory,
            augmented_inventory,
            output_fingerprint,
        )

    def _build_manifest(
        self,
        *,
        curated_inventory: CuratedInventory,
        training_inventory: TrainingInventory,
        augmented_inventory: TrainingInventory,
        preprocessing_manifest: PreprocessingManifest,
        output_fingerprint: str,
    ) -> AugmentationManifest:
        """Build one completed augmentation manifest from current evidence."""

        return AugmentationManifest(
            **self._identity_fields(),
            preprocessing_manifest_hash=(
                self._settings_fingerprint_calculator.calculate(
                    payload=preprocessing_manifest.to_payload(),
                )
            ),
            augmentation_settings_hash=(
                self._settings_fingerprint_calculator.calculate(
                    payload=self._augmentation_settings_payload,
                )
            ),
            augmentation_strategy_hash=(
                self._settings_fingerprint_calculator.calculate(
                    payload=self._augmentation_strategy_payload,
                )
            ),
            output_fingerprint=output_fingerprint,
            training_snapshot_directory=self._require_path(
                training_inventory.directory,
                field="training snapshot directory",
            ),
            augmented_training_directory=self._require_path(
                augmented_inventory.directory,
                field="augmented training directory",
            ),
            augmented_dataset_manifest_path=self._require_path(
                augmented_inventory.manifest_path,
                field="augmented dataset manifest",
            ),
            input_chunk_count=curated_inventory.chunk_count,
            input_sample_count=training_inventory.sample_count,
            augmented_sample_count=augmented_inventory.sample_count,
            rejected_augmented_count=augmented_inventory.rejected_augmented_count,
            variants_by_modality=augmented_inventory.variants_by_modality,
            variants_by_operation=augmented_inventory.variants_by_operation,
            media_outputs={
                key: ArtifactManifest._normalize_value(value)
                for key, value in augmented_inventory.media_outputs.items()
            },
            rejections_by_modality=augmented_inventory.rejections_by_modality,
            quality_checks_passed=augmented_inventory.quality_checks_passed,
            built_at=self._utc_now_iso(),
            lifecycle_stage="augmented",
            status=WorkflowLifecycleStatus.COMPLETED,
            final=True,
        )

    def _require_preprocessing_manifest(self) -> PreprocessingManifest:
        manifest_path = (
            self._artifact_path_registry.preprocessing_manifest_path()
        )
        manifest = self._read_manifest(
            path=manifest_path,
            parser=PreprocessingManifest.from_payload,
            label="preprocessing",
        )
        self._require_same_generation(manifest)
        crawl_manifest = self._require_crawl_manifest()
        current_crawl_hash = self._settings_fingerprint_calculator.calculate(
            payload=crawl_manifest.to_payload(),
        )
        if manifest.crawl_manifest_hash != current_crawl_hash:
            raise AugmentationManifestWriteError(
                "preprocessing manifest is stale for current crawl manifest"
            )
        if (
            manifest.crawl_output_fingerprint
            != crawl_manifest.output_fingerprint
        ):
            raise AugmentationManifestWriteError(
                "preprocessing manifest is stale for current crawl output"
            )
        return manifest

    def _require_crawl_manifest(self) -> CrawlManifest:
        manifest = self._read_manifest(
            path=self._artifact_path_registry.crawl_manifest_path(),
            parser=CrawlManifest.from_payload,
            label="crawl",
        )
        self._require_same_generation(manifest)
        return manifest

    @staticmethod
    def _read_manifest(
        *,
        path: Path,
        parser: Callable[[dict[str, object]], _Manifest],
        label: str,
    ) -> _Manifest:
        if not path.is_file():
            raise AugmentationManifestWriteError(
                f"{label} manifest must exist before augmentation"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AugmentationManifestWriteError(
                f"cannot parse {label} manifest"
            ) from exc
        if not isinstance(payload, dict):
            raise AugmentationManifestWriteError(
                f"{label} manifest must be a JSON object"
            )
        try:
            return parser(payload)
        except (TypeError, ValueError) as exc:
            raise AugmentationManifestWriteError(
                f"{label} manifest is invalid"
            ) from exc

    @staticmethod
    def _require_path(path: Path | None, *, field: str) -> Path:
        if path is None:
            raise AugmentationManifestWriteError(
                f"cannot write augmentation manifest: {field} is missing"
            )
        return path

    @staticmethod
    def _require_text(value: str | None, *, field: str) -> str:
        if value is None or not value.strip():
            raise AugmentationManifestWriteError(
                f"cannot write augmentation manifest: {field} is missing"
            )
        return value
