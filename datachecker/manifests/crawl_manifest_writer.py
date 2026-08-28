"""Writer for canonical crawl manifests from raw crawl output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
from uuid import uuid4

from datachecker.manifests.artifact_manifest import format_manifest_path
from datachecker.manifests.crawl_manifest import (
    CoverageSummary,
    CrawlManifest,
    SchemaValidationSummary,
)
from datachecker.manifests.manifest_file_writer import (
    ManifestFileWriter,
    ManifestWriterBase,
    Now,
)
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus

# Point 30: centralized pipeline counter keys
PIPELINE_COUNTER_KEYS: tuple[str, ...] = (
    "relationships_count",
    "metadata_count",
    "updates_count",
    "errors_count",
    "discovered_assets_count",
    "rejected_assets_count",
    "manifest_write_count",
    "total_bytes_written",
)


class CrawlManifestWriteError(RuntimeError):
    """Specific error for crawl manifest write failures."""

    pass


class CorruptCrawlManifestError(CrawlManifestWriteError):
    """Raised after an existing crawl manifest fails strict validation."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        quarantine_path: Path | None,
        corruption_type: str,
        corruption_message: str,
        quarantine_error: str | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.quarantine_path = quarantine_path
        self.corruption_type = corruption_type
        self.corruption_message = corruption_message
        self.quarantine_error = quarantine_error

        quarantine_detail = (
            f"quarantined at {quarantine_path}"
            if quarantine_path is not None
            else "quarantine failed; original file was left in place"
        )
        message = (
            f"existing crawl manifest is corrupt: {manifest_path}; "
            f"cause={corruption_type}: {corruption_message}; "
            f"{quarantine_detail}"
        )
        if quarantine_error:
            message = f"{message}; quarantine_error={quarantine_error}"
        super().__init__(message)


if TYPE_CHECKING:
    from config.collection.training_input_gate import CrawlOutputGateSettings
    from config.path_resolution.workflow_artifact_paths import (
        ArtifactPathRegistry,
    )
    from config.source_catalog.catalog_settings import SourceProfileSettings
    from datachecker.fingerprints import (
        FileFingerprintCalculator,
        SettingsFingerprintCalculator,
        SourceFingerprintCalculator,
    )
    from datachecker.inventory.raw_run_inventory import (
        RawInventory,
        RawInventoryReader,
    )
    from datachecker.manifests.artifact_manifest import RunArtifactIdentity
    from logger.project_logger import ProjectLogger


class CrawlManifestWriter(ManifestWriterBase):
    """Persist canonical crawl manifests from raw crawl output."""

    def __init__(
        self,
        *,
        artifact_path_registry: ArtifactPathRegistry,
        raw_inventory_reader: RawInventoryReader,
        settings_fingerprint_calculator: SettingsFingerprintCalculator,
        source_fingerprint_calculator: SourceFingerprintCalculator,
        file_fingerprint_calculator: FileFingerprintCalculator,
        crawl_settings_payload: dict[str, object],
        seed_urls: tuple[str, ...],
        source_profile: SourceProfileSettings,
        logger: ProjectLogger,
        project_root: Path | None,
        file_writer: ManifestFileWriter,
        artifact_identity: RunArtifactIdentity,
        crawl_output_gate: CrawlOutputGateSettings,
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
        self._settings_fingerprint_calculator = settings_fingerprint_calculator
        self._source_fingerprint_calculator = source_fingerprint_calculator
        self._file_fingerprint_calculator = file_fingerprint_calculator
        self._crawl_settings_payload = crawl_settings_payload
        self._seed_urls = seed_urls
        self._source_profile = source_profile
        self._crawl_output_gate = crawl_output_gate

    def crawl_manifest_path(self) -> Path:
        return Path(self._artifact_path_registry.crawl_manifest_path())

    def has_finalized_raw_output(
        self,
        *,
        raw_run_directory: Path,
        run_summary_path: Path,
        attempt_id: str,
        raw_run_id: str,
        crawl_session_id: str,
    ) -> bool:
        """Return whether raw crawl artifacts are ready for manifest promotion."""

        inventory = self._raw_inventory_reader.read(
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
        )
        try:
            self._validate_promotion_candidate(
                inventory=inventory,
                raw_run_directory=raw_run_directory,
                run_summary_path=run_summary_path,
                attempt_id=attempt_id,
                raw_run_id=raw_run_id,
                crawl_session_id=crawl_session_id,
            )
        except CrawlManifestWriteError:
            return False
        return True

    def write_crawl_manifest(
        self,
        *,
        raw_run_directory: Path,
        run_summary_path: Path,
        attempt_id: str,
        raw_run_id: str,
        crawl_session_id: str,
    ) -> CrawlManifest:
        """Persist the canonical crawl manifest."""

        inventory = self._raw_inventory_reader.read(
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
        )
        manifest_path = self.crawl_manifest_path()
        # Point 32: cache timestamp per write
        now_iso = self._utc_now_iso()
        existing_manifest = self._read_existing_manifest(
            path=manifest_path,
            quarantined_at=now_iso,
        )

        self._validate_promotion_candidate(
            inventory=inventory,
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
            attempt_id=attempt_id,
            raw_run_id=raw_run_id,
            crawl_session_id=crawl_session_id,
        )

        if (
            existing_manifest is not None
            and existing_manifest.identity_fields() == self._identity_fields()
            and inventory.fingerprint is not None
            and existing_manifest.output_fingerprint == inventory.fingerprint
        ):
            self._logger.info(
                "workflow_crawl_manifest_reused",
                manifest_path=format_manifest_path(manifest_path),
                output_fingerprint=inventory.fingerprint,
            )
            return existing_manifest

        output_fingerprint = inventory.fingerprint
        inventory_raw_run_directory = inventory.directory
        raw_records_manifest_path = inventory.records_path
        raw_errors_manifest_path = inventory.errors_path
        raw_run_summary_path = inventory.summary_path
        if (
            output_fingerprint is None
            or inventory_raw_run_directory is None
            or raw_records_manifest_path is None
            or raw_errors_manifest_path is None
            or raw_run_summary_path is None
        ):
            raise CrawlManifestWriteError(
                "cannot write crawl manifest without finalized raw crawl "
                "output"
            )

        if not raw_records_manifest_path.exists():
            raise CrawlManifestWriteError(
                "raw_records_manifest_path must physically exist before "
                "writing crawl manifest"
            )

        coverage_summary = self._build_coverage_summary(
            inventory=inventory,
            raw_run_summary_path=raw_run_summary_path,
        )
        schema_validation_summary = SchemaValidationSummary(
            raw_schema_valid=inventory.schema_valid,
            reasons=list(inventory.raw_schema_errors),
        )

        raw_object_records_total = inventory.unique_valid_object_count

        manifest = CrawlManifest(
            **self._identity_fields(),
            source_registry_hash=(
                self._source_fingerprint_calculator.calculate(
                    seed_urls=self._seed_urls,
                    source_profile=self._source_profile,
                )
            ),
            crawl_settings_hash=(
                self._settings_fingerprint_calculator.calculate(
                    payload=self._crawl_settings_payload,
                )
            ),
            output_fingerprint=output_fingerprint,
            raw_run_directory=inventory_raw_run_directory,
            raw_records_manifest_path=raw_records_manifest_path,
            raw_errors_manifest_path=raw_errors_manifest_path,
            run_summary_path=raw_run_summary_path,
            fetched_url_count=inventory.fetched_url_count,
            failed_url_count=inventory.failed_url_count,
            output_file_count=inventory.file_count,
            raw_object_records_total=raw_object_records_total,
            started_at=inventory.started_at,
            completed_at=inventory.completed_at or now_iso,
            lifecycle_stage="raw",
            status=WorkflowLifecycleStatus.COMPLETED,
            final=True,
            records_manifest_hash=(
                self._file_fingerprint_calculator.calculate(
                    path=raw_records_manifest_path,
                )
            ),
            coverage_summary=coverage_summary,
            schema_validation_summary=schema_validation_summary,
        )
        self._write_manifest(path=manifest_path, payload=manifest.to_payload())
        self._logger.info(
            "workflow_crawl_manifest_written",
            manifest_path=format_manifest_path(manifest_path),
            raw_run_directory=format_manifest_path(manifest.raw_run_directory),
        )
        return manifest

    def _build_coverage_summary(
        self,
        *,
        inventory: RawInventory,
        raw_run_summary_path: Path,
    ) -> CoverageSummary:
        modality_counts = inventory.modality_counts
        minimum_modality_counts = self._minimum_modality_counts()
        missing = {
            modality: max(
                0,
                minimum_modality_counts.get(modality, 0)
                - modality_counts.get(modality, 0),
            )
            for modality in set(modality_counts) | set(minimum_modality_counts)
        }
        pipeline_counters = self._read_pipeline_counters(
            summary_path=raw_run_summary_path,
            logger=self._logger,
        )
        return CoverageSummary(
            modality_counts=dict(modality_counts),
            minimum_modality_counts=dict(minimum_modality_counts),
            missing={
                key: value for key, value in missing.items() if value > 0
            },
            pipeline_counters=dict(pipeline_counters),
        )

    def _validate_promotion_candidate(
        self,
        *,
        inventory: RawInventory,
        raw_run_directory: Path,
        run_summary_path: Path,
        attempt_id: str,
        raw_run_id: str,
        crawl_session_id: str,
    ) -> None:
        if self._raw_crawl_output_missing(inventory=inventory):
            raise CrawlManifestWriteError(
                "cannot write crawl manifest without finalized raw crawl output"
            )
        if inventory.unique_valid_object_count <= 0:
            raise CrawlManifestWriteError(
                "refusing empty canonical promotion: no valid current objects"
            )

        payload = self._read_summary_payload(summary_path=run_summary_path)
        expected_identity = {
            "attempt_id": attempt_id,
            "raw_run_id": raw_run_id,
            "generation_id": self._artifact_identity.generation_id,
            "workflow_id": self._artifact_identity.workflow_id,
            "crawl_session_id": crawl_session_id,
        }
        for field, expected in expected_identity.items():
            actual = payload.get(field)
            if not isinstance(actual, str) or actual != expected:
                raise CrawlManifestWriteError(
                    f"raw run summary identity mismatch for {field}"
                )
        self._require_summary_path(
            payload=payload,
            field="raw_run_directory",
            expected=raw_run_directory,
        )
        self._require_summary_path(
            payload=payload,
            field="run_summary_path",
            expected=run_summary_path,
        )
        if (
            payload.get("status") != "completed"
            or payload.get("final") is not True
        ):
            raise CrawlManifestWriteError(
                "raw run summary is not a completed final run"
            )

        summary_total = self._strict_nonnegative_count(
            payload.get("object_records_total"),
            field="object_records_total",
        )
        if summary_total != inventory.unique_valid_object_count:
            raise CrawlManifestWriteError(
                "raw run summary object count does not match valid current objects"
            )
        summary_modalities = self._strict_modality_counts(
            payload.get("modality_counts"),
            field="modality_counts",
        )
        if summary_modalities != self._normalized_modality_counts(
            inventory.modality_counts
        ):
            raise CrawlManifestWriteError(
                "raw run summary modality counts do not match valid current objects"
            )

        readiness = payload.get("output_readiness")
        if not isinstance(readiness, dict):
            raise CrawlManifestWriteError(
                "raw run lacks output readiness evidence"
            )
        self._validate_readiness(
            readiness=readiness,
            object_records_total=inventory.unique_valid_object_count,
            modality_counts=summary_modalities,
        )

    def _validate_readiness(
        self,
        *,
        readiness: dict[str, object],
        object_records_total: int,
        modality_counts: dict[str, int],
    ) -> None:
        ready = readiness.get("ready")
        if not isinstance(ready, bool):
            raise CrawlManifestWriteError(
                "output_readiness.ready must be a boolean"
            )
        unmet = readiness.get("unmet_requirements")
        if ready and (not isinstance(unmet, (list, tuple)) or unmet):
            raise CrawlManifestWriteError(
                "a ready output readiness report must have an "
                "empty unmet_requirements list"
            )
        if (
            not ready
            and unmet is not None
            and not isinstance(unmet, (list, tuple))
        ):
            raise CrawlManifestWriteError(
                "output_readiness.unmet_requirements must be a list"
            )
        readiness_total = self._strict_nonnegative_count(
            readiness.get("object_records_total"),
            field="output_readiness.object_records_total",
        )
        if readiness_total != object_records_total:
            raise CrawlManifestWriteError(
                "output readiness object count does not match valid current objects"
            )
        readiness_modalities = self._strict_modality_counts(
            readiness.get("modality_counts"),
            field="output_readiness.modality_counts",
        )
        if readiness_modalities != modality_counts:
            raise CrawlManifestWriteError(
                "output readiness modalities do not match valid current objects"
            )
        successful_requests = self._strict_nonnegative_count(
            readiness.get("successful_requests_total"),
            field="output_readiness.successful_requests_total",
        )
        quality_score = readiness.get("quality_score")
        if isinstance(quality_score, bool) or not isinstance(
            quality_score, (int, float)
        ):
            raise CrawlManifestWriteError(
                "output_readiness.quality_score must be numeric"
            )
        if not ready:
            return
        if not self._crawl_output_gate.enabled:
            return
        if (
            object_records_total
            < self._crawl_output_gate.min_raw_objects_total
        ):
            raise CrawlManifestWriteError("raw object minimum is not met")
        if (
            successful_requests
            < self._crawl_output_gate.min_successful_requests_total
        ):
            raise CrawlManifestWriteError(
                "successful request minimum is not met"
            )
        if float(quality_score) < self._crawl_output_gate.min_quality_score:
            raise CrawlManifestWriteError("quality minimum is not met")
        for modality, minimum in self._minimum_modality_counts().items():
            if modality_counts.get(modality, 0) < minimum:
                raise CrawlManifestWriteError(
                    f"minimum {modality} count is not met"
                )

    def _minimum_modality_counts(self) -> dict[str, int]:
        minimum = self._crawl_output_gate.minimum_records
        return {
            "page": minimum.page,
            "document": minimum.document,
            "image": minimum.image,
            "audio": minimum.audio,
            "video": minimum.video,
        }

    @staticmethod
    def _normalized_modality_counts(
        modality_counts: dict[str, int],
    ) -> dict[str, int]:
        return {
            modality: count
            for modality, count in sorted(modality_counts.items())
            if count > 0
        }

    @staticmethod
    def _strict_nonnegative_count(value: object, *, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CrawlManifestWriteError(
                f"{field} must be a non-negative integer"
            )
        return value

    def _strict_modality_counts(
        self,
        value: object,
        *,
        field: str,
    ) -> dict[str, int]:
        if not isinstance(value, dict):
            raise CrawlManifestWriteError(f"{field} must be an object")
        allowed_modalities = {"page", "document", "image", "audio", "video"}
        if any(
            not isinstance(modality, str) or modality not in allowed_modalities
            for modality in value
        ):
            raise CrawlManifestWriteError(
                f"{field} contains an unsupported modality"
            )
        return self._normalized_modality_counts(
            {
                modality: self._strict_nonnegative_count(
                    count,
                    field=f"{field}.{modality}",
                )
                for modality, count in value.items()
            }
        )

    @staticmethod
    def _require_summary_path(
        *,
        payload: dict[str, object],
        field: str,
        expected: Path,
    ) -> None:
        actual = payload.get(field)
        if not isinstance(actual, str) or not actual.strip():
            raise CrawlManifestWriteError(
                f"raw run summary lacks {field} identity"
            )
        if Path(actual).resolve() != expected.resolve():
            raise CrawlManifestWriteError(
                f"raw run summary identity mismatch for {field}"
            )

    def _read_summary_payload(
        self, *, summary_path: Path
    ) -> dict[str, object]:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CrawlManifestWriteError(
                f"cannot parse raw run summary: {type(exc).__name__}"
            ) from exc
        if not isinstance(payload, dict):
            raise CrawlManifestWriteError("raw run summary must be an object")
        return payload

    @staticmethod
    def _read_pipeline_counters(
        *,
        summary_path: Path | None,
        logger: ProjectLogger,
    ) -> dict[str, object]:
        if summary_path is None or not summary_path.exists():
            return {}

        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            if logger:
                logger.warning(
                    "pipeline_counters_parse_failed",
                    path=format_manifest_path(summary_path),
                    error_type=type(exc).__name__,
                )
            return {}

        if not isinstance(payload, dict):
            return {}

        counters: dict[str, object] = {}
        for key in PIPELINE_COUNTER_KEYS:
            value = payload.get(key)
            if value is not None:
                counters[key] = value

        modality_counts = payload.get("modality_counts")
        if isinstance(modality_counts, dict):
            counters["modality_counts"] = modality_counts

        return counters

    def _read_existing_manifest(
        self,
        *,
        path: Path,
        quarantined_at: str,
    ) -> CrawlManifest | None:
        if not path.exists():
            return None

        try:
            serialized = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise CrawlManifestWriteError(
                f"cannot read existing crawl manifest {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        except UnicodeError as exc:
            self._raise_corrupt_existing_manifest(
                path=path,
                quarantined_at=quarantined_at,
                cause=exc,
            )

        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as exc:
            self._raise_corrupt_existing_manifest(
                path=path,
                quarantined_at=quarantined_at,
                cause=exc,
            )

        if not isinstance(payload, dict):
            self._raise_corrupt_existing_manifest(
                path=path,
                quarantined_at=quarantined_at,
                cause=TypeError("crawl manifest root must be a JSON object"),
            )

        try:
            return CrawlManifest.from_payload(payload)
        except (TypeError, ValueError) as exc:
            self._raise_corrupt_existing_manifest(
                path=path,
                quarantined_at=quarantined_at,
                cause=exc,
            )

    def _raise_corrupt_existing_manifest(
        self,
        *,
        path: Path,
        quarantined_at: str,
        cause: BaseException,
    ) -> NoReturn:
        quarantine_path: Path | None = None
        quarantine_error: str | None = None

        try:
            quarantine_path = self._quarantine_corrupt_manifest(
                path=path,
                quarantined_at=quarantined_at,
            )
        except OSError as exc:
            quarantine_error = f"{type(exc).__name__}: {exc}"

        self._logger.error(
            "workflow_crawl_manifest_corrupt",
            recovery_rules="quarantine_and_abort",
            manifest_path=format_manifest_path(path),
            quarantine_path=(
                format_manifest_path(quarantine_path)
                if quarantine_path is not None
                else None
            ),
            corruption_type=type(cause).__name__,
            corruption_message=str(cause) or None,
            quarantine_error=quarantine_error,
        )

        raise CorruptCrawlManifestError(
            manifest_path=path,
            quarantine_path=quarantine_path,
            corruption_type=type(cause).__name__,
            corruption_message=str(cause),
            quarantine_error=quarantine_error,
        ) from cause

    @staticmethod
    def _quarantine_corrupt_manifest(
        *,
        path: Path,
        quarantined_at: str,
    ) -> Path:
        quarantine_directory = path.parent / ".quarantine"
        quarantine_directory.mkdir(parents=True, exist_ok=True)

        timestamp = (
            "".join(
                character
                for character in quarantined_at
                if character.isalnum()
            )[:24]
            or "unknown-time"
        )
        quarantine_path = quarantine_directory / (
            f"{path.name}.{timestamp}.{uuid4().hex[:12]}.corrupt"
        )

        path.replace(quarantine_path)
        ManifestFileWriter._fsync_directory(quarantine_directory)
        ManifestFileWriter._fsync_directory(path.parent)
        return quarantine_path

    @staticmethod
    def _raw_crawl_output_missing(*, inventory: RawInventory) -> bool:
        return (
            inventory.directory is None
            or inventory.summary_path is None
            or inventory.records_path is None
            or inventory.errors_path is None
            or inventory.fingerprint is None
            or not inventory.schema_valid
        )
