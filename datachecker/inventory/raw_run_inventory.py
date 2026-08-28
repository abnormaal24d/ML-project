"""Raw crawl artifact inventory reader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from pydantic import ValidationError

from schemas.raw_dataset import RawDatasetRecord
from schemas.raw_payload_evidence import (
    raw_payload_evidence_matches as _raw_payload_evidence_matches,
)

DeadlineCheckpoint = Callable[[str], None]


def no_deadline_checkpoint(stage: str) -> None:
    pass


if TYPE_CHECKING:
    from config.collection.training_input_gate import DataCheckerSettings
    from config.path_resolution.workflow_artifact_paths import (
        ArtifactPathRegistry,
    )
    from datachecker.fingerprints import DatasetFingerprintCalculator


@dataclass(slots=True, frozen=True)
class RawInventory:
    """Resolved raw crawl artifacts."""

    directory: Path | None
    summary_path: Path | None
    records_path: Path | None
    errors_path: Path | None
    fingerprint: str | None
    file_count: int
    fetched_url_count: int
    failed_url_count: int
    modality_counts: dict[str, int]
    started_at: str | None
    completed_at: str | None
    status: str | None
    final: bool
    schema_valid: bool
    raw_schema_errors: tuple[str, ...] = ()
    objects_found_on_disk: int = 0
    objects_registered_in_jsonl: int = 0
    latest_attempt_raw_run_directory: Path | None = None
    # P0.7: best-effort count of unique valid objects (prefers current_objects.jsonl + dedup)
    unique_valid_object_count: int = 0


@dataclass(slots=True, frozen=True)
class ValidCurrentRecords:
    """The one deduplicated, payload-backed view of a raw run."""

    records: tuple[dict[str, object], ...]
    modality_counts: dict[str, int]
    errors: tuple[str, ...]

    @property
    def valid_count(self) -> int:
        return len(self.records)


class RawInventoryReader:
    """Resolve the latest completed raw crawl run."""

    def __init__(
        self,
        *,
        settings: DataCheckerSettings,
        artifact_path_registry: ArtifactPathRegistry,
        dataset_fingerprint_calculator: DatasetFingerprintCalculator,
        raw_schema_version: str,
    ) -> None:
        self._settings = settings
        self._artifact_path_registry = artifact_path_registry
        self._dataset_fingerprint_calculator = dataset_fingerprint_calculator
        self._raw_schema_version = raw_schema_version

    def read(
        self,
        *,
        raw_run_directory: Path | None = None,
        run_summary_path: Path | None = None,
        checkpoint: DeadlineCheckpoint = no_deadline_checkpoint,
    ) -> RawInventory:
        """Build inventory for the raw run explicitly bound to this attempt."""
        dataset_paths = self._artifact_path_registry.dataset_paths
        raw_sync_directory = Path(dataset_paths.raw_sync_directory)
        records_relative_path = Path(dataset_paths.manifest_filename)
        errors_relative_path = (
            raw_sync_directory / dataset_paths.raw_sync_errors_filename
        )
        run_directory = raw_run_directory
        summary_path = run_summary_path
        records_path = (
            None
            if run_directory is None
            else run_directory / records_relative_path
        )
        errors_path = (
            None
            if run_directory is None
            else run_directory / errors_relative_path
        )

        # Compute canonical paths from config
        expected_summary_path = (
            None
            if run_directory is None
            else run_directory
            / raw_sync_directory
            / dataset_paths.raw_sync_summary_filename
        )
        expected_manifest_relative_path = records_relative_path.as_posix()

        payload = {}
        if summary_path and summary_path.is_file():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass

        files: tuple[Path, ...] = ()
        if run_directory and run_directory.is_dir():
            found_files = []
            for i, p in enumerate(run_directory.rglob("*")):
                if i % 256 == 0:
                    checkpoint("raw_inventory_file_scan")
                if p.is_file():
                    found_files.append(p)
            files = tuple(found_files)

        current_objects_path = (
            None
            if run_directory is None
            else run_directory
            / dataset_paths.raw_sync_directory
            / dataset_paths.raw_sync_current_objects_filename
        )
        current_records = self.valid_current_records(
            run_directory=run_directory,
            current_objects_path=current_objects_path,
            expected_schema_version=self._raw_schema_version,
            expected_run_id=run_directory.name if run_directory else "",
            checkpoint=checkpoint,
        )
        schema_valid, schema_errors = self._raw_run_schema_valid(
            run_directory=run_directory,
            summary_path=summary_path,
            records_path=records_path,
            errors_path=errors_path,
            current_records=current_records,
            expected_summary_path=expected_summary_path,
            expected_manifest_relative_path=expected_manifest_relative_path,
            expected_schema_version=self._raw_schema_version,
            expected_run_id=run_directory.name if run_directory else "",
            checkpoint=checkpoint,
        )

        unique_valid = current_records.valid_count

        # latest attempt for debug (may differ from chosen directory)
        status = str(payload.get("status")) if payload.get("status") else None
        is_final = payload.get("final") is True

        return RawInventory(
            directory=run_directory,
            summary_path=summary_path,
            records_path=records_path,
            errors_path=errors_path,
            fingerprint=self._calculate_fingerprint(
                paths=files,
                root=run_directory,
                checkpoint=checkpoint,
            ),
            file_count=len(files),
            fetched_url_count=unique_valid,
            failed_url_count=RawInventoryReader._summary_count(
                payload.get("failed_url_count")
            ),
            modality_counts=current_records.modality_counts,
            started_at=str(payload.get("started_at"))
            if payload.get("started_at")
            else None,
            completed_at=str(payload.get("completed_at"))
            if payload.get("completed_at")
            else None,
            status=status,
            final=is_final,
            schema_valid=schema_valid,
            raw_schema_errors=schema_errors,
            objects_found_on_disk=unique_valid,
            objects_registered_in_jsonl=unique_valid,
            latest_attempt_raw_run_directory=run_directory,
            unique_valid_object_count=unique_valid,
        )

    def _calculate_fingerprint(
        self,
        *,
        paths: tuple[Path, ...],
        root: Path | None,
        checkpoint: DeadlineCheckpoint,
    ) -> str | None:
        if not paths:
            return None
        return self._dataset_fingerprint_calculator.calculate(
            paths=paths,
            root=root,
            checkpoint=checkpoint,
        )

    @staticmethod
    def valid_current_records(
        *,
        run_directory: Path | None,
        current_objects_path: Path | None,
        expected_schema_version: str,
        expected_run_id: str,
        checkpoint: DeadlineCheckpoint = no_deadline_checkpoint,
    ) -> ValidCurrentRecords:
        """Read only current records and fail closed on invalid payload evidence."""
        if run_directory is None or current_objects_path is None:
            return ValidCurrentRecords((), {}, ("current_objects_missing",))
        if not current_objects_path.is_file():
            return ValidCurrentRecords((), {}, ("current_objects_missing",))

        latest: dict[str, dict[str, object]] = {}
        errors: list[str] = []
        try:
            with current_objects_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line_number % 256 == 0:
                        checkpoint("raw_current_objects_scan")
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        errors.append(
                            f"current_record_json_invalid:{line_number}"
                        )
                        continue
                    if not isinstance(payload, dict):
                        errors.append(
                            f"current_record_not_object:{line_number}"
                        )
                        continue
                    # Validate with RawDatasetRecord schema
                    try:
                        record = RawDatasetRecord.model_validate(payload)
                    except ValidationError:
                        errors.append(
                            f"current_record_schema_invalid:{line_number}"
                        )
                        continue
                    if record.schema_version != expected_schema_version:
                        errors.append(
                            f"current_record_schema_version_mismatch:{line_number}"
                        )
                        continue
                    if record.run_id != expected_run_id:
                        errors.append(
                            f"current_record_run_id_mismatch:{line_number}"
                        )
                        continue
                    key = RawInventoryReader._record_identity(payload)
                    if key is None:
                        errors.append(
                            f"current_record_identity_missing:{line_number}"
                        )
                        continue
                    if key in latest:
                        errors.append(
                            f"current_record_duplicate_identity:{key}"
                        )
                        continue
                    latest[key] = payload
        except (OSError, UnicodeError) as exc:
            return ValidCurrentRecords(
                (), {}, (f"current_records_read_failed:{type(exc).__name__}",)
            )

        records: list[dict[str, object]] = []
        counts: dict[str, int] = {}
        for record_index, (identity, payload) in enumerate(
            latest.items(),
            start=1,
        ):
            if record_index % 256 == 0:
                checkpoint("raw_current_records_validation")
            modality, error = RawInventoryReader._validated_record_modality(
                run_directory=run_directory,
                payload=payload,
            )
            if error is not None:
                errors.append(f"{error}:{identity}")
                continue
            if modality is None:
                raise RuntimeError(
                    "validated modality is unexpectedly missing"
                )
            records.append(payload)
            counts[modality] = counts.get(modality, 0) + 1

        return ValidCurrentRecords(
            records=tuple(records),
            modality_counts=dict(sorted(counts.items())),
            errors=tuple(errors),
        )

    @staticmethod
    def _record_identity(payload: dict[str, object]) -> str | None:
        value = payload.get("stable_url_id")
        return (
            value.strip() if isinstance(value, str) and value.strip() else None
        )

    @staticmethod
    def _validated_record_modality(
        *,
        run_directory: Path,
        payload: dict[str, object],
    ) -> tuple[str | None, str | None]:
        storage_path = payload.get("storage_relative_path")
        if not isinstance(storage_path, str) or not storage_path.strip():
            return None, "payload_path_missing"
        candidate = (run_directory / storage_path).resolve()
        try:
            candidate.relative_to(run_directory.resolve())
        except ValueError:
            return None, "payload_path_escapes_run"
        if not candidate.is_file():
            return None, "payload_file_missing"

        raw_modality = payload.get("modality")
        if not isinstance(raw_modality, str):
            return None, "modality_missing"
        modality = raw_modality.strip().lower()
        if modality == "feed":
            modality = "page"
        if modality not in {"page", "document", "image", "audio", "video"}:
            return None, "modality_unknown"

        raw_mime = payload.get("mime_type")
        mime = (
            raw_mime.strip().lower()
            if isinstance(raw_mime, str) and raw_mime.strip()
            else None
        )
        suffix = candidate.suffix.lower()

        if not _raw_payload_evidence_matches(
            modality=modality,
            mime_type=mime,
            suffix=suffix,
        ):
            return None, f"mime_modality_mismatch:{modality}"

        # Validate payload byte size and SHA-256
        byte_size = payload.get("byte_size")
        content_sha256 = payload.get("content_sha256")
        if (
            isinstance(byte_size, int)
            and candidate.stat().st_size != byte_size
        ):
            return None, "payload_size_mismatch"
        if isinstance(content_sha256, str) and content_sha256.strip():
            digest = hashlib.sha256()
            try:
                with candidate.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != content_sha256.lower():
                    return None, "payload_sha256_mismatch"
            except OSError:
                return None, "payload_sha256_read_failed"

        return modality, None

    @staticmethod
    def _validate_manifest_history(
        *,
        records_path: Path,
        expected_schema_version: str,
        expected_run_id: str,
        checkpoint: DeadlineCheckpoint,
    ) -> tuple[set[str], int, tuple[str, ...]]:
        """Validate objects.jsonl history and return fetch_record_ids, count, errors."""
        fetch_record_ids: set[str] = set()
        record_count = 0
        errors: list[str] = []

        try:
            with records_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line_number % 256 == 0:
                        checkpoint("raw_manifest_history_scan")
                    text = line.strip()
                    if not text:
                        continue

                    try:
                        payload = json.loads(text)
                        record = RawDatasetRecord.model_validate(payload)
                    except (json.JSONDecodeError, ValidationError):
                        errors.append(f"manifest_record_invalid:{line_number}")
                        continue

                    if record.schema_version != expected_schema_version:
                        errors.append(
                            f"manifest_record_schema_mismatch:{line_number}"
                        )
                        continue

                    if record.run_id != expected_run_id:
                        errors.append(
                            f"manifest_record_run_id_mismatch:{line_number}"
                        )
                        continue

                    if record.fetch_record_id in fetch_record_ids:
                        errors.append(
                            f"manifest_fetch_record_duplicate:{record.fetch_record_id}"
                        )
                        continue

                    fetch_record_ids.add(record.fetch_record_id)
                    record_count += 1
        except (OSError, UnicodeError) as exc:
            errors.append(f"manifest_history_read_failed:{type(exc).__name__}")

        return fetch_record_ids, record_count, tuple(errors)

    @staticmethod
    def _is_completed_final_run(*, payload: dict[str, object]) -> bool:
        return (
            str(payload.get("status")) == "completed"
            and payload.get("final") is True
        )

    @staticmethod
    def _summary_count(value: object) -> int:
        """Return a safe non-negative summary count for inventory display."""

        if isinstance(value, bool) or value is None:
            return 0
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return 0

    @staticmethod
    def _valid_summary_count(value: object) -> bool:
        """Accept only the integer forms the reader can represent safely."""

        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return value >= 0
        return isinstance(value, str) and value.strip().isdigit()

    @staticmethod
    def _normalized_modality_counts(
        value: object,
    ) -> dict[str, int] | None:
        """Canonicalize one summary/count mapping for exact reconciliation."""

        if not isinstance(value, dict):
            return None
        allowed = {"page", "document", "image", "audio", "video"}
        counts: dict[str, int] = {}
        for raw_modality, raw_count in value.items():
            if not isinstance(raw_modality, str):
                return None
            modality = raw_modality.strip().casefold()
            if modality == "feed":
                modality = "page"
            if (
                modality not in allowed
                or isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count < 0
            ):
                return None
            counts[modality] = counts.get(modality, 0) + raw_count
        return {
            modality: count
            for modality, count in sorted(counts.items())
            if count > 0
        }

    def _raw_run_schema_valid(
        self,
        *,
        run_directory: Path | None,
        summary_path: Path | None,
        records_path: Path | None,
        errors_path: Path | None,
        current_records: ValidCurrentRecords,
        expected_summary_path: Path | None,
        expected_manifest_relative_path: str,
        expected_schema_version: str,
        expected_run_id: str,
        checkpoint: DeadlineCheckpoint = no_deadline_checkpoint,
    ) -> tuple[bool, tuple[str, ...]]:
        errors: list[str] = []
        if run_directory is None or summary_path is None:
            errors.append("no_raw_run_found")
            return False, tuple(errors)
        if not run_directory.exists() or not run_directory.is_dir():
            errors.append("run_directory_not_dir")
            return False, tuple(errors)
        if not summary_path.exists() or not summary_path.is_file():
            errors.append("summary_path_missing")
            return False, tuple(errors)
        if records_path is None or not records_path.is_file():
            errors.append("missing_objects_jsonl")
        if errors_path is None or not errors_path.is_file():
            errors.append("missing_errors_jsonl")
        if errors:
            return False, tuple(errors)
        if records_path is None:
            return False, ("missing_objects_jsonl",)

        try:
            summary_payload = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return False, (f"summary_parse_failed:{type(exc).__name__}",)
        if not isinstance(summary_payload, dict):
            return False, ("summary_not_object",)

        # Schema version validation
        if summary_payload.get("schema_version") != expected_schema_version:
            errors.append(
                "raw_schema_version_mismatch:"
                f"{summary_payload.get('schema_version')}"
                f"!={expected_schema_version}"
            )

        # Lifecycle stage validation
        if summary_payload.get("lifecycle_stage") != "raw":
            errors.append("lifecycle_stage_invalid")

        # Raw run ID validation
        reported_run_id = summary_payload.get("raw_run_id")
        if (
            not isinstance(reported_run_id, str)
            or reported_run_id != run_directory.name
        ):
            errors.append(
                f"raw_run_id_mismatch:{reported_run_id}!={run_directory.name}"
            )

        # Canonical summary path validation
        if expected_summary_path is None:
            errors.append("expected_summary_path_missing")
        elif summary_path.resolve() != expected_summary_path.resolve():
            errors.append("summary_path_not_canonical")

        # Canonical manifest path validation
        reported_manifest_path = summary_payload.get("manifest_path")
        if reported_manifest_path != expected_manifest_relative_path:
            errors.append(
                "manifest_path_mismatch:"
                f"{reported_manifest_path}"
                f"!={expected_manifest_relative_path}"
            )

        if not self._is_completed_final_run(payload=summary_payload):
            status = (
                str(summary_payload.get("status"))
                if summary_payload.get("status")
                else None
            )
            is_final = summary_payload.get("final") is True
            errors.append(
                f"raw_run_found_but_not_final:status={status}:final={is_final}"
            )
            return False, tuple(errors)

        errors.extend(current_records.errors)
        if not current_records.records:
            errors.append("no_current_valid_records")

        observed_modality = self._normalized_modality_counts(
            current_records.modality_counts
        )
        reported_modality = self._normalized_modality_counts(
            summary_payload.get("modality_counts")
        )
        if observed_modality is None:
            errors.append("current_modality_counts_invalid")
            observed_modality = {}
        if reported_modality is None:
            errors.append("modality_counts_invalid")
            reported_modality = {}
        for k in sorted(set(observed_modality) | set(reported_modality)):
            observed = observed_modality.get(k, 0)
            reported = reported_modality.get(k, 0)
            if reported != observed:
                errors.append(f"modality_mismatch:{k}:{reported}!={observed}")

        reported_total = summary_payload.get("object_records_total")
        if (
            isinstance(reported_total, bool)
            or not isinstance(reported_total, int)
            or reported_total != current_records.valid_count
        ):
            errors.append(
                "object_records_total_mismatch:"
                f"{reported_total}!={current_records.valid_count}"
            )

        if not self._valid_summary_count(
            summary_payload.get("failed_url_count")
        ):
            errors.append("failed_url_count_invalid")

        # Validate required_records - fail closed
        required_records = summary_payload.get("required_records")
        if not isinstance(required_records, list):
            errors.append("required_records_invalid")
        else:
            run_root = run_directory.resolve()
            reported_required: set[str] = set()

            # Build expected required records from config
            dataset_paths = self._artifact_path_registry.dataset_paths
            sync_dir = Path(dataset_paths.raw_sync_directory)

            expected_required_records = {
                Path(dataset_paths.manifest_filename).as_posix(),
                (sync_dir / dataset_paths.raw_sync_errors_filename).as_posix(),
                (
                    sync_dir
                    / dataset_paths.raw_sync_discovered_assets_filename
                ).as_posix(),
                (
                    sync_dir / dataset_paths.raw_sync_rejected_assets_filename
                ).as_posix(),
                (
                    sync_dir / dataset_paths.raw_sync_current_objects_filename
                ).as_posix(),
                (
                    sync_dir / dataset_paths.raw_sync_metadata_filename
                ).as_posix(),
            }

            for record_index, item in enumerate(required_records, start=1):
                if record_index % 256 == 0:
                    checkpoint("raw_required_records_scan")
                relative_path = item.strip() if isinstance(item, str) else ""
                if not relative_path:
                    errors.append("empty_required_record")
                    continue
                if relative_path in reported_required:
                    errors.append(f"duplicate_required_record:{relative_path}")
                    continue
                reported_required.add(relative_path)
                candidate = (run_root / relative_path).resolve()
                try:
                    candidate.relative_to(run_root)
                except ValueError:
                    errors.append(
                        f"required_record_escapes_run:{relative_path}"
                    )
                    continue
                if not candidate.is_file():
                    errors.append(f"missing_required:{relative_path}")

            # Check for missing canonical required records
            for required in sorted(
                expected_required_records - reported_required
            ):
                errors.append(f"required_record_not_declared:{required}")

        # Validate manifest history (objects.jsonl)
        history_fetch_record_ids, history_record_count, history_errors = (
            self._validate_manifest_history(
                records_path=records_path,
                expected_schema_version=expected_schema_version,
                expected_run_id=expected_run_id,
                checkpoint=checkpoint,
            )
        )
        errors.extend(history_errors)

        # Reconcile manifest_write_count with history
        reported_write_count = summary_payload.get("manifest_write_count")
        if (
            isinstance(reported_write_count, bool)
            or not isinstance(reported_write_count, int)
            or reported_write_count < 0
        ):
            errors.append("manifest_write_count_invalid")
        elif reported_write_count != history_record_count:
            errors.append(
                "manifest_write_count_mismatch:"
                f"{reported_write_count}!={history_record_count}"
            )

        # Verify all current records exist in history
        for record in current_records.records:
            fetch_record_id = record.get("fetch_record_id")
            if (
                not isinstance(fetch_record_id, str)
                or fetch_record_id not in history_fetch_record_ids
            ):
                errors.append(
                    f"current_record_missing_from_history:{fetch_record_id}"
                )

        # Every recorded schema discrepancy is evidence that the finalized
        # inventory no longer reconciles with the durable run artifacts.
        return not errors, tuple(errors)
