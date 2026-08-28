from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from config.settings.logging import EventRateLimitRulesSettings, LoggingSettings
from config.validation.cross_section import basic as basic_validators
from crawler.storage.datasets.writing.dataset_run_finalizer import (
    DatasetRunFinalizer,
)
from crawler.storage.datasets.writing.dataset_snapshot_builder import (
    DatasetSnapshotBuilder,
)
from mmcrawler_datasets.record_components import parsing as record_parsing
from mmcrawler_datasets.training_samples import models as sample_models
from mmcrawler_datasets.training_samples.artifact_path import (
    ValidatedArtifactPath,
)
from preprocessing.media.speech import prosody_contracts
from preprocessing.media.speech.prosody_contracts import (
    ProsodyFeatures,
    ProsodyStatus,
    parse_prosody,
)
from preprocessing.privacy.clearance import ApprovedObjectRole


def test_cross_section_validators_are_imported_from_domain_modules() -> None:
    assert callable(basic_validators._validate_dataset_splits)


def test_record_parser_is_imported_from_component_module() -> None:
    assert callable(record_parsing.parse_record)
    assert callable(record_parsing.require_training_record)


def test_training_object_requires_typed_path_and_role(tmp_path: Path) -> None:
    artifact = tmp_path / "objects" / "image.jpg"
    artifact.parent.mkdir()
    artifact.write_bytes(b"image")
    validated = ValidatedArtifactPath(
        relative_path="objects/image.jpg",
        resolved_path=artifact,
        project_root=tmp_path,
    )
    training_object = sample_models.TrainingObject(
        object_id="image-1",
        object_path=validated,
        object_sha256=__import__("hashlib").sha256(b"image").hexdigest(),
        object_mime_type="image/jpeg",
        role=ApprovedObjectRole.PRIMARY_MEDIA,
    )
    sample = sample_models.TrainingSample(
        sample_id="sample-1",
        objects=(training_object,),
    )

    payload = sample.to_dict()

    assert payload["objects"][0]["object_path"] == "objects/image.jpg"  # type: ignore[index]
    assert payload["objects"][0]["role"] == "primary_media"  # type: ignore[index]


def test_prosody_contracts_are_directly_importable() -> None:
    assert ProsodyFeatures is prosody_contracts.ProsodyFeatures
    parsed = parse_prosody({"energy": 0.25})
    assert parsed == ProsodyFeatures(energy=0.25)
    assert ProsodyStatus.AVAILABLE.value == "available"


def test_logging_defaults_are_owned_by_logging_settings() -> None:
    defaults = LoggingSettings()
    override = EventRateLimitRulesSettings(
        min_interval_sec=11.0,
        field_names=("component_path", "host"),
    )
    configured = defaults.model_copy(
        update={
            "event_rate_limit_governance": {
                **defaults.event_rate_limit_governance,
                "rate_limiter_sleep": override,
            }
        }
    )

    assert len(defaults.event_rate_limit_governance) == 18
    assert configured.event_rate_limit_governance["rate_limiter_sleep"] is override
    assert "autoscaler_tick" in configured.event_rate_limit_governance


class _RecordIndex:
    def __init__(self, records: list[object]) -> None:
        self._records = records

    def records(self) -> tuple[object, ...]:
        return tuple(self._records)

    def latest_records(self) -> tuple[object, ...]:
        return self.records()


class _SyncUpdater:
    relationships_count = 2
    metadata_count = 3
    updates_count = 4


def test_dataset_snapshot_builder_projects_only_payload_backed_records(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "objects" / "page.html"
    payload.parent.mkdir()
    payload.write_text("<html></html>", encoding="utf-8")
    valid = SimpleNamespace(
        media_identity="page-1",
        stable_url_id=None,
        normalized_url=None,
        object_id="object-1",
        storage_relative_path="objects/page.html",
        modality="page",
        mime_type="text/html",
        content_type=None,
    )
    missing = SimpleNamespace(
        media_identity="page-2",
        stable_url_id=None,
        normalized_url=None,
        object_id="object-2",
        storage_relative_path="objects/missing.html",
        modality="page",
        mime_type="text/html",
        content_type=None,
    )
    builder = DatasetSnapshotBuilder(
        run_id="run-1",
        run_directory=tmp_path,
        record_index=_RecordIndex([valid, missing]),  # type: ignore[arg-type]
        sync_updater=_SyncUpdater(),  # type: ignore[arg-type]
    )

    snapshot = builder.build(total_bytes_written=123)

    assert snapshot.run_id == "run-1"
    assert snapshot.object_records_total == 1
    assert snapshot.modality_counts == (("page", 1),)
    assert snapshot.total_bytes_written == 123


class _Logger:
    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


def test_dataset_run_lifecycle_projects_terminal_contract() -> None:
    """Test that the finalizer correctly projects terminal contract."""

    class _MockManifestWriter:
        def __init__(self) -> None:
            self.write_count = 0
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _MockCompactor:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def refresh_summary(self, **kwargs: object) -> Path:
            self.calls.append(dict(kwargs))
            return Path("/tmp/test-run_run_manifest.json")

        def close(self) -> None:
            pass

    manifest_writer = _MockManifestWriter()
    sync_compactor = _MockCompactor()
    logger = _Logger()

    run_finalizer = DatasetRunFinalizer(
        logger=logger,
        now=lambda: datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        run_id="test-run",
        run_directory=Path("/tmp/test"),
        manifest_path=Path("/tmp/test/manifest.jsonl"),
        manifest_writer=manifest_writer,
        sync_compactor=sync_compactor,
    )

    completed_at = run_finalizer.finalize(
        total_bytes_written=42,
        status="failed",
        final=True,
        readiness_report=None,
        terminal_reason="boom",
        terminal_details={"stage": "write"},
    )

    assert completed_at == "2026-07-26T12:00:00+00:00"
    assert manifest_writer.write_count == 0
    assert manifest_writer.closed is True

    assert len(sync_compactor.calls) == 1
    assert sync_compactor.calls[0]["status"] == "failed"
    assert sync_compactor.calls[0]["terminal_reason"] == "boom"
