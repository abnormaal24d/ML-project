"""Processing-activity registry wiring into governance record assembly."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from crawler.governance.processing_activity import (
    PROCESSING_ACTIVITIES_SETTING,
    ProcessingActivityConfigError,
    ProcessingActivityRegistry,
    load_processing_activities,
)
from crawler.storage.datasets.records.governance import (
    processing_activity_permission,
)
from logger.project_logger import ProjectLogger
from orchestration.settings_loader import (
    load,
    validate_runtime_configuration,
)

_NOW = datetime(2026, 8, 9, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _registry_payload(*, dpia_review_expires_at: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "activities": [
            {
                "activity_id": "training_dataset_build",
                "purpose": "Build governed multimodal training datasets",
                "personal_data_allowed": False,
                "dpia_status": "approved",
                "dpia_review_expires_at": dpia_review_expires_at,
                "retention_days": 365,
                "rules_version": "1.0.0",
                "enabled": True,
            }
        ],
    }


def test_load_processing_activities_from_json(tmp_path: Path) -> None:
    path = tmp_path / "activities.json"
    path.write_text(
        __import__("json").dumps(
            _registry_payload(dpia_review_expires_at="2099-01-01T00:00:00Z")
        ),
        encoding="utf-8",
    )
    registry = load_processing_activities(path)
    assert isinstance(registry, ProcessingActivityRegistry)
    assert registry.require(activity_id="training_dataset_build").enabled


def test_processing_activity_permission_fail_closed_for_unknown() -> None:
    registry = ProcessingActivityRegistry.model_validate(
        _registry_payload(dpia_review_expires_at="2099-01-01T00:00:00Z")
    )
    assert processing_activity_permission(
        registry=registry,
        activity_id="missing_activity",
        now=_NOW,
    ) == (False, False)


def test_processing_activity_permission_reflects_expiry() -> None:
    registry = ProcessingActivityRegistry.model_validate(
        _registry_payload(dpia_review_expires_at="2020-01-01T00:00:00Z")
    )
    assert processing_activity_permission(
        registry=registry,
        activity_id="training_dataset_build",
        now=_NOW,
    ) == (False, True)


def test_processing_activity_permission_allows_when_current() -> None:
    registry = ProcessingActivityRegistry.model_validate(
        _registry_payload(dpia_review_expires_at="2099-01-01T00:00:00Z")
    )
    assert processing_activity_permission(
        registry=registry,
        activity_id="training_dataset_build",
        now=_NOW,
    ) == (True, True)


def test_processing_activity_permission_fails_closed_without_registry() -> (
    None
):
    assert processing_activity_permission(
        registry=None,
        activity_id=None,
        now=_NOW,
    ) == (False, False)


def test_runtime_preflight_loads_packaged_processing_activities(
    tmp_path: Path,
) -> None:
    settings = load(
        project_root=tmp_path,
        config_root=PROJECT_ROOT,
        environment="dev",
        profile="dev",
    )
    readiness = validate_runtime_configuration(
        settings=settings,
        config_root=PROJECT_ROOT,
    )
    registry = readiness.processing_activity_registry
    now = datetime.now(timezone.utc)
    activity = registry.require(activity_id="training_dataset_build")
    assert activity.dpia_status == "approved"
    assert activity.permits_training(now=now)


def test_missing_processing_activity_config_has_safe_context(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path / "secret-customer-directory" / "processing_activities.json"
    )

    with pytest.raises(ProcessingActivityConfigError) as raised:
        load_processing_activities(path)

    error = raised.value

    assert error.code == "processing_activity_config_missing"
    assert error.component == "processing_activity_registry"
    assert error.setting == PROCESSING_ACTIVITIES_SETTING
    assert error.setting == "governance.processing_activities_file"
    assert error.basename == "processing_activities.json"

    assert str(tmp_path) not in str(error)
    assert "secret-customer-directory" not in str(error)


def test_processing_activity_config_rejects_directory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "processing_activities.json"
    path.mkdir()

    with pytest.raises(
        ProcessingActivityConfigError,
        match="processing_activity_config_not_a_file",
    ):
        load_processing_activities(path)


def test_processing_activity_config_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "processing_activities.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(
        ProcessingActivityConfigError,
        match="processing_activity_config_invalid_json",
    ):
        load_processing_activities(path)


def test_processing_activity_config_rejects_invalid_encoding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "processing_activities.json"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(
        ProcessingActivityConfigError,
        match="processing_activity_config_invalid_encoding",
    ):
        load_processing_activities(path)


def test_processing_activity_config_rejects_invalid_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "processing_activities.json"
    path.write_text(
        '{"activities": [{"enabled": true}]}',
        encoding="utf-8",
    )

    with pytest.raises(
        ProcessingActivityConfigError,
        match="processing_activity_config_invalid_schema",
    ):
        load_processing_activities(path)


def test_processing_activity_config_log_keeps_safe_fields(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    logger = ProjectLogger(logging.getLogger("test_processing_activity"))

    logger.error(
        "startup_configuration_error",
        component="processing_activity_registry",
        setting=PROCESSING_ACTIVITIES_SETTING,
        basename="processing_activities.json",
        issue="processing_activity_config_missing",
        required=True,
        path=str(tmp_path / "governance" / "processing_activities.json"),
    )

    records = [
        record
        for record in caplog.records
        if record.name == "test_processing_activity"
    ]
    assert records
    record = records[-1]

    assert record.component == "processing_activity_registry"
    assert record.setting == PROCESSING_ACTIVITIES_SETTING
    assert record.basename == "processing_activities.json"
    assert record.issue == "processing_activity_config_missing"
    assert record.required is True
    assert record.path == "[LOCAL_PATH]"
