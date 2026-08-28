from __future__ import annotations

from pathlib import Path

import pytest

from config.load import load_settings
from config.releases.release_requirements import (
    ReleaseConfigurationError,
    release_requirements_from_settings,
    validate_release_requirements,
)
from orchestration.settings_loader import load

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _production_transcription_pins(production_whisper_env: None) -> None:
    """Resolve production settings with explicit test-only model pins."""


def _isolated_config_root(tmp_path: Path) -> Path:
    import shutil

    config_root = tmp_path / "config-root"
    shutil.copytree(
        PROJECT_ROOT / "config/files",
        config_root / "config/files",
    )
    shutil.copytree(
        PROJECT_ROOT / "config/profiles",
        config_root / "config/profiles",
    )
    return config_root


def test_production_contract_is_loaded_and_validated() -> None:
    settings = load(
        project_root=PROJECT_ROOT,
        config_root=PROJECT_ROOT,
        environment="prod",
        profile="prod",
    )

    assert settings.training.release_stage == "candidate"
    assert "image" in settings.multimodal.enabled_modalities
    assert settings.training.tasks == tuple(
        task.name for task in settings.release.tasks
    )
    assert len(settings.training.tasks) == 14
    assert settings.training.task_aware_batching is True
    assert settings.training.drop_last is True
    assert settings.training.dynamic_sampling is True
    assert (
        settings.preprocessing.audio_validation.require_transcript_for_audio_text_pair
        is True
    )
    assert all(
        task.min_samples > 0 and task.metrics
        for task in settings.release.tasks
    )
    assert settings.release.limits.max_batch_latency_ms == 250
    assert settings.release.limits.max_peak_memory_mb == 16384

    contract = release_requirements_from_settings(settings)
    assert contract is not None
    assert contract.runtime_limits is not None
    assert contract.runtime_limits.max_batch_latency_ms == 250
    assert contract.runtime_limits.max_peak_memory_mb == 16384


def test_required_task_drift_fails_closed() -> None:
    settings = load_settings(
        profile="prod",
        project_root=PROJECT_ROOT,
        config_root=PROJECT_ROOT,
        environment="prod",
    )
    contract = release_requirements_from_settings(settings)
    assert contract is not None

    with pytest.raises(
        ReleaseConfigurationError,
        match="Enabled tasks are outside release scope",
    ):
        validate_release_requirements(
            release_requirements=contract,
            enabled_modalities=settings.multimodal.enabled_modalities,
            enabled_tasks=(*settings.training.tasks, "text_to_image"),
            active_release_stage=settings.training.release_stage,
        )


def test_release_stage_drift_fails_closed() -> None:
    settings = load_settings(
        profile="prod",
        project_root=PROJECT_ROOT,
        config_root=PROJECT_ROOT,
        environment="prod",
    )
    contract = release_requirements_from_settings(settings)
    assert contract is not None

    with pytest.raises(
        ReleaseConfigurationError,
        match="active release stage must exactly match",
    ):
        validate_release_requirements(
            release_requirements=contract,
            enabled_modalities=settings.multimodal.enabled_modalities,
            enabled_tasks=settings.training.tasks,
            active_release_stage="production_model",
        )
