from __future__ import annotations

from pathlib import Path

import pytest

from config.releases.release_requirements import (
    ReleaseConfigurationError,
    release_requirements_from_settings,
    validate_release_requirements,
)
from release.task_contract_validation import validate_release_task_contracts

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _candidate_transcription_pins(production_whisper_env: None) -> None:
    """Resolve candidate settings with explicit test-only model pins."""


def _production_settings():
    from config.load import load_settings

    return load_settings(
        "prod",
        project_root=PROJECT_ROOT,
        config_root=PROJECT_ROOT,
        environment="prod",
    )


def test_image_is_required_for_production_v1() -> None:
    settings = _production_settings()
    contract = release_requirements_from_settings(settings)

    assert "image" in contract.required_modalities
    assert "image" not in contract.optional_modalities
    assert "image_text_pair" in contract.required_tasks
    assert "multimodal_retrieval" in contract.required_tasks


def test_required_image_cannot_be_disabled() -> None:
    settings = _production_settings()
    contract = release_requirements_from_settings(settings)

    with pytest.raises(
        ReleaseConfigurationError,
        match="Required modalities are disabled",
    ):
        validate_release_requirements(
            release_requirements=contract,
            enabled_modalities=("text", "document"),
            enabled_tasks=contract.required_tasks,
            active_release_stage=contract.release_stage,
        )


def test_required_retrieval_task_cannot_be_disabled() -> None:
    settings = _production_settings()
    contract = release_requirements_from_settings(settings)
    enabled_tasks = tuple(
        task
        for task in contract.required_tasks
        if task != "multimodal_retrieval"
    )

    with pytest.raises(
        ReleaseConfigurationError,
        match="Required tasks are disabled",
    ):
        validate_release_requirements(
            release_requirements=contract,
            enabled_modalities=contract.required_modalities,
            enabled_tasks=enabled_tasks,
            active_release_stage=contract.release_stage,
        )


def test_normal_required_scope_passes() -> None:
    settings = _production_settings()
    contract = release_requirements_from_settings(settings)

    validate_release_requirements(
        release_requirements=contract,
        enabled_modalities=contract.required_modalities,
        enabled_tasks=contract.required_tasks,
        active_release_stage=contract.release_stage,
    )


def test_required_audio_task_without_sample_builder_is_rejected(
    monkeypatch,
) -> None:
    """Audio task (audio_qa) is required but lacks a sample builder."""
    # Provide evidence for all tasks EXCEPT audio_qa builder
    required_without_audio_qa = {
        "text_pretrain",
        "instruction_following",
        "document_text_pair",
        "pdf_text_pair",
        "ocr_parse",
        "doc_qa",
        "image_text_pair",
        "vqa",
        "audio_text_pair",
        "video_text_pair",
        "video_qa",
        "multimodal_retrieval",
        "cross_modal_consistency",
    }

    # Patch evidence registries through the registry module globals so the
    # immutable frozenset imports in release.task_contract_validation are replaced.
    monkeypatch.setattr(
        "release.task_contract_validation.SAMPLE_BUILDER_TASKS",
        frozenset(required_without_audio_qa),
    )

    settings = _production_settings()
    contract = release_requirements_from_settings(settings)

    with pytest.raises(ValueError, match=r"audio_qa.*sample_builder"):
        validate_release_task_contracts(settings, contract)
