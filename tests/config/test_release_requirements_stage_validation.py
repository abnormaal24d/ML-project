from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from config.releases.release_requirements import (
    ReleaseConfigurationError,
    ReleaseRequirements,
    validate_release_requirements,
)
from release.task_contract_validation import validate_release_task_contracts

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _production_transcription_pins(production_whisper_env: None) -> None:
    """Resolve production settings with explicit test-only model pins."""


def _enabled_tasks() -> tuple[str, ...]:
    return (
        "text_pretrain",
        "instruction_following",
        "document_text_pair",
        "pdf_text_pair",
        "ocr_parse",
        "doc_qa",
        "image_text_pair",
        "vqa",
        "audio_text_pair",
        "audio_qa",
        "video_text_pair",
        "video_qa",
        "multimodal_retrieval",
        "cross_modal_consistency",
    )


def _enabled_modalities() -> tuple[str, ...]:
    return (
        "text",
        "document",
        "image",
        "audio",
        "video",
    )


def _production_settings():
    from config.load import load_settings

    return load_settings(
        "prod",
        project_root=PROJECT_ROOT,
        config_root=PROJECT_ROOT,
        environment="prod",
    )


def test_contract_rejects_runtime_stage_drift() -> None:
    contract = ReleaseRequirements(
        release_id="test",
        release_stage="production_model",
        required_modalities=(),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=(),
        optional_tasks=(),
        blocked_capabilities=(),
    )

    with pytest.raises(
        ReleaseConfigurationError,
        match="active release stage must exactly match",
    ):
        validate_release_requirements(
            release_requirements=contract,
            enabled_modalities=(),
            enabled_tasks=(),
            active_release_stage="candidate",
        )


def test_contract_accepts_matching_production_stage() -> None:
    contract = ReleaseRequirements(
        release_id="test",
        release_stage="production_model",
        required_modalities=(),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=(),
        optional_tasks=(),
        blocked_capabilities=(),
    )

    validate_release_requirements(
        release_requirements=contract,
        enabled_modalities=(),
        enabled_tasks=(),
        active_release_stage="production_model",
    )


def _minimal_contract_with_full_evidence() -> ReleaseRequirements:
    """Create a minimal contract with only tasks that have full implementation evidence."""
    return ReleaseRequirements(
        release_id="test_minimal",
        release_stage="production_model",
        required_modalities=("text",),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("text_pretrain",),
        optional_tasks=(),
        blocked_capabilities=(),
    )


def _minimal_contract_with_blocked() -> ReleaseRequirements:
    """Create a minimal contract with text tasks and a blocked capability."""
    from config.releases.release_requirements import BlockedCapability

    return ReleaseRequirements(
        release_id="test_blocked",
        release_stage="production_model",
        required_modalities=("text",),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("text_pretrain",),
        optional_tasks=(),
        blocked_capabilities=(
            BlockedCapability(
                capability="image_generation",
                production_status="blocked",
                reason="test blocked capability",
            ),
        ),
    )


def test_required_task_with_insufficient_maturity_is_rejected() -> None:
    contract = _minimal_contract_with_full_evidence()
    contract = dataclasses.replace(
        contract,
        required_tasks=contract.required_tasks + ("table_qa",),
    )
    settings = _production_settings()
    new_tasks = tuple(settings.training.tasks) + ("table_qa",)
    new_training = settings.training.model_copy(update={"tasks": new_tasks})
    new_settings = settings.model_copy(update={"training": new_training})

    with pytest.raises(
        ValueError,
        match=r"table_qa.*maturity",
    ):
        validate_release_task_contracts(
            new_settings,
            contract,
        )


def test_blocked_capability_still_activated_is_rejected() -> None:
    contract = _minimal_contract_with_blocked()
    settings = _production_settings()
    new_image = settings.multimodal.image_generator.model_copy(
        update={"enabled": True}
    )
    new_image_dec = settings.multimodal.image_decoder.model_copy(
        update={"enabled": True}
    )
    new_model = settings.multimodal.model_copy(
        update={
            "image_generator": new_image,
            "image_decoder": new_image_dec,
        }
    )
    new_settings = settings.model_copy(update={"multimodal": new_model})

    with pytest.raises(
        ValueError,
        match=r"blocked capability 'image_generation' is activated",
    ):
        validate_release_task_contracts(new_settings, contract)
