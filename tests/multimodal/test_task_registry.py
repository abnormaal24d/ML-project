"""Invariants for the single canonical multimodal task registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmcrawler_datasets.assembly.build import SAMPLE_BUILDER_TASKS
from multimodal.tasks.registry import (
    TASKS,
    get_task,
    require_task,
    resolved_input_modalities,
    resolved_output_modalities,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_ACTIVE_OBJECTIVES = {
    "language_modeling",
    "text_mlm",
    "classification",
    "contrastive",
    "ocr_sequence",
    "chart_reasoning",
    "math_reasoning",
    "visual_grounding",
    "emotion",
    "speaker_contrastive",
    "speech_translation",
    "audio_generation",
    "image_generation",
    "image_editing",
    "video_generation",
    "causal_language_modeling",
}


def test_registry_contains_unique_tasks() -> None:
    assert len(TASKS) == 61
    assert len(set(TASKS)) == 61


def test_registry_does_not_invent_a_task_for_absent_values() -> None:
    assert get_task(None) is None
    assert get_task("") is None


def test_every_task_has_valid_contract() -> None:
    for definition in TASKS.values():
        assert definition.name
        assert definition.family
        assert definition.required_input_modalities
        assert definition.output_modalities
        assert definition.evaluation_method
        assert definition.loss_key
        assert definition.maturity in {
            "stable",
            "beta",
            "experimental",
            "disabled",
        }
        assert definition.sensitivity in {
            "standard",
            "sensitive",
            "biometric_identity",
        }
        assert definition.sample_source in {
            "self_supervised",
            "crawler_derived",
            "external",
        }


def test_routing_defaults_come_from_registry() -> None:
    for name, definition in TASKS.items():
        assert resolved_input_modalities(name) == (
            definition.required_input_modalities
        )
        assert resolved_output_modalities(name) == (
            definition.output_modalities
        )


def test_beta_sensitive_task_requires_both_approvals() -> None:
    definition = require_task("audio_emotion")
    assert definition.required_approvals == frozenset({"beta", "sensitive"})
    assert definition.production_blocked is False


def test_biometric_task_is_production_blocked() -> None:
    definition = require_task("speaker_id")
    assert definition.production_blocked is True
    assert definition.required_approvals == frozenset()
    assert definition.maturity == "disabled"


def test_stable_pair_tasks_are_production_ready() -> None:
    for name in ("document_text_pair", "pdf_text_pair"):
        definition = require_task(name)
        assert definition.maturity == "stable"
        assert definition.production_blocked is False
        assert definition.required_approvals == frozenset()


def test_document_builders_use_canonical_registered_task_names() -> None:
    assert "document_text_pair" in SAMPLE_BUILDER_TASKS
    assert "pdf_text_pair" in SAMPLE_BUILDER_TASKS
    assert "doc_qa" in SAMPLE_BUILDER_TASKS
    assert "document_qa" not in SAMPLE_BUILDER_TASKS
    for task_name in ("document_text_pair", "pdf_text_pair", "doc_qa"):
        definition = require_task(task_name)
        assert definition.name in TASKS
        assert definition.family == "document"


def test_unmaterialized_generation_and_reasoning_tasks_are_not_claimed() -> (
    None
):
    assert "text_to_image" not in SAMPLE_BUILDER_TASKS
    assert "multimodal_evidence_qa" not in SAMPLE_BUILDER_TASKS


def test_production_required_tasks_have_real_builder_and_loss_evidence(
    production_whisper_env: None,
) -> None:
    from config.load import load_settings
    from config.releases.release_requirements import (
        release_requirements_from_settings,
    )
    from multimodal.tasks.registry import task_has_trainable_loss

    settings = load_settings(
        "prod", project_root=PROJECT_ROOT, environment="prod"
    )
    requirements = release_requirements_from_settings(settings)
    required_tasks = set(requirements.required_tasks)
    assert required_tasks <= SAMPLE_BUILDER_TASKS
    missing_losses = {
        task_name
        for task_name in required_tasks
        if not task_has_trainable_loss(task_name)
    }
    assert missing_losses == set()


def test_production_config_tasks_are_registered(
    production_whisper_env: None,
) -> None:
    from config.load import load_settings

    settings = load_settings(
        "prod", project_root=PROJECT_ROOT, environment="prod"
    )
    training = settings.training
    referenced = (
        set(training.tasks)
        | set(training.approved_beta_tasks)
        | set(training.sensitive_task_approvals)
    )
    unknown = sorted(referenced - set(TASKS))
    assert not unknown, f"prod config references unknown tasks: {unknown}"


def test_default_config_does_not_enable_disabled_tasks() -> None:
    from config.load import load_settings

    settings = load_settings("dev", environment="dev")
    disabled = {
        task_name
        for task_name in settings.training.tasks
        if require_task(task_name).maturity == "disabled"
    }
    assert disabled == set()


def test_hard_negative_capability_comes_from_registry() -> None:
    enabled = {
        name
        for name, definition in TASKS.items()
        if definition.supports_hard_negatives
    }
    assert "semantic_search" in enabled
    assert "multimodal_retrieval" in enabled
    assert "scene_retrieval" in enabled
    assert "passage_retrieval" in enabled
    assert "document_comparison" in enabled
    assert "chart_qa" not in enabled
    assert "doc_qa" not in enabled


@pytest.mark.usefixtures("production_whisper_env")
def test_registered_approvals_are_applicable() -> None:
    from config.load import load_settings

    settings = load_settings(
        "prod", project_root=PROJECT_ROOT, environment="prod"
    )
    training = settings.training

    for task_name in training.approved_beta_tasks:
        assert "beta" in require_task(task_name).required_approvals

    for task_name in training.sensitive_task_approvals:
        assert "sensitive" in require_task(task_name).required_approvals


def test_active_tasks_have_supported_objectives() -> None:
    unsupported = {
        name: definition.loss_key
        for name, definition in TASKS.items()
        if definition.maturity != "disabled"
        and definition.loss_key not in SUPPORTED_ACTIVE_OBJECTIVES
    }
    assert unsupported == {}


def test_speech_to_audio_uses_audio_generation_objective() -> None:
    assert require_task("speech_to_audio").loss_key == "audio_generation"


def test_unknown_task_type_fails_closed() -> None:
    from mmcrawler_datasets.record_components.validation import validate_sample
    from mmcrawler_datasets.schema import MultimodalSample

    sample = MultimodalSample(
        sample_id="sample-1",
        task_type="imaginary_task",
        text="content",
    )
    assert validate_sample(sample=sample) == (
        "unknown_task_type:imaginary_task",
    )


def test_removed_task_architecture_is_not_referenced() -> None:
    root = Path(__file__).resolve().parents[2]
    forbidden_tokens = (
        "TaskDefinitionMetadata",
        "TASK_METADATA_BY_NAME",
        "TaskSettingsCatalog",
    )
    forbidden_imports = (
        "task_capability_resolver",
        "task_registry_validator",
        "task_registry_loader",
        "multimodal.tasks.task_families",
        "config.multimodal.task_settings",
        "config.multimodal.modality_routing_defaults",
    )
    for path in root.rglob("*.py"):
        if any(
            part in {".git", ".venv", "venv", "__pycache__"}
            for part in path.parts
        ):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        for value in forbidden_tokens:
            assert value not in source, f"{value} remains in {path}"
        for value in forbidden_imports:
            assert value not in source, f"{value} remains in {path}"
