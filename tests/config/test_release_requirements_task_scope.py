from __future__ import annotations

from pathlib import Path

import pytest

from config.releases.release_requirements import (
    ReleaseRequirements,
    release_requirements_from_settings,
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


def _minimal_contract_with_full_evidence() -> ReleaseRequirements:
    """Create a minimal contract with only tasks that have full implementation evidence."""
    return ReleaseRequirements(
        release_id="test_minimal",
        release_stage="production_model",
        required_modalities=("text", "document"),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("text_pretrain", "doc_qa"),
        optional_tasks=(),
        blocked_capabilities=(),
    )


def test_production_contract_requires_ocr_parse() -> None:
    settings = _production_settings()
    contract = release_requirements_from_settings(settings)

    assert "ocr_parse" in contract.required_tasks
    assert "ocr_parse" not in contract.optional_tasks


def test_release_scope_summary_contains_no_external_evaluation_suites() -> (
    None
):
    settings = _production_settings()
    contract = release_requirements_from_settings(settings)

    summary = contract.scope_summary()
    assert set(summary) == {
        "release_id",
        "release_stage",
        "in_scope",
        "out_of_scope",
        "blocked_modalities",
    }


def test_task_output_modality_not_decodable_is_rejected(monkeypatch) -> None:
    """Task with audio output (speech_translation) but audio not in model output modalities."""
    import evaluator.results as _evaluator_results
    import mmcrawler_datasets.assembly.build as _assembly_build
    from multimodal.tasks import validation

    # Provide full evidence for speech_translation and base tasks
    required = {
        "text_pretrain",
        "doc_qa",
        "speech_translation",
    }
    monkeypatch.setattr(
        _assembly_build, "SAMPLE_BUILDER_TASKS", frozenset(required)
    )
    monkeypatch.setattr(
        validation, "COLLATION_SUPPORTED_TASKS", frozenset(required)
    )
    monkeypatch.setattr(
        "multimodal.tasks.validation.task_has_trainable_loss",
        lambda name: name in required,
    )
    monkeypatch.setattr(
        _evaluator_results,
        "SUPPORTED_EVALUATION_METHODS",
        frozenset(
            {
                "exact_match_f1",
                "retrieval_or_contrastive",
                "vqa_accuracy",
                "cer_wer",
                "retrieval_accuracy",
                "classification_f1",
                "language_modeling_or_reconstruction",
                "masked_language_modeling",
                "cer_wer_layout",
                "rouge_or_token_f1",
                "bleu",
                "simple_bleu",
                "wer_cer",
                "embedding_quality",
                "bleu_audio_latency",
            }
        ),
    )
    monkeypatch.setattr(
        validation,
        "MODEL_OUTPUT_MODALITIES",
        frozenset({"text", "json", "code", "class", "embedding"}),
    )

    # Create contract requiring speech_translation (has audio output)
    contract = ReleaseRequirements(
        release_id="test_speech",
        release_stage="production_model",
        required_modalities=("text", "audio"),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("text_pretrain", "speech_translation"),
        optional_tasks=(),
        blocked_capabilities=(),
    )
    settings = _production_settings()
    # Need audio output modality enabled and vocoder for speech_translation
    settings = settings.model_copy(
        update={
            "multimodal": settings.multimodal.model_copy(
                update={
                    "output_modalities": (
                        "class",
                        "embedding",
                        "text",
                        "json",
                        "audio",
                    ),
                    "vocoder": settings.multimodal.vocoder.model_copy(
                        update={"enabled": True}
                    ),
                    "audio_tokenizer": (
                        settings.multimodal.audio_tokenizer.model_copy(
                            update={
                                "enabled": True,
                                "codec": "discrete",
                                "n_codebooks": 1,
                            }
                        )
                    ),
                }
            ),
            "training": settings.training.model_copy(
                update={
                    "tasks": [*settings.training.tasks, "speech_translation"],
                }
            ),
        }
    )

    with pytest.raises(ValueError, match=r"speech_translation.*model_output"):
        validate_release_task_contracts(settings, contract)


def test_required_generative_task_without_inference_support_is_rejected() -> (
    None
):
    """Required text-generating task (instruction_following) with text_decoder disabled."""
    contract = ReleaseRequirements(
        release_id="test_inference",
        release_stage="production_model",
        required_modalities=("text",),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("text_pretrain", "instruction_following"),
        optional_tasks=(),
        blocked_capabilities=(),
    )
    settings = _production_settings()
    # Disable text decoder - instruction_following is already in tasks
    new_decoder = settings.multimodal.text_decoder.model_copy(
        update={"enabled": False}
    )
    new_model = settings.multimodal.model_copy(
        update={"text_decoder": new_decoder}
    )
    new_settings = settings.model_copy(update={"multimodal": new_model})

    with pytest.raises(ValueError, match=r"instruction_following.*inference"):
        validate_release_task_contracts(new_settings, contract)
