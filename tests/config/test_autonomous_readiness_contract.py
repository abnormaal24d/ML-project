"""Fail-fast autonomous readiness configuration contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.load import load_settings
from multimodal.tasks.configuration_validation import (
    validate_multimodal_cross_section_settings,
)
from schemas.autonomous_readiness import AUTONOMOUS_REQUIRED_TASKS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_prod_settings_satisfy_autonomous_readiness_contract(
    production_whisper_env: None,
) -> None:
    settings = load_settings(
        "prod",
        project_root=PROJECT_ROOT,
        environment="prod",
    )

    assert (
        settings.datasets.training.dataset_validator.require_autonomous_multimodal_readiness
        is True
    )

    validate_multimodal_cross_section_settings(settings)


def test_autonomous_readiness_rejects_disabled_required_task(
    production_whisper_env: None,
) -> None:
    settings = load_settings(
        "prod",
        project_root=PROJECT_ROOT,
        environment="prod",
    )

    remaining = tuple(
        task for task in settings.training.tasks if task != "audio_text_pair"
    )

    broken = settings.model_copy(
        update={
            "training": settings.training.model_copy(
                update={"tasks": remaining}
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="autonomous readiness requires these training tasks",
    ):
        validate_multimodal_cross_section_settings(broken)


def test_autonomous_readiness_rejects_zero_dataset_minimum(
    production_whisper_env: None,
) -> None:
    settings = load_settings(
        "prod",
        project_root=PROJECT_ROOT,
        environment="prod",
    )

    validator = settings.datasets.training.dataset_validator
    minima = dict(validator.min_task_samples)
    minima["audio_text_pair"] = 0

    broken = settings.model_copy(
        update={
            "datasets": settings.datasets.model_copy(
                update={
                    "training": settings.datasets.training.model_copy(
                        update={
                            "dataset_validator": validator.model_copy(
                                update={"min_task_samples": minima}
                            )
                        }
                    )
                }
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="autonomous readiness requires positive dataset minima",
    ):
        validate_multimodal_cross_section_settings(broken)


def test_autonomous_contract_tasks_match_production_policy() -> None:
    assert set(AUTONOMOUS_REQUIRED_TASKS) == {
        "text_pretrain",
        "image_text_pair",
        "audio_text_pair",
        "video_text_pair",
    }
