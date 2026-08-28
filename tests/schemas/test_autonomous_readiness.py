"""Canonical autonomous multimodal readiness contract tests."""

from __future__ import annotations

from multimodal.tasks.registry import SAMPLE_BUILDER_TASKS, TASKS
from schemas.autonomous_readiness import (
    AUTONOMOUS_REQUIRED_MODALITIES,
    AUTONOMOUS_REQUIRED_TASKS,
    missing_autonomous_modalities,
    missing_autonomous_tasks,
)


def test_autonomous_tasks_are_registered_and_buildable() -> None:
    assert set(AUTONOMOUS_REQUIRED_TASKS) <= set(TASKS)
    assert set(AUTONOMOUS_REQUIRED_TASKS) <= SAMPLE_BUILDER_TASKS


def test_autonomous_contract_contains_expected_modalities_and_tasks() -> None:
    assert AUTONOMOUS_REQUIRED_MODALITIES == (
        "text",
        "image",
        "audio",
        "video",
    )
    assert AUTONOMOUS_REQUIRED_TASKS == (
        "text_pretrain",
        "image_text_pair",
        "audio_text_pair",
        "video_text_pair",
    )


def test_missing_autonomous_helpers_report_only_absent_evidence() -> None:
    assert (
        missing_autonomous_tasks(
            {
                "text_pretrain": 1,
                "image_text_pair": 1,
                "audio_text_pair": 1,
                "video_text_pair": 1,
            }
        )
        == ()
    )
    assert missing_autonomous_tasks(
        {
            "text_pretrain": 1,
            "image_text_pair": 1,
            "audio_text_pair": 0,
            "video_text_pair": 1,
        }
    ) == ("audio_text_pair",)
    assert missing_autonomous_modalities(
        {"text": 1, "image": 1, "audio": 0, "video": 1}
    ) == ("audio",)
