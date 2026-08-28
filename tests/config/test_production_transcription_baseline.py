"""Production transcription baseline, provenance, and fail-fast contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.errors import ConfigError
from config.load import load_settings
from config.settings.fingerprint_sections import build_settings_payloads
from config.validation.cross_section.coordinator import (
    validate_structural_settings,
)
from multimodal.tasks.configuration_validation import (
    validate_multimodal_cross_section_settings,
)
from multimodal.tasks.registry import SAMPLE_BUILDER_TASKS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_WHISPER_ENV_KEYS = (
    "APP_OVERRIDE__preprocessing__transcription__model_name",
    "APP_OVERRIDE__preprocessing__transcription__model_revision",
    "APP_OVERRIDE__preprocessing__transcription__model_artifact_hash",
    "APP_OVERRIDE__preprocessing__transcription__backend_version",
)

DE_SCOPED_NO_BUILDER_TASKS = (
    "summarization",
    "information_extraction",
    "document_summarization",
    "audio_summarization",
    "video_summarization",
)


def _load(environment: str):
    return load_settings(
        environment,
        project_root=PROJECT_ROOT,
        config_root=PROJECT_ROOT,
        environment=environment,
    )


def test_prod_transcription_baseline_is_enabled_and_pinned(
    production_whisper_env: None,
) -> None:
    settings = _load("prod")

    assert settings.application.environment == "prod"
    assert settings.training.release_stage == "candidate"
    transcription = settings.preprocessing.transcription
    assert transcription.enabled is True
    assert transcription.backend == "whisper"
    assert transcription.local_files_only is True
    assert transcription.production_mode is True
    assert transcription.model_name == "/tmp/mmcrawler-test-whisper"
    assert transcription.model_revision == "test-only-revision"
    assert transcription.model_artifact_hash == "0" * 64
    assert transcription.backend_version == "1.1.1"

    assert settings.collection.processors.audio.run_transcription is True
    assert settings.collection.processors.video.run_transcription is True
    assert settings.collection.processors.video.generate_transcriptions is True
    assert settings.collection.processors.video.extract_audio_track is True


def test_whisper_artifact_change_invalidates_crawl_fingerprint(
    production_whisper_env: None,
) -> None:
    settings = _load("prod")

    changed_transcription = settings.preprocessing.transcription.model_copy(
        update={"model_artifact_hash": "1" * 64, "model_revision": "v2"}
    )
    changed_preprocessing = settings.preprocessing.model_copy(
        update={"transcription": changed_transcription}
    )
    changed = settings.model_copy(
        update={"preprocessing": changed_preprocessing}
    )

    baseline = build_settings_payloads(
        settings=settings,
        checker_settings=settings.collection.datachecker,
    )
    upgraded = build_settings_payloads(
        settings=changed,
        checker_settings=changed.collection.datachecker,
    )

    assert upgraded.crawl != baseline.crawl
    assert (
        baseline.crawl["transcription_recipe"]["model_artifact_hash"]
        == "0" * 64
    )
    assert (
        upgraded.crawl["transcription_recipe"]["model_artifact_hash"]
        == "1" * 64
    )


def test_crawl_payload_has_no_transcription_recipe_when_asr_not_requested() -> (
    None
):
    settings = _load("dev")
    processors = settings.collection.processors
    isolated_processors = processors.model_copy(
        update={
            "audio": processors.audio.model_copy(
                update={"run_transcription": False}
            ),
            "video": processors.video.model_copy(
                update={"run_transcription": False}
            ),
        }
    )
    silent = settings.model_copy(
        update={
            "collection": settings.collection.model_copy(
                update={"processors": isolated_processors}
            )
        }
    )
    payloads = build_settings_payloads(
        settings=silent,
        checker_settings=silent.collection.datachecker,
    )
    assert "transcription_recipe" not in payloads.crawl


def test_prod_fails_closed_without_whisper_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in _WHISPER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ConfigError, match="production Whisper"):
        _load("prod")


def test_disabled_transcription_with_audio_qa_fails() -> None:
    settings = _load("dev")
    training = settings.training.model_copy(
        update={"tasks": (*settings.training.tasks, "audio_qa")}
    )
    transcription = settings.preprocessing.transcription.model_copy(
        update={"enabled": False, "backend": "disabled"}
    )
    processors = settings.collection.processors
    isolated_processors = processors.model_copy(
        update={
            "audio": processors.audio.model_copy(
                update={"run_transcription": False}
            ),
            "video": processors.video.model_copy(
                update={"run_transcription": False}
            ),
        }
    )
    invalid = settings.model_copy(
        update={
            "training": training,
            "preprocessing": settings.preprocessing.model_copy(
                update={"transcription": transcription}
            ),
            "collection": settings.collection.model_copy(
                update={"processors": isolated_processors}
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="transcription-dependent tasks require "
        "preprocessing.transcription.enabled=true",
    ):
        validate_multimodal_cross_section_settings(invalid)


def test_audio_text_pair_with_required_transcript_and_no_asr_fails() -> None:
    settings = _load("dev")
    audio_validation = settings.preprocessing.audio_validation.model_copy(
        update={"require_transcript_for_audio_text_pair": True}
    )
    translation = settings.preprocessing.transcription.model_copy(
        update={"enabled": False, "backend": "disabled"}
    )
    processors = settings.collection.processors
    isolated_processors = processors.model_copy(
        update={
            "audio": processors.audio.model_copy(
                update={"run_transcription": False}
            ),
            "video": processors.video.model_copy(
                update={"run_transcription": False}
            ),
        }
    )
    invalid = settings.model_copy(
        update={
            "preprocessing": settings.preprocessing.model_copy(
                update={
                    "audio_validation": audio_validation,
                    "transcription": translation,
                }
            ),
            "collection": settings.collection.model_copy(
                update={"processors": isolated_processors}
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="transcription-dependent tasks require "
        "preprocessing.transcription.enabled=true",
    ):
        validate_multimodal_cross_section_settings(invalid)


def test_video_transcription_without_generate_flag_fails() -> None:
    settings = _load("dev")
    processors = settings.collection.processors
    isolated_processors = processors.model_copy(
        update={
            "video": processors.video.model_copy(
                update={
                    "run_transcription": True,
                    "generate_transcriptions": False,
                    "extract_audio_track": True,
                }
            )
        }
    )
    invalid = settings.model_copy(
        update={
            "collection": settings.collection.model_copy(
                update={"processors": isolated_processors}
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match=r"video\.generate_transcriptions=true",
    ):
        validate_structural_settings(invalid)


def test_video_transcription_without_extract_audio_track_fails() -> None:
    settings = _load("dev")
    processors = settings.collection.processors
    isolated_processors = processors.model_copy(
        update={
            "video": processors.video.model_copy(
                update={
                    "run_transcription": True,
                    "generate_transcriptions": True,
                    "extract_audio_track": False,
                }
            )
        }
    )
    invalid = settings.model_copy(
        update={
            "collection": settings.collection.model_copy(
                update={"processors": isolated_processors}
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match=r"video\.extract_audio_track=true",
    ):
        validate_structural_settings(invalid)


def test_prod_enabled_tasks_all_have_sample_builders(
    production_whisper_env: None,
) -> None:
    settings = _load("prod")
    training = settings.training
    validator = settings.datasets.training.dataset_validator

    training_minimums = training.effective_min_task_samples()
    validator_minimums = validator.effective_min_task_samples()

    active = {
        task
        for task in training.tasks
        if int(training_minimums.get(task, 0)) > 0
        or int(validator_minimums.get(task, 0)) > 0
    }

    assert active <= set(SAMPLE_BUILDER_TASKS)

    for task in DE_SCOPED_NO_BUILDER_TASKS:
        assert task not in training.tasks
        assert int(training_minimums.get(task, 0)) == 0
        assert int(validator_minimums.get(task, 0)) == 0
