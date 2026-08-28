"""Strict configuration contract: rejection, override, and secret hygiene."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.augmentation.augmentation_settings import AugmentationSettings
from config.augmentation.video_settings import VideoAugmentationSettings
from config.base.settings_model import SettingsModel
from config.collection.discovery import (
    DiscoveryFeedbackSettings,
    SchedulingSettings,
    WorkerPoolSettings,
)
from config.collection.http_rules import TimeoutRulesSettings
from config.collection.modality_acceptance import ModalityAcceptanceSettings
from config.collection.pacing import PacingSettings
from config.collection.processors import (
    BaseProcessorSettings,
    DocumentProcessorSettings,
    TaskProcessorSettings,
)
from config.collection.settings import CollectionSettings
from config.coverage.settings import CoverageSettings
from config.environment.runtime_environment import configured_token_value
from config.errors import ConfigError
from config.load import load_settings
from config.media_toolchain import MediaToolchainSettings
from config.multimodal.model_settings import ModelSettings
from config.multimodal.training_settings import TrainingSettings
from config.preprocessing.settings import PreprocessingSettings
from config.settings.app import AppSettings
from config.settings.classification import ClassificationSettings
from config.settings.crawler import (
    CrawlerSettings,
    CrawlStateStoreSettings,
)
from config.settings.datasets import DatasetSettings
from config.settings.fingerprint_sections import build_settings_payloads
from config.settings.gate import CrawlOutputGateSettings
from config.settings.logging import LoggingSettings
from config.settings.paths import PathSettings
from config.settings.release import ReleaseSettings
from config.settings.root import Settings
from config.settings.sources import SourcesSettings
from multimodal.tasks.configuration_validation import (
    validate_multimodal_cross_section_settings,
)
from multimodal.tasks.validation import validate_multimodal_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _valid_values() -> dict[str, object]:
    settings = load_settings(
        "dev",
        project_root=PROJECT_ROOT,
        env={},
    )
    return settings.model_dump()


def _media_preflight_settings() -> Settings:
    settings = load_settings(
        "dev",
        project_root=PROJECT_ROOT,
        env={},
    )
    processors = settings.collection.processors
    isolated_processors = processors.model_copy(
        update={
            "audio": processors.audio.model_copy(
                update={"run_transcription": False}
            ),
            "video": processors.video.model_copy(
                update={"run_transcription": False, "run_ocr": False}
            ),
            "image": processors.image.model_copy(
                update={"run_ocr": False, "extract_metadata": False}
            ),
            "document": processors.document.model_copy(
                update={"run_ocr": False, "extract_text": False}
            ),
        }
    )
    audio_validation = settings.preprocessing.audio_validation.model_copy(
        update={
            "chromaprint_executable": "configured-fpcalc",
            "chromaprint_expected_version": "1.5.1",
            "chromaprint_timeout_seconds": 11.0,
        }
    )
    video_augmentation = VideoAugmentationSettings(
        enabled=True,
        probe_timeout_seconds=17.0,
    )
    media_toolchain = settings.media_toolchain.model_copy(
        update={
            "ffmpeg_executable": "configured-ffmpeg",
            "ffprobe_executable": "configured-ffprobe",
            "ffmpeg_expected_version": "8.1.2",
            "ffprobe_expected_version": "8.1.2",
        }
    )
    return settings.model_copy(
        update={
            "collection": settings.collection.model_copy(
                update={"processors": isolated_processors}
            ),
            "preprocessing": settings.preprocessing.model_copy(
                update={"audio_validation": audio_validation}
            ),
            "augmentation": settings.augmentation.model_copy(
                update={"video": video_augmentation}
            ),
            "media_toolchain": media_toolchain,
        }
    )


def test_production_configuration_loads_completely(
    production_whisper_env: None,
) -> None:
    settings = load_settings(
        "prod",
        project_root=PROJECT_ROOT,
    )
    assert settings.profile == "prod"
    assert settings.application.environment == "prod"
    assert settings.meta is not None
    assert settings.meta.profile == "prod"
    assert settings.training.distributed_strategy == "none"
    assert settings.training.ocr_sequence_loss_weight > 0.0


def test_preprocessing_fingerprint_payload_owns_every_preprocessing_setting() -> (
    None
):
    settings = load_settings(
        "dev",
        project_root=PROJECT_ROOT,
        env={},
    )
    baseline = build_settings_payloads(
        settings=settings,
        checker_settings=settings.collection.datachecker,
    )
    updated_media_privacy = settings.preprocessing.media_privacy.model_copy(
        update={"probe_timeout_seconds": 31.0}
    )
    updated = settings.model_copy(
        update={
            "preprocessing": settings.preprocessing.model_copy(
                update={"media_privacy": updated_media_privacy}
            )
        }
    )
    changed = build_settings_payloads(
        settings=updated,
        checker_settings=updated.collection.datachecker,
    )

    assert baseline.preprocessing != changed.preprocessing
    assert baseline.normalization != changed.normalization
    assert baseline.deduplication == changed.deduplication
    assert baseline.splitting == changed.splitting
    assert baseline.validation == changed.validation


def test_dense_ocr_task_requires_dedicated_positive_loss_weight(
    production_whisper_env: None,
) -> None:
    settings = load_settings(
        "prod",
        project_root=PROJECT_ROOT,
    )
    training = settings.training.model_copy(
        update={"ocr_sequence_loss_weight": 0.0}
    )
    invalid_settings = settings.model_copy(update={"training": training})

    with pytest.raises(
        ValueError,
        match=r"positive supported loss term.*ocr_parse",
    ):
        validate_multimodal_cross_section_settings(invalid_settings)


def test_enabled_task_without_producible_loss_is_rejected() -> None:
    settings = load_settings(
        "dev",
        project_root=PROJECT_ROOT,
        env={},
    )
    training = settings.training.model_copy(
        update={
            "tasks": (*settings.training.tasks, "audio_emotion"),
        }
    )
    invalid_settings = settings.model_copy(update={"training": training})

    with pytest.raises(
        ValueError,
        match=r"positive supported loss term.*audio_emotion",
    ):
        validate_multimodal_cross_section_settings(invalid_settings)


def test_speech_to_audio_requires_a_discrete_enabled_tokenizer() -> None:
    settings = load_settings(
        "dev",
        project_root=PROJECT_ROOT,
        env={},
    )
    training = settings.training.model_copy(
        update={"tasks": (*settings.training.tasks, "speech_to_audio")}
    )
    invalid = settings.model_copy(update={"training": training})

    with pytest.raises(ValueError, match="audio_tokenizer.codec='discrete'"):
        validate_multimodal_settings(invalid)


def test_speech_to_audio_accepts_only_one_tokenizer_codebook() -> None:
    settings = load_settings(
        "dev",
        project_root=PROJECT_ROOT,
        env={},
    )
    training = settings.training.model_copy(
        update={"tasks": (*settings.training.tasks, "speech_to_audio")}
    )
    discrete_tokenizer = settings.multimodal.audio_tokenizer.model_copy(
        update={"codec": "discrete", "n_codebooks": 2}
    )
    model = settings.multimodal.model_copy(
        update={"audio_tokenizer": discrete_tokenizer}
    )
    invalid = settings.model_copy(
        update={"training": training, "multimodal": model}
    )

    with pytest.raises(
        ValueError, match="exactly one audio tokenizer codebook"
    ):
        validate_multimodal_settings(invalid)


@pytest.mark.parametrize(
    "task_name",
    ("speech_reconstruction", "speech_translation"),
)
def test_all_audio_generation_tasks_require_discrete_tokenizer(
    task_name: str,
) -> None:
    settings = load_settings(
        "dev",
        project_root=PROJECT_ROOT,
        env={},
    )
    training = settings.training.model_copy(
        update={"tasks": (*settings.training.tasks, task_name)}
    )
    invalid = settings.model_copy(update={"training": training})

    with pytest.raises(
        ValueError,
        match=(
            r"audio generation tasks require .*codec='discrete'.*" + task_name
        ),
    ):
        validate_multimodal_settings(invalid)


def test_unknown_top_level_configuration_key_is_rejected() -> None:
    values = _valid_values()
    values["not_a_section"] = {}
    with pytest.raises(ValidationError):
        Settings.model_validate(values)


def test_unknown_nested_configuration_key_is_rejected() -> None:
    values = _valid_values()
    values["collection"]["not_a_field"] = 1
    with pytest.raises(ValidationError):
        Settings.model_validate(values)


@pytest.mark.parametrize(
    "removed_setting",
    (
        "pdf_header_scan_bytes",
        "text_reader_encodings",
        "remove_duplicate_lines",
        "normalize_whitespace",
        "reject_binary_without_metadata",
    ),
)
def test_removed_document_processor_settings_are_rejected(
    removed_setting: str,
) -> None:
    with pytest.raises(ValueError, match=removed_setting):
        DocumentProcessorSettings.model_validate({removed_setting: False})


def test_document_native_text_replaces_legacy_preview_setting() -> None:
    settings = DocumentProcessorSettings.model_validate(
        {
            "pdf_text_extraction": {"max_pages": 3},
            "native_text": {"max_characters": 1234},
        }
    )

    assert settings.pdf_text_extraction.max_pages == 3
    assert settings.native_text.max_characters == 1234

    with pytest.raises(ValueError, match="text_preview"):
        DocumentProcessorSettings.model_validate(
            {"text_preview": {"max_characters": 1234}}
        )


def test_wrong_configuration_type_is_rejected() -> None:
    values = _valid_values()
    values["collection"]["autoscaler"]["max_workers"] = "many"
    with pytest.raises(ValidationError):
        Settings.model_validate(values)


def test_missing_required_configuration_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ModalityAcceptanceSettings(fetch_max_bytes=10)


def test_negative_timeout_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TimeoutRulesSettings(request_timeout_seconds=-5.0)
    with pytest.raises(ValidationError):
        TimeoutRulesSettings(connect_timeout_seconds=0.0)


def test_enabled_video_augmentation_requires_pinned_tool_versions() -> None:
    settings = Settings(
        profile="dev",
        application=AppSettings(),
        paths=PathSettings(),
        logging=LoggingSettings(),
        sources=SourcesSettings(),
        collection=CollectionSettings(),
        media_toolchain=MediaToolchainSettings(),
        preprocessing=PreprocessingSettings(),
        datasets=DatasetSettings(),
        augmentation=AugmentationSettings(
            video=VideoAugmentationSettings(enabled=True),
        ),
        coverage=CoverageSettings(),
        crawler=CrawlerSettings(),
        crawl_output_gate=CrawlOutputGateSettings(),
        multimodal=ModelSettings(),
        training=TrainingSettings(),
        release=ReleaseSettings(),
        classification=ClassificationSettings(),
    )
    with pytest.raises(
        ValueError,
        match="ffmpeg_expected_version and ffprobe_expected_version",
    ):
        validate_multimodal_cross_section_settings(settings)


def test_media_dependency_preflight_uses_configured_binaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _media_preflight_settings()
    calls: list[tuple[tuple[str, ...], float]] = []
    # Use resolved paths as keys (what Path.resolve() returns on Windows)
    versions = {
        r"C:\tools\configured-fpcalc": "fpcalc version 1.5.1",
        r"C:\tools\configured-ffmpeg": "ffmpeg version 8.1.2",
        r"C:\tools\configured-ffprobe": "ffprobe version 8.1.2",
    }

    def mock_which(executable: str) -> str:
        return f"/tools/{executable}"

    monkeypatch.setattr(
        "preprocessing.media.adapters.versioned_executable._which",
        mock_which,
    )

    def run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> object:
        calls.append((command, float(kwargs["timeout"])))
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": versions.get(command[0], ""),
                "stderr": "",
            },
        )()

    monkeypatch.setattr("subprocess.run", run)

    from orchestration.runtime_dependency_preflight import (
        validate_optional_dependencies,
    )

    report = validate_optional_dependencies(settings=settings)

    assert report.optional_dependency_status["chromaprint_available"] is True
    assert report.optional_dependency_status["ffmpeg_available"] is True
    assert report.optional_dependency_status["ffprobe_available"] is True
    assert calls == [
        ((r"C:\tools\configured-fpcalc", "-version"), 11.0),
        ((r"C:\tools\configured-ffmpeg", "-version"), 17.0),
        ((r"C:\tools\configured-ffprobe", "-version"), 17.0),
    ]


def test_cross_timeout_constraints_are_enforced() -> None:
    with pytest.raises(
        ValueError, match="large_media_request_timeout_seconds"
    ):
        TimeoutRulesSettings(large_media_request_timeout_seconds=10.0)
    with pytest.raises(ValueError, match="head_preflight_timeout_seconds"):
        TimeoutRulesSettings(head_preflight_timeout_seconds=100.0)


def test_zero_worker_count_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BaseProcessorSettings(analysis_workers=0)
    with pytest.raises(ValidationError):
        BaseProcessorSettings(timeout_seconds=0)
    with pytest.raises(ValidationError):
        BaseProcessorSettings(max_retries=-1)


@pytest.mark.parametrize(
    "value",
    ("", ".", "..", "../outside", "/absolute", r"C:\\outside"),
)
def test_crawler_runtime_directories_reject_unsafe_paths(value: str) -> None:
    with pytest.raises(ValidationError):
        CrawlerSettings(control_directory=value)
    with pytest.raises(ValidationError):
        CrawlStateStoreSettings(state_subdirectory=value)


@pytest.mark.parametrize(
    "value",
    ("", ".", "..", "../flag", "nested/flag", r"C:\\flag"),
)
def test_crawler_runtime_files_are_safe_basenames(value: str) -> None:
    with pytest.raises(ValidationError):
        CrawlerSettings(pause_flag_filename=value)
    with pytest.raises(ValidationError):
        CrawlStateStoreSettings(checkpoint_filename=value)


def test_pacing_has_one_bounded_adaptive_rate_contract() -> None:
    settings = PacingSettings(
        default_rps=0.4,
        min_rps=0.2,
        max_rps=0.8,
        ramp_up_factor=1.1,
        backoff_factor=0.5,
        error_cooldown_seconds=12.0,
    )
    assert settings.min_rps < settings.default_rps < settings.max_rps

    with pytest.raises(ValidationError, match="default_rps"):
        PacingSettings(default_rps=1.0, min_rps=0.2, max_rps=0.8)


def test_environment_configuration_overrides() -> None:
    settings = load_settings(
        "dev",
        project_root=PROJECT_ROOT,
        env={
            "APP_OVERRIDE__training__batch_size": "12",
        },
    )
    assert settings.training.batch_size == 12


def test_environment_override_with_wrong_type_is_rejected() -> None:
    with pytest.raises(ConfigError, match="invalid value"):
        load_settings(
            "dev",
            project_root=PROJECT_ROOT,
            env={
                "APP_OVERRIDE__training__batch_size": "many",
            },
        )


def test_secrets_are_never_logged(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "SENTINEL-TOKEN-123")
    assert configured_token_value("HF_TOKEN") == "SENTINEL-TOKEN-123"
    with caplog.at_level(
        logging.DEBUG,
        logger="config.load",
    ):
        load_settings(
            "dev",
            project_root=PROJECT_ROOT,
        )
    assert "SENTINEL-TOKEN-123" not in caplog.text


def test_settings_model_is_frozen() -> None:
    settings = load_settings(
        "dev",
        project_root=PROJECT_ROOT,
        env={},
    )
    with pytest.raises(ValidationError):
        settings.application.environment = "prod"


def test_settings_model_rejects_extra_keys_directly() -> None:
    class _Strict(SettingsModel):
        value: int

    with pytest.raises(ValidationError):
        _Strict(value=1, unknown=True)


def test_worker_pool_uses_a_distinct_finalizer_drain_timeout() -> None:
    settings = WorkerPoolSettings(finalizer_drain_timeout_seconds=12.5)

    assert settings.stop_timeout_seconds == 30.0
    assert settings.finalizer_drain_timeout_seconds == 12.5


def test_scheduling_owns_the_default_retry_wait() -> None:
    settings = SchedulingSettings()

    assert settings.default_retry_wait_seconds == 5.0

    with pytest.raises(ValidationError):
        SchedulingSettings(default_retry_wait_seconds=-0.1)

    with pytest.raises(ValidationError):
        TaskProcessorSettings(empty_backoff_seconds=5.0)


def test_discovery_feedback_owns_host_quality_defaults() -> None:
    settings = DiscoveryFeedbackSettings()

    assert settings.default_host_quality == 0.5
    assert settings.seed_host_quality == 0.65


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("default_host_quality", -0.01),
        ("default_host_quality", 1.01),
        ("seed_host_quality", -0.01),
        ("seed_host_quality", 1.01),
    ),
)
def test_discovery_feedback_rejects_host_quality_outside_unit_interval(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        DiscoveryFeedbackSettings.model_validate({field_name: value})


@pytest.mark.parametrize("value", (0.0, 1.0))
def test_discovery_feedback_accepts_host_quality_interval_boundaries(
    value: float,
) -> None:
    settings = DiscoveryFeedbackSettings(
        default_host_quality=value,
        seed_host_quality=value,
    )

    assert settings.default_host_quality == value
    assert settings.seed_host_quality == value


def test_image_decode_pixel_limits_are_phase_owned_and_positive() -> None:
    from config.collection.modality_acceptance import ImageAcceptanceSettings

    acceptance = ImageAcceptanceSettings(
        fetch_max_bytes=25_000_000,
        preprocessing_max_bytes=25_000_000,
    )

    assert acceptance.max_decode_pixels == 40_000_000

    with pytest.raises(ValidationError):
        ImageAcceptanceSettings(
            fetch_max_bytes=25_000_000,
            preprocessing_max_bytes=25_000_000,
            max_decode_pixels=0,
        )


def test_generated_schema_matches_checked_in() -> None:
    """Ensure docs/configuration_schema.json matches Pydantic model output.

    This catches drift between the Python Settings model and the checked-in
    JSON schema, which can happen when config fields are added/removed without
    updating the generated artifact.
    """

    generated = Settings.model_json_schema()
    document_properties = generated["$defs"]["DocumentProcessorSettings"][
        "properties"
    ]
    assert "native_text" in document_properties
    assert "text_preview" not in document_properties
    assert {
        "pdf_header_scan_bytes",
        "text_reader_encodings",
        "remove_duplicate_lines",
        "normalize_whitespace",
        "reject_binary_without_metadata",
    }.isdisjoint(document_properties)

    checked_in_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "configuration_schema.json"
    )
    checked_in = json.loads(checked_in_path.read_text(encoding="utf-8"))

    # Remove wrapper metadata that Pydantic adds but checked-in schema may not
    generated.pop("$comment", None)
    checked_in.pop("$comment", None)

    assert generated == checked_in, (
        "Generated Pydantic schema differs from checked-in configuration_schema.json. "
        "Run `python regenerate_schema.py` to regenerate."
    )


def test_language_heuristic_rejects_removed_minimum_text_length():
    from config.settings.classification import LanguageHeuristicSettings

    with pytest.raises(ValidationError, match="minimum_text_length"):
        LanguageHeuristicSettings(minimum_text_length=1)


def test_language_heuristic_rejects_removed_max_tokens():
    from config.settings.classification import LanguageHeuristicSettings

    with pytest.raises(ValidationError, match="max_tokens"):
        LanguageHeuristicSettings(max_tokens=1)


def test_language_heuristic_rejects_removed_use_character_distribution():
    from config.settings.classification import LanguageHeuristicSettings

    with pytest.raises(ValidationError, match="use_character_distribution"):
        LanguageHeuristicSettings(use_character_distribution=False)
