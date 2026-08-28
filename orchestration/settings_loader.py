"""Load and validate the canonical settings tree for orchestration.

``config.load`` owns source selection, merging, path resolution, and
structural/config-owned validation.  This module is the single orchestration
seam that adds runtime domain preflight checks before settings reach an
entrypoint or composition root.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from config.environment.runtime_environment import world_size
from config.environment.source_selection import (
    DEFAULT_CONFIG_DIR,
    packaged_config_root,
)
from config.errors import ConfigError, RuntimeDependencyError
from config.load import load_settings
from config.releases.release_requirements import (
    release_requirements_from_settings,
    validate_release_requirements,
)
from config.settings.root import Settings
from crawler.governance.processing_activity import (
    PROCESSING_ACTIVITIES_CONFIG_RELATIVE_PATH,
    PROCESSING_ACTIVITIES_SETTING,
    ProcessingActivityConfigError,
    ProcessingActivityRegistry,
    load_processing_activities,
)
from multimodal.tasks.configuration_validation import (
    validate_multimodal_cross_section_settings,
)
from multimodal.tasks.validation import validate_multimodal_settings
from orchestration.errors import (
    BackendConfigurationError,
    SettingsLoadError,
    SettingsValidationError,
    StartupConfigurationError,
)
from orchestration.runtime_dependency_preflight import (
    OptionalDependencyReport,
    validate_optional_dependencies,
)
from release.task_contract_validation import validate_release_task_contracts


@dataclass(frozen=True, slots=True)
class RuntimeReadiness:
    """Read-only artifacts produced by the canonical runtime preflight."""

    processing_activity_registry: ProcessingActivityRegistry
    dependency_report: OptionalDependencyReport


def _selected_config_root(config_root: str | Path | None) -> Path:
    """Return the artifact root used by orchestration preflight checks."""

    if config_root is None:
        return packaged_config_root()
    return Path(config_root).expanduser().resolve()


def _validate_release_contract(settings: Settings) -> None:
    requirements = release_requirements_from_settings(settings)
    if requirements is None:
        return

    validate_release_requirements(
        release_requirements=requirements,
        enabled_modalities=settings.multimodal.enabled_modalities,
        enabled_tasks=settings.training.tasks,
        active_release_stage=settings.training.release_stage,
    )
    validate_release_task_contracts(
        settings,
        requirements,
    )


def _settings_error(error: BaseException) -> SettingsLoadError:
    """Classify one expected loader/validator failure without logging it."""

    error_type = type(error).__name__
    context: dict[str, object] = {"error_type": error_type}
    if isinstance(error, ConfigError):
        if error.setting is not None:
            context["setting"] = error.setting
        if error.issue is not None:
            context["issue"] = error.issue
    error_class = (
        SettingsValidationError
        if isinstance(error, ValueError)
        else SettingsLoadError
    )
    return error_class(
        str(error) or "runtime settings could not be loaded",
        stage="bootstrap",
        component="settings",
        cause=error,
        context=context,
    )


def load(
    *,
    project_root: str | Path | None = None,
    config_root: str | Path | None = None,
    environment: str | None = None,
    profile: str | None = None,
    overrides: Sequence[str] | None = None,
    use_cuda: bool | None = None,
) -> Settings:
    """Load settings and apply all orchestration-owned startup validation."""

    selected_config_root = _selected_config_root(config_root)
    effective_overrides = list(overrides or ())
    if use_cuda:
        effective_overrides.append("training.device=cuda")

    try:
        settings = load_settings(
            profile=profile,
            project_root=project_root,
            config_root=selected_config_root,
            environment=environment,
            overrides=effective_overrides,
        )
    except (
        ConfigError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise _settings_error(error) from error

    try:
        validate_multimodal_settings(settings)
        validate_multimodal_cross_section_settings(settings)
        _validate_release_contract(settings)
    except (
        ConfigError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise _settings_error(error) from error

    return settings


def validate_runtime_configuration(
    *,
    settings: Settings,
    config_root: str | Path | None = None,
) -> RuntimeReadiness:
    """Validate every runtime prerequisite without mutating workflow state."""

    distributed_strategy = settings.training.distributed_strategy
    try:
        configured_world_size = world_size()
    except ValueError as error:
        raise BackendConfigurationError(
            "distributed launcher configuration is invalid",
            stage="bootstrap",
            component="training_backend",
            cause=error,
            context={"error_type": type(error).__name__},
        ) from error
    if configured_world_size != 1 or distributed_strategy in {"ddp", "fsdp"}:
        raise BackendConfigurationError(
            "the autonomous workflow is single-process; configure "
            "distributed_strategy='none' or 'auto' with WORLD_SIZE=1 "
            f"(strategy={distributed_strategy!r}, "
            f"WORLD_SIZE={configured_world_size})",
            stage="bootstrap",
            component="training_backend",
            context={
                "distributed_strategy": distributed_strategy,
                "world_size": configured_world_size,
            },
        )

    try:
        from training.runtime.preparation import prepare_training_backend
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise BackendConfigurationError(
            "the configured training backend requires PyTorch",
            stage="bootstrap",
            component="training_backend",
            cause=error,
            context={"issue": "training_runtime_dependency_missing"},
        ) from error

    try:
        prepare_training_backend(training_settings=settings.training)
    except ValueError as error:
        raise BackendConfigurationError(
            str(error) or "training backend configuration is invalid",
            stage="bootstrap",
            component="training_backend",
            cause=error,
            context={"error_type": type(error).__name__},
        ) from error

    selected_config_root = _selected_config_root(config_root)
    processing_activities_path = (
        selected_config_root
        / DEFAULT_CONFIG_DIR
        / PROCESSING_ACTIVITIES_CONFIG_RELATIVE_PATH
    )
    try:
        processing_activity_registry = load_processing_activities(
            processing_activities_path,
            setting=PROCESSING_ACTIVITIES_SETTING,
        )
    except ProcessingActivityConfigError as error:
        raise StartupConfigurationError(
            "processing activity configuration is invalid",
            stage="bootstrap",
            component=error.component,
            cause=error,
            context={
                "setting": error.setting,
                "file": error.basename,
                "issue": error.code,
            },
        ) from error

    try:
        _validate_required_local_artifacts(settings=settings)
        dependency_report = validate_optional_dependencies(settings=settings)
    except RuntimeDependencyError as error:
        context: dict[str, object] = {"issue": error.issue}
        if error.setting is not None:
            context["setting"] = error.setting
        if error.required_artifact is not None:
            context["file"] = error.required_artifact
        raise BackendConfigurationError(
            str(error) or "runtime dependency configuration is invalid",
            stage="bootstrap",
            component="runtime_dependencies",
            cause=error,
            context=context,
        ) from error

    return RuntimeReadiness(
        processing_activity_registry=processing_activity_registry,
        dependency_report=dependency_report,
    )


def _validate_required_local_artifacts(*, settings: Settings) -> None:
    """Validate model artifacts selected by active preprocessing features."""

    processors = settings.collection.processors
    transcription = settings.preprocessing.transcription
    if transcription.enabled and (
        processors.audio.run_transcription
        or processors.video.run_transcription
    ):
        from preprocessing.media.adapters.whisper_model_loader import (
            installed_backend_version,
            validate_whisper_artifact,
        )

        validate_whisper_artifact(
            settings=transcription,
            observed_backend_version=installed_backend_version(),
        )

    diarization = settings.preprocessing.diarization
    if diarization.enabled and diarization.backend == "pyannote":
        from preprocessing.media.adapters.pyannote_adapter import (
            validate_diarization_artifact,
        )

        validate_diarization_artifact(settings=diarization)

    if (
        processors.image.run_ocr
        or processors.video.run_ocr
        or processors.document.run_ocr
    ):
        from preprocessing.media.adapters.tesseract_engine import (
            validate_tesseract_artifact,
        )

        validate_tesseract_artifact(settings=settings.preprocessing.ocr)


__all__ = ["RuntimeReadiness", "load", "validate_runtime_configuration"]
