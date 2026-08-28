"""Task-domain cross-section validation for canonical runtime settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.collection.modality_acceptance import (
    DEFAULT_AUDIO_MAX_BYTES,
    DEFAULT_IMAGE_MAX_BYTES,
    DEFAULT_VIDEO_MAX_BYTES,
)
from config.environment.source_selection import PRODUCTION_ENVIRONMENTS
from config.multimodal.training_settings import resolve_objective_loss_weights
from multimodal.tasks.contracts import TaskApproval
from multimodal.tasks.registry import (
    SAMPLE_BUILDER_TASKS,
    TASKS,
    get_task,
    task_producible_loss_terms,
)
from schemas.autonomous_readiness import AUTONOMOUS_REQUIRED_TASKS
from schemas.multimodal_tasks import canonical_task_names

if TYPE_CHECKING:
    from config.settings.root import Settings


def validate_multimodal_cross_section_settings(settings: Settings) -> None:
    """Validate cross-section invariants owned by the multimodal task domain."""

    _validate_enabled_task_loss_coverage(settings)
    _validate_task_build_support(settings)
    _validate_autonomous_readiness_contract(settings)
    _validate_task_media_acceptance(settings)
    _validate_task_maturity_governance(settings)
    _validate_task_specific_release_requirements(settings)
    _validate_transcript_dependent_tasks(settings)
    _validate_video_augmentation_toolchain(settings)


def _validate_enabled_task_loss_coverage(settings: Settings) -> None:
    """Require every configured task to have an active executable loss."""

    training = settings.training
    enabled_tasks = canonical_task_names(
        training.tasks,
        field_name="multimodal.training.tasks",
    )
    positive_terms = {
        term
        for term, weight in resolve_objective_loss_weights(training).items()
        if float(weight) > 0.0
    }
    # Dense training always supplies decoder labels for causal tasks, so its
    # legacy ``sequence`` fallback cannot produce a loss.
    legacy_sequence_supported = (
        training.training_backend != "dense_transformer"
    )

    uncovered: dict[str, list[str]] = {}
    for task_name in sorted(enabled_tasks):
        # Task-catalog ownership belongs to multimodal domain validation.
        # Structural cross-section validation must therefore leave unknown
        # task names for ``validate_multimodal_settings`` to diagnose.
        if get_task(task_name) is None:
            continue
        supported_terms = task_producible_loss_terms(task_name)
        if not legacy_sequence_supported:
            supported_terms = supported_terms - {"sequence"}
        if supported_terms.intersection(positive_terms):
            continue
        uncovered[task_name] = sorted(supported_terms)

    if uncovered:
        raise ValueError(
            "every enabled multimodal task must map to at least one positive "
            "supported loss term; "
            f"training_backend={training.training_backend!r}, "
            f"uncovered={uncovered}"
        )


def _validate_task_build_support(settings: Settings) -> None:
    """Require full build support for every active task with a positive minimum.

    An enabled task with a positive dataset/training minimum must be
    registered and must have a dataset sample builder. Skipped when
    ``disable_undercovered_tasks`` is enabled (dev), because the runtime
    deliberately drops production-undercovered tasks there.
    """

    training = settings.training
    if training.disable_undercovered_tasks:
        return

    validator = settings.datasets.training.dataset_validator
    enabled_tasks = canonical_task_names(
        training.tasks,
        field_name="multimodal.training.tasks",
    )

    training_minimums = training.effective_min_task_samples()
    validator_minimums = validator.effective_min_task_samples()

    active = sorted(
        task
        for task in enabled_tasks
        if int(training_minimums.get(task, 0)) > 0
        or int(validator_minimums.get(task, 0)) > 0
    )

    unknown = sorted(task for task in active if task not in TASKS)
    if unknown:
        raise ValueError(
            "every active task with a positive minimum must be registered "
            f"in the task catalog: {unknown}"
        )

    missing_builders = sorted(
        task for task in active if task not in SAMPLE_BUILDER_TASKS
    )
    if missing_builders:
        raise ValueError(
            "every active task with a positive minimum must have a dataset "
            f"sample builder: {missing_builders}"
        )


def _validate_autonomous_readiness_contract(settings: Settings) -> None:
    """Fail fast when autonomous readiness is enabled but inconsistent."""

    validator = settings.datasets.training.dataset_validator
    if not validator.require_autonomous_multimodal_readiness:
        return

    training = settings.training
    enabled_tasks = set(training.tasks)

    unknown = sorted(
        task for task in AUTONOMOUS_REQUIRED_TASKS if task not in TASKS
    )
    if unknown:
        raise ValueError(
            f"autonomous readiness contains unknown tasks: {unknown}"
        )

    not_buildable = sorted(
        task
        for task in AUTONOMOUS_REQUIRED_TASKS
        if task not in SAMPLE_BUILDER_TASKS
    )
    if not_buildable:
        raise ValueError(
            "autonomous readiness contains tasks without sample builders: "
            f"{not_buildable}"
        )

    disabled = sorted(set(AUTONOMOUS_REQUIRED_TASKS) - enabled_tasks)
    if disabled:
        raise ValueError(
            "autonomous readiness requires these training tasks "
            f"to be enabled: {disabled}"
        )

    dataset_minimums = validator.effective_min_task_samples()
    missing_dataset_minimums = sorted(
        task
        for task in AUTONOMOUS_REQUIRED_TASKS
        if int(dataset_minimums.get(task, 0)) <= 0
    )
    if missing_dataset_minimums:
        raise ValueError(
            "autonomous readiness requires positive dataset minima: "
            f"{missing_dataset_minimums}"
        )

    training_minimums = training.effective_min_task_samples()
    missing_training_minimums = sorted(
        task
        for task in AUTONOMOUS_REQUIRED_TASKS
        if int(training_minimums.get(task, 0)) <= 0
    )
    if missing_training_minimums:
        raise ValueError(
            "autonomous readiness requires positive training minima: "
            f"{missing_training_minimums}"
        )


def _validate_task_media_acceptance(settings: Settings) -> None:
    """Ensure enabled tasks have sufficient media acceptance limits."""

    enabled_tasks = canonical_task_names(
        settings.training.tasks,
        field_name="multimodal.training.tasks",
    )
    if not enabled_tasks:
        return

    acceptance = settings.collection.modality_acceptance
    image_tasks = enabled_tasks & _tasks_requiring_modality("image")
    if image_tasks:
        _require_media_limit(
            field_name="collection.modality_acceptance.image.fetch_max_bytes",
            configured=acceptance.image.fetch_max_bytes,
            required=DEFAULT_IMAGE_MAX_BYTES,
            tasks=image_tasks,
        )
        _require_media_limit(
            field_name=(
                "collection.modality_acceptance.image.preprocessing_max_bytes"
            ),
            configured=acceptance.image.preprocessing_max_bytes,
            required=DEFAULT_IMAGE_MAX_BYTES,
            tasks=image_tasks,
        )

    audio_tasks = enabled_tasks & _tasks_requiring_modality("audio")
    if audio_tasks:
        _require_media_limit(
            field_name="collection.modality_acceptance.audio.fetch_max_bytes",
            configured=acceptance.audio.fetch_max_bytes,
            required=DEFAULT_AUDIO_MAX_BYTES,
            tasks=audio_tasks,
        )

    video_tasks = enabled_tasks & _tasks_requiring_modality("video")
    if video_tasks:
        _require_media_limit(
            field_name="collection.modality_acceptance.video.fetch_max_bytes",
            configured=acceptance.video.fetch_max_bytes,
            required=DEFAULT_VIDEO_MAX_BYTES,
            tasks=video_tasks,
        )


def _validate_task_maturity_governance(settings: Settings) -> None:
    """Validate task governance from registry maturity and sensitivity."""

    training = settings.training
    environment = settings.application.environment
    enabled_tasks = canonical_task_names(
        training.tasks,
        field_name="multimodal.training.tasks",
    )
    beta_approvals = canonical_task_names(
        training.approved_beta_tasks,
        field_name="multimodal.training.approved_beta_tasks",
    )
    sensitive_approvals = canonical_task_names(
        training.sensitive_task_approvals,
        field_name="multimodal.training.sensitive_task_approvals",
    )

    disabled_tasks = sorted(
        task_name
        for task_name in enabled_tasks
        if (definition := get_task(task_name)) is not None
        and definition.maturity == "disabled"
    )
    if disabled_tasks:
        raise ValueError(
            "disabled tasks cannot be enabled in any environment: "
            f"{disabled_tasks}"
        )

    invalid_beta_approvals = sorted(
        task_name
        for task_name in beta_approvals
        if (definition := get_task(task_name)) is not None
        and "beta" not in definition.required_approvals
    )
    if invalid_beta_approvals:
        raise ValueError(
            "approved_beta_tasks contains tasks that do not require "
            f"beta approval: {invalid_beta_approvals}"
        )

    invalid_sensitive_approvals = sorted(
        task_name
        for task_name in sensitive_approvals
        if (definition := get_task(task_name)) is not None
        and "sensitive" not in definition.required_approvals
    )
    if invalid_sensitive_approvals:
        raise ValueError(
            "sensitive_task_approvals contains tasks that do not require "
            f"sensitive approval: {invalid_sensitive_approvals}"
        )

    if environment not in PRODUCTION_ENVIRONMENTS:
        return

    violations: list[str] = []
    for task_name in sorted(enabled_tasks):
        definition = get_task(task_name)
        if definition is None:
            continue
        if definition.production_blocked:
            violations.append(
                f"{task_name}: maturity={definition.maturity!r} "
                f"and sensitivity={definition.sensitivity!r} "
                "do not permit production use"
            )
            continue

        granted: set[TaskApproval] = set()
        if task_name in beta_approvals:
            granted.add("beta")
        if task_name in sensitive_approvals:
            granted.add("sensitive")
        for approval in sorted(definition.required_approvals - granted):
            violations.append(f"{task_name}: missing {approval} approval")

    if violations:
        raise ValueError(
            "production task governance violations: " + "; ".join(violations)
        )


def _validate_task_specific_release_requirements(
    settings: Settings,
) -> None:
    """Validate production-only preprocessing requirements for tasks."""

    training = settings.training
    if training.release_stage != "production_model":
        return

    enabled_tasks = canonical_task_names(
        training.tasks,
        field_name="multimodal.training.tasks",
    )
    processors = settings.collection.processors
    transcription = settings.preprocessing.transcription
    if "speech_transcription" in enabled_tasks and (
        not transcription.enabled
        or not processors.audio.run_transcription
        or not processors.video.run_transcription
    ):
        raise ValueError(
            "speech_transcription requires pinned preprocessing transcription "
            "for both audio and video collection"
        )
    if (
        "ocr_parse" in enabled_tasks
        and settings.preprocessing.ocr.backend == "disabled"
    ):
        raise ValueError(
            "ocr_parse requires an enabled production OCR backend"
        )


def _require_media_limit(
    *,
    field_name: str,
    configured: int | None,
    required: int,
    tasks: set[str],
) -> None:
    """Require a minimum media limit when dependent tasks are enabled."""

    if configured is None or configured >= required:
        return
    raise ValueError(
        f"{field_name} must be at least {required} because these enabled "
        f"tasks require the modality: {sorted(tasks)}"
    )


def _tasks_requiring_modality(modality: str) -> set[str]:
    """Return tasks whose canonical input schema uses a modality."""

    normalized_modality = modality.strip().lower()
    return {
        name
        for name, definition in TASKS.items()
        if normalized_modality
        in {
            str(configured_modality).strip().lower()
            for configured_modality in definition.required_input_modalities
        }
    }


def _validate_transcript_dependent_tasks(settings: Settings) -> None:
    """Reject enabled tasks whose transcripts cannot be produced.

    This is a domain invariant for every environment, not a production-only
    rule: a task that requires transcript evidence must never load with a
    configuration that cannot produce it.
    """

    tasks = set(settings.training.tasks)
    audio_validation = settings.preprocessing.audio_validation
    processors = settings.collection.processors
    transcription = settings.preprocessing.transcription

    requires_audio_transcript = "audio_qa" in tasks or (
        "audio_text_pair" in tasks
        and audio_validation.require_transcript_for_audio_text_pair
    )
    requires_video_transcript = "speech_transcription" in tasks

    if not requires_audio_transcript and not requires_video_transcript:
        return

    if not transcription.enabled:
        raise ValueError(
            "enabled transcription-dependent tasks require "
            "preprocessing.transcription.enabled=true"
        )

    if transcription.backend != "whisper":
        raise ValueError(
            "enabled transcription-dependent tasks require the configured "
            "Whisper backend"
        )

    if requires_audio_transcript and not processors.audio.run_transcription:
        raise ValueError(
            "enabled transcription-dependent audio tasks require "
            "collection.processors.audio.run_transcription=true"
        )

    if requires_video_transcript and not processors.video.run_transcription:
        raise ValueError(
            "speech_transcription requires "
            "collection.processors.video.run_transcription=true"
        )


def _validate_video_augmentation_toolchain(settings: Settings) -> None:
    """Require pinned FFmpeg/FFprobe versions when video augmentation is enabled."""
    video_aug = settings.augmentation.video
    if not video_aug.enabled:
        return
    toolchain = settings.media_toolchain
    if (
        toolchain.ffmpeg_expected_version is None
        or toolchain.ffprobe_expected_version is None
    ):
        raise ValueError(
            "enabled video augmentation requires pinned "
            "ffmpeg_expected_version and ffprobe_expected_version"
        )
