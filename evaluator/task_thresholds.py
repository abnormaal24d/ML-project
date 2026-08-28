"""Sample-count and supervised-task metric checks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mmcrawler_datasets.snapshots.training_dataset_manifest import (
    _nonnegative_int,
)
from schemas.multimodal_tasks import canonical_task_name
from schemas.release import ReleaseReason, detail

if TYPE_CHECKING:
    from config.settings.datasets import DatasetValidatorSettings


def check_sample_minimums(
    *,
    total: int,
    train: int,
    val: int,
    test: int,
    settings: DatasetValidatorSettings,
) -> tuple[str, ...]:
    """Validate against min_*_samples from settings."""
    reasons: list[str] = []

    checks = [
        (
            "total",
            total,
            settings.min_total_samples,
            ReleaseReason.MIN_TOTAL_SAMPLES,
        ),
        (
            "train",
            train,
            settings.min_train_samples,
            ReleaseReason.MIN_TRAIN_SAMPLES,
        ),
        (
            "val",
            val,
            settings.min_val_samples,
            ReleaseReason.MIN_VAL_SAMPLES,
        ),
        (
            "test",
            test,
            settings.min_test_samples,
            ReleaseReason.MIN_TEST_SAMPLES,
        ),
    ]

    for _name, observed, minimum, reason in checks:
        if observed < minimum:
            reasons.append(reason)

    return tuple(reasons)


def check_task_samples(
    *,
    task_counts: dict[str, int],
    min_task_samples: dict[str, int],
) -> tuple[str, ...]:
    """Per-task minimum samples."""
    reasons: list[str] = []
    for task, min_samples in (min_task_samples or {}).items():
        observed = int(task_counts.get(task, 0))
        if observed < min_samples:
            reasons.append(ReleaseReason.TASK_MIN_SAMPLES)
    return tuple(reasons)


def check_counts(
    *,
    settings: DatasetValidatorSettings,
    total: int,
    train: int,
    val: int,
    test: int,
    task_counts: dict[str, int],
) -> tuple[str, ...]:
    """Return split and per-task sample-count reasons."""

    return (
        *check_sample_minimums(
            total=total,
            train=train,
            val=val,
            test=test,
            settings=settings,
        ),
        *check_task_samples(
            task_counts=task_counts,
            min_task_samples=settings.min_task_samples,
        ),
    )


def coerce_task_metric_payload(raw: object) -> dict[str, dict[str, float]]:
    """Normalize evaluator task metrics to finite float-compatible values."""

    if not isinstance(raw, dict):
        return {}
    payload: dict[str, dict[str, float]] = {}
    for task_type, raw_metrics in raw.items():
        if not isinstance(raw_metrics, dict):
            continue
        metrics: dict[str, float] = {}
        for metric_name, value in raw_metrics.items():
            if isinstance(value, bool) or value is None:
                continue
            try:
                metrics[str(metric_name)] = float(value)
            except (TypeError, ValueError):
                continue
        if metrics:
            payload[canonical_task_name(task_type)] = metrics
    return payload


def check_tasks(
    *,
    settings: DatasetValidatorSettings,
    task_counts: dict[str, int],
    task_metrics: dict[str, dict[str, float]],
) -> tuple[str, ...]:
    """Return supervised-task metric acceptance reasons."""

    return (
        *_check_visual_tasks(
            settings=settings,
            task_counts=task_counts,
            task_metrics=task_metrics,
        ),
        *_check_audio_tasks(
            settings=settings,
            task_counts=task_counts,
            task_metrics=task_metrics,
        ),
        *_check_generation_tasks(
            settings=settings,
            task_counts=task_counts,
            task_metrics=task_metrics,
        ),
    )


def check_supervision(
    *,
    settings: DatasetValidatorSettings,
    evaluation: dict[str, object],
    task_metrics: dict[str, dict[str, float]],
    effective_task_counts: dict[str, int],
) -> tuple[str, ...]:
    """Return required supervised-evaluation evidence reasons."""

    labeled_sample_count = _nonnegative_int(
        evaluation.get("labeled_sample_count")
    )
    evaluation_mode = str(evaluation.get("evaluation_mode") or "")
    unlabeled_evaluation = (
        labeled_sample_count == 0
        and evaluation_mode
        in {
            "unlabeled_loss_only",
            "unlabeled_data_quality_only",
            "unlabeled_no_model_metrics",
        }
        and evaluation.get("evaluation_error")
        in {
            None,
            "split_has_no_labels",
        }
    )
    if not settings.require_supervised_metrics_when_labeled:
        return ()
    has_supervised_tasks = any(
        sample_count > 0 for sample_count in effective_task_counts.values()
    )
    if unlabeled_evaluation and not has_supervised_tasks:
        return ()
    reasons: list[str] = []
    if not evaluation.get("valid", False):
        reasons.append(ReleaseReason.EVALUATION_INVALID)
    if labeled_sample_count <= 0:
        reasons.append(ReleaseReason.SUPERVISED_METRICS_MISSING)
    for task_type, sample_count in sorted(effective_task_counts.items()):
        if sample_count > 0 and not task_metrics.get(task_type):
            reasons.append(
                detail(
                    ReleaseReason.TASK_METRIC_MISSING,
                    task_type,
                    "evaluation",
                )
            )
    retrieval_tasks = {
        "multimodal_retrieval",
        "cross_modal_consistency",
        "scene_retrieval",
    }
    if any(effective_task_counts.get(task, 0) > 0 for task in retrieval_tasks):
        if evaluation.get("retrieval_accuracy") is None:
            reasons.append(ReleaseReason.RETRIEVAL_ACCURACY_MISSING)
    return tuple(reasons)


def _check_visual_tasks(
    *,
    settings: DatasetValidatorSettings,
    task_counts: dict[str, int],
    task_metrics: dict[str, dict[str, float]],
) -> tuple[str, ...]:
    """Return VQA, OCR, and document task reasons."""

    reasons: list[str] = []
    if task_present(task_counts=task_counts, task_type="vqa"):
        append_min_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="vqa",
            metric_names=(
                "vqa_accuracy",
                "exact_match",
                "sequence_exact_match",
            ),
            minimum=settings.min_vqa_accuracy,
            reason=ReleaseReason.VQA_ACCURACY_LOW,
        )
        append_max_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="vqa",
            metric_names=("visual_grounding_box_mse",),
            maximum=settings.max_visual_grounding_box_mse,
            reason=ReleaseReason.VQA_BOX_MSE_HIGH,
        )
    if task_present(task_counts=task_counts, task_type="ocr_parse"):
        append_max_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="ocr_parse",
            metric_names=("character_error_rate",),
            maximum=settings.max_ocr_character_error_rate,
            reason=ReleaseReason.OCR_CHARACTER_ERROR_HIGH,
        )
        append_max_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="ocr_parse",
            metric_names=("word_error_rate",),
            maximum=settings.max_ocr_word_error_rate,
            reason=ReleaseReason.OCR_WORD_ERROR_HIGH,
        )
        append_max_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="ocr_parse",
            metric_names=("layout_box_mse",),
            maximum=settings.max_layout_box_mse,
            reason=ReleaseReason.OCR_LAYOUT_MSE_HIGH,
        )
    if task_present(task_counts=task_counts, task_type="doc_qa"):
        append_min_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="doc_qa",
            metric_names=("doc_qa_f1", "token_f1", "sequence_token_accuracy"),
            minimum=settings.min_doc_qa_f1,
            reason=ReleaseReason.DOC_QA_F1_LOW,
        )
        append_max_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="doc_qa",
            metric_names=("layout_box_mse",),
            maximum=settings.max_layout_box_mse,
            reason=ReleaseReason.DOC_QA_LAYOUT_MSE_HIGH,
        )
    return tuple(reasons)


def _check_audio_tasks(
    *,
    settings: DatasetValidatorSettings,
    task_counts: dict[str, int],
    task_metrics: dict[str, dict[str, float]],
) -> tuple[str, ...]:
    """Return speech and audio classification task reasons."""

    reasons: list[str] = []
    if task_present(task_counts=task_counts, task_type="speech_translation"):
        append_min_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="speech_translation",
            metric_names=("simple_bleu", "sequence_token_accuracy"),
            minimum=settings.min_speech_translation_bleu,
            reason=ReleaseReason.TRANSLATION_BLEU_LOW,
        )
    if task_present(task_counts=task_counts, task_type="speech_transcription"):
        append_max_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="speech_transcription",
            metric_names=("character_error_rate",),
            maximum=settings.max_ocr_character_error_rate,
            reason=ReleaseReason.TRANSCRIPTION_CHARACTER_ERROR_HIGH,
        )
        append_max_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="speech_transcription",
            metric_names=("word_error_rate",),
            maximum=settings.max_ocr_word_error_rate,
            reason=ReleaseReason.TRANSCRIPTION_WORD_ERROR_HIGH,
        )
    if task_present(task_counts=task_counts, task_type="audio_emotion"):
        append_min_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="audio_emotion",
            metric_names=("emotion_f1", "emotion_accuracy"),
            minimum=settings.min_emotion_f1,
            reason=ReleaseReason.EMOTION_F1_LOW,
        )
    if task_present(task_counts=task_counts, task_type="speaker_id"):
        append_min_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="speaker_id",
            metric_names=("speaker_accuracy", "speaker_retrieval_accuracy"),
            minimum=settings.min_speaker_accuracy,
            reason=ReleaseReason.SPEAKER_ACCURACY_LOW,
        )
    return tuple(reasons)


def _check_generation_tasks(
    *,
    settings: DatasetValidatorSettings,
    task_counts: dict[str, int],
    task_metrics: dict[str, dict[str, float]],
) -> tuple[str, ...]:
    """Return generation and editing task reasons."""

    reasons: list[str] = []

    if task_present(task_counts=task_counts, task_type="text_to_image"):
        append_max_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="text_to_image",
            metric_names=("image_mse",),
            maximum=settings.max_image_generation_mse,
            reason=ReleaseReason.IMAGE_MSE_HIGH,
        )

    if task_present(task_counts=task_counts, task_type="text_to_video"):
        append_min_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="text_to_video",
            metric_names=("video_token_accuracy",),
            minimum=settings.min_video_token_accuracy,
            reason=ReleaseReason.VIDEO_TOKEN_ACCURACY_LOW,
        )

    if task_present(task_counts=task_counts, task_type="image_editing"):
        append_max_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="image_editing",
            metric_names=("edit_preservation_mse",),
            maximum=settings.max_image_editing_preservation_mse,
            reason=ReleaseReason.IMAGE_EDIT_MSE_HIGH,
        )
        append_max_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="image_editing",
            metric_names=("visual_grounding_box_mse",),
            maximum=settings.max_visual_grounding_box_mse,
            reason=ReleaseReason.IMAGE_EDIT_BOX_MSE_HIGH,
        )

    if task_present(task_counts=task_counts, task_type="video_editing"):
        append_min_metric_reason(
            reasons=reasons,
            task_metrics=task_metrics,
            task_type="video_editing",
            metric_names=("video_token_accuracy",),
            minimum=settings.min_video_token_accuracy,
            reason=ReleaseReason.VIDEO_TOKEN_ACCURACY_LOW,
        )

    return tuple(reasons)


def task_present(*, task_counts: dict[str, int], task_type: str) -> bool:
    """Return whether a normalized task has at least one sample."""

    return task_counts.get(canonical_task_name(task_type), 0) > 0


def append_min_metric_reason(
    *,
    reasons: list[str],
    task_metrics: dict[str, dict[str, float]],
    task_type: str,
    metric_names: tuple[str, ...],
    minimum: float | None,
    reason: ReleaseReason,
) -> None:
    """Append a reason when the first available metric is below minimum."""

    if minimum is None:
        return
    value = first_metric_value(
        task_metrics=task_metrics,
        task_type=task_type,
        metric_names=metric_names,
    )
    if value is None:
        reasons.append(
            detail(
                ReleaseReason.TASK_METRIC_MISSING, task_type, metric_names[0]
            )
        )
        return
    if value < float(minimum):
        reasons.append(reason)


def append_max_metric_reason(
    *,
    reasons: list[str],
    task_metrics: dict[str, dict[str, float]],
    task_type: str,
    metric_names: tuple[str, ...],
    maximum: float | None,
    reason: ReleaseReason,
) -> None:
    """Append a reason when the first available metric is above maximum."""

    if maximum is None:
        return
    value = first_metric_value(
        task_metrics=task_metrics,
        task_type=task_type,
        metric_names=metric_names,
    )
    if value is None:
        reasons.append(
            detail(
                ReleaseReason.TASK_METRIC_MISSING, task_type, metric_names[0]
            )
        )
        return
    if value > float(maximum):
        reasons.append(reason)


def first_metric_value(
    *,
    task_metrics: dict[str, dict[str, float]],
    task_type: str,
    metric_names: tuple[str, ...],
) -> float | None:
    """Return the first finite metric value for a task."""

    metrics = task_metrics.get(task_type)
    if not metrics:
        return None
    for metric_name in metric_names:
        value = metrics.get(metric_name)
        if value is not None and math.isfinite(value):
            return value
    return None
