"""Shared machine-readable contracts for model-release decisions."""

from enum import StrEnum


class ReleaseStatus(StrEnum):
    """Allowed positions on the model-release ladder."""

    FAILED = "FAILED"
    PIPELINE_ACCEPTED = "PIPELINE_ACCEPTED"
    DATASET_ACCEPTED = "DATASET_ACCEPTED"
    MODEL_CANDIDATE = "MODEL_CANDIDATE"
    MODEL_ACCEPTED = "MODEL_ACCEPTED"


class ReleaseReason(StrEnum):
    """Canonical reason codes stored in manifests and reports."""

    CHECKPOINT_MISSING = "checkpoint_missing"
    METRICS_PATH_INVALID = "metrics_path_invalid"
    METRICS_MISSING = "metrics_missing"
    METRICS_INVALID = "metrics_invalid"
    METRICS_EMPTY = "metrics_empty"
    NO_TRAINING_SAMPLES = "no_training_samples"
    NO_COMPLETED_EPOCHS = "no_completed_epochs"
    NO_COMPLETED_BATCHES = "no_completed_batches"
    FINAL_EVALUATION_INVALID = "final_evaluation_invalid"
    TRAIN_LOSS_INVALID = "train_loss_missing_or_non_finite"
    VAL_LOSS_INVALID = "val_loss_missing_or_non_finite"
    TEST_LOSS_INVALID = "test_loss_missing_or_non_finite"
    VAL_LOSS_MISSING = "val_loss_missing"
    TEST_LOSS_MISSING = "test_loss_missing"
    AVERAGE_LOSS_NON_FINITE = "non_finite_average_loss"
    LAST_EPOCH_LOSS_NON_FINITE = "non_finite_last_epoch_loss"
    TRAIN_LOSS_NON_FINITE = "non_finite_train_loss"
    VAL_LOSS_NON_FINITE = "non_finite_val_loss"
    TEST_LOSS_NON_FINITE = "non_finite_test_loss"
    VAL_LOSS_RISING = "overfitting_val_loss_rising_three_epochs"
    LOSS_VALUES_MISSING = "loss_values_missing"
    TRAIN_LOSS_INVALID_VALUE = "invalid_train_loss"
    LOSS_RATIO_EXCEEDED = "max_test_train_loss_ratio_exceeded"
    TEST_TRAIN_RATIO_UNBOUNDED = "test_train_loss_ratio_unbounded"
    TEST_TRAIN_RATIO_EXCEEDED = "test_train_loss_ratio_exceeded"
    MODEL_TEST_TRAIN_RATIO_UNBOUNDED = "model_test_train_loss_ratio_unbounded"
    MODEL_TEST_TRAIN_RATIO_EXCEEDED = "model_test_train_loss_ratio_exceeded"

    MIN_TOTAL_SAMPLES = "min_total_samples_not_met"
    MIN_TRAIN_SAMPLES = "min_train_samples_not_met"
    MIN_VAL_SAMPLES = "min_val_samples_not_met"
    MIN_TEST_SAMPLES = "min_test_samples_not_met"
    MIN_TRAINING_BATCHES = "min_training_batches_not_met"
    TASK_MIN_SAMPLES = "task_min_samples_not_met"
    TASK_MISSING_SAMPLES = "task_missing_samples"
    TASK_EVAL_SAMPLES_MISSING = "task_eval_samples_missing"
    MODEL_MIN_TOTAL_SAMPLES = "model_min_total_samples_not_met"
    MODEL_MIN_TRAIN_SAMPLES = "model_min_train_samples_not_met"
    MODEL_MIN_VAL_SAMPLES = "model_min_val_samples_not_met"
    MODEL_MIN_TEST_SAMPLES = "model_min_test_samples_not_met"
    MODEL_MIN_TRAINING_BATCHES = "model_min_training_batches_not_met"

    EFFECTIVE_SAMPLES_MISSING = "effective_training_samples_missing"
    EFFECTIVE_SAMPLE_MISMATCH = "effective_training_sample_count_mismatch"
    EFFECTIVE_TASK_COUNTS_MISSING = "effective_training_task_counts_missing"
    EFFECTIVE_MODALITY_COUNTS_MISSING = (
        "effective_training_modality_counts_missing"
    )
    EFFECTIVE_TASK_MINIMUM = "effective_training_task_min_samples_not_met"
    AUTONOMOUS_MODALITIES_MISSING = (
        "effective_training_autonomous_modalities_missing"
    )
    AUTONOMOUS_TASKS_MISSING = "effective_training_autonomous_tasks_missing"
    SIGNALS_MISSING = "training_signal_by_modality_missing"
    SIGNAL_MISSING = "training_signal_missing"
    SIGNAL_PARAMETERS_MISSING = "training_signal_parameters_missing"
    SIGNAL_GRADIENT_MISSING = "training_signal_gradient_missing"
    SIGNAL_UPDATE_MISSING = "training_signal_update_missing"

    COVERAGE_REPORT_MISSING = "coverage_report_missing"
    REQUIRED_REPORTS_MISSING = "missing_required_reports"
    EVALUATION_REPORT_MISSING = "evaluation_report_missing"
    ACTIVE_MODALITIES_LOW = "active_modalities_below_min"
    BATCH_LATENCY_HIGH = "batch_latency_above_max"
    PEAK_MEMORY_HIGH = "peak_memory_above_max"
    RUNTIME_MEASUREMENT_MISSING = "runtime_measurement_missing"
    MODALITY_COVERAGE_LOW = "modality_coverage_below_target"
    RAW_COVERAGE_LOW = "raw_modality_coverage_below_min"
    COVERAGE_REPORT_UNREADABLE = "coverage_report_unreadable"

    TASK_METRIC_MISSING = "task_metric_missing"
    EVALUATION_METRIC_MISSING = "evaluation_metric_missing"
    EVALUATION_METRIC_LOW = "evaluation_metric_below_threshold"
    EVALUATION_METRIC_HIGH = "evaluation_metric_above_threshold"
    EVALUATION_METRIC_NON_FINITE = "evaluation_metric_non_finite"
    VQA_ACCURACY_LOW = "task_metric_below_threshold:vqa_accuracy"
    VQA_BOX_MSE_HIGH = (
        "task_metric_above_threshold:vqa_visual_grounding_box_mse"
    )
    OCR_CHARACTER_ERROR_HIGH = (
        "task_metric_above_threshold:ocr_character_error_rate"
    )
    OCR_WORD_ERROR_HIGH = (
        "task_metric_above_threshold:ocr_parse_word_error_rate"
    )
    OCR_LAYOUT_MSE_HIGH = (
        "task_metric_above_threshold:ocr_parse_layout_box_mse"
    )
    DOC_QA_F1_LOW = "task_metric_below_threshold:doc_qa_f1"
    DOC_QA_LAYOUT_MSE_HIGH = (
        "task_metric_above_threshold:doc_qa_layout_box_mse"
    )
    TRANSLATION_BLEU_LOW = (
        "task_metric_below_threshold:speech_translation_bleu"
    )
    TRANSCRIPTION_CHARACTER_ERROR_HIGH = (
        "task_metric_above_threshold:speech_transcription_character_error_rate"
    )
    TRANSCRIPTION_WORD_ERROR_HIGH = (
        "task_metric_above_threshold:speech_transcription_word_error_rate"
    )
    EMOTION_F1_LOW = "task_metric_below_threshold:emotion_f1"
    SPEAKER_ACCURACY_LOW = "task_metric_below_threshold:speaker_accuracy"
    IMAGE_MSE_HIGH = "task_metric_above_threshold:image_generation_mse"
    VIDEO_TOKEN_ACCURACY_LOW = (
        # Release reason code, not credential material.
        "task_metric_below_threshold:video_token_accuracy"  # nosec B105
    )
    IMAGE_EDIT_MSE_HIGH = (
        "task_metric_above_threshold:image_edit_preservation_mse"
    )
    IMAGE_EDIT_BOX_MSE_HIGH = (
        "task_metric_above_threshold:image_editing_visual_grounding_box_mse"
    )

    DATASET_VALIDATION_INVALID = "dataset_validation_report_invalid"
    DATASET_VALIDATION_ERROR = "dataset_validation_error"
    EVALUATION_INVALID = "evaluation_invalid"
    ACCURACY_MISSING = "accuracy_missing"
    MACRO_F1_MISSING = "macro_f1_missing"
    RETRIEVAL_ACCURACY_MISSING = "retrieval_accuracy_missing"
    SUPERVISED_METRICS_MISSING = "supervised_metrics_missing"
    SMOKE_CHECKPOINT = "smoke_trainer_checkpoint"
    MODEL_QUALITY_GATE_FAILED = "model_quality_gate_failed"
    PRODUCTION_REQUIREMENTS = (
        "production_model_acceptance_requirements_not_met"
    )
    RELEASE_REQUIREMENTS_MISSING = "release_requirements_missing"
    RELEASE_REQUIREMENTS_MISMATCH = "release_requirements_mismatch"

    LEAKAGE_REPORT_MISSING = "leakage_report_missing"
    LEAKAGE_EVIDENCE_INCOMPLETE = "leakage_evidence_incomplete"
    LEAKAGE_FOUND = "leakage_found"
    RELEASE_ARTIFACTS_MISSING = "missing_model_release_artifacts"
    RELEASE_ARTIFACT_INVALID = "invalid_release_artifact"
    RELEASE_ARTIFACT_EMPTY = "empty_release_artifact"
    RELEASE_ARTIFACT_PLACEHOLDER = "placeholder_release_artifact"
    SERVING_ARTIFACT_MISSING = "serving_artifact_missing"
    SERVING_ARTIFACT_INVALID = "invalid_serving_artifact"
    SERVING_EXPORT_SKIPPED = "serving_export_skipped"
    ACCEPTANCE_REPORT_FAILED = "acceptance_report_not_passed"
    LEAKAGE_SCHEMA_INVALID = "leakage_schema_invalid"
    LEAKAGE_REPORT_FAILED = "leakage_report_not_passed"
    LEAKAGE_EVIDENCE_MISSING = "leakage_evidence_missing"
    LEAKAGE_DETECTED = "leakage_detected"
    MODEL_CARD_INCOMPLETE = "model_card_incomplete"
    BACKEND_INVALID = (
        "production_model cannot use training_backend='pipeline_smoke'"
    )
    DATASET_CARD_MISSING = "dataset_card.json missing"
    MODEL_CARD_MISSING = "model_card.md missing for production_model"
    REPRODUCIBILITY_MISSING = "reproducibility_report.json missing"
    LEAKAGE_JSON_MISSING = "leakage_report.json missing"
    LEAKAGE_JSON_INVALID = "invalid leakage json"
    LEAKAGE_SCHEMA_MISSING = "leakage schema_version missing or invalid"
    # Release reason code, not credential material.
    LEAKAGE_DID_NOT_PASS = "leakage report did not pass"  # nosec B105
    LEAKAGE_INCOMPLETE = "leakage evidence is incomplete"
    LEAKAGE_PRESENT = "leakage detected"

    BENCHMARK_SUITE_MISSING = "benchmark_suite_missing"
    BASELINE_REFERENCE_MISSING = "baseline_reference_missing"


def detail(reason: ReleaseReason, *values: object) -> str:
    """Append stable detail fields to a canonical reason code."""

    if not values:
        return reason.value

    suffix = ":".join(_detail_value(value) for value in values)
    return f"{reason.value}:{suffix}"


def _detail_value(value: object) -> str:
    """Convert one detail value to stable machine-readable text."""

    if isinstance(value, ReleaseReason):
        return value.value
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, "g")
    return type(value).__name__


__all__ = ["ReleaseReason", "ReleaseStatus", "detail"]
