from __future__ import annotations

from config.settings.datasets import DatasetValidatorSettings
from evaluator.loss_thresholds import (
    check_loss_ratio,
    check_losses,
    finite,
    ratio_reasons,
    val_loss_rises,
)
from evaluator.results import EvaluationResult
from evaluator.task_thresholds import (
    append_max_metric_reason,
    append_min_metric_reason,
    check_counts,
    check_sample_minimums,
    check_supervision,
    check_task_samples,
    check_tasks,
    coerce_task_metric_payload,
    first_metric_value,
    task_present,
)
from schemas.release import ReleaseReason


def _metrics(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "train_loss": 0.5,
        "validation_loss": 0.6,
        "test_loss": 0.7,
        "average_loss": 0.6,
        "last_epoch_loss": 0.6,
        "epochs": 1,
        "batches": 10,
        "samples": 80,
        "effective_train_sample_count": 80,
        "effective_task_counts": {
            "image_text_pair": 40,
            "audio_text_pair": 40,
        },
        "effective_modality_counts": {"image": 50, "audio": 50},
        "training_signal_by_modality": {},
        "epoch_history": (),
        "per_modality_losses": {},
        "per_task_losses": {},
    }
    values.update(overrides)
    from training.runtime.results import TrainingMetrics

    return TrainingMetrics(**values)


def _evaluation(**overrides: object) -> EvaluationResult:
    values: dict[str, object] = {
        "validation_loss": 0.6,
        "test_loss": 0.7,
        "evaluation_mode": "test",
        "labeled_sample_count": 20,
        "dataset_split_counts": {},
        "task_metrics": {},
        "test_task_metrics": {},
        "metrics": {},
        "leakage_report_path": None,
        "reproducibility_report_path": None,
        "valid": True,
        "failure_reasons": (),
    }
    values.update(overrides)
    return EvaluationResult(**values)


def test_check_loss_ratio_reports_missing_values() -> None:
    assert check_loss_ratio(train_loss=None, test_loss=0.5, max_ratio=2.0) == (
        ReleaseReason.LOSS_VALUES_MISSING,
    )
    assert check_loss_ratio(train_loss=0.5, test_loss=None, max_ratio=2.0) == (
        ReleaseReason.LOSS_VALUES_MISSING,
    )
    assert check_loss_ratio(train_loss=0.0, test_loss=0.5, max_ratio=2.0) == (
        ReleaseReason.TRAIN_LOSS_INVALID_VALUE,
    )
    assert check_loss_ratio(train_loss=1.0, test_loss=3.0, max_ratio=2.0) == (
        ReleaseReason.LOSS_RATIO_EXCEEDED,
    )
    assert check_loss_ratio(train_loss=1.0, test_loss=1.5, max_ratio=2.0) == ()


def test_check_losses_reports_missing_losses() -> None:
    reasons = check_losses(
        settings=DatasetValidatorSettings(),
        metrics=_metrics(),
        evaluation=_evaluation(validation_loss=None, test_loss=None),
    )
    assert ReleaseReason.VAL_LOSS_MISSING in reasons
    assert ReleaseReason.TEST_LOSS_MISSING in reasons


def test_check_losses_reports_non_finite_losses() -> None:
    reasons = check_losses(
        settings=DatasetValidatorSettings(),
        metrics=_metrics(
            average_loss=float("inf"),
            last_epoch_loss=float("nan"),
        ),
        evaluation=_evaluation(
            validation_loss=float("nan"),
            test_loss=float("inf"),
        ),
    )
    assert ReleaseReason.AVERAGE_LOSS_NON_FINITE in reasons
    assert ReleaseReason.LAST_EPOCH_LOSS_NON_FINITE in reasons
    assert ReleaseReason.TRAIN_LOSS_NON_FINITE not in reasons
    assert ReleaseReason.VAL_LOSS_NON_FINITE in reasons
    assert ReleaseReason.TEST_LOSS_NON_FINITE in reasons


def test_check_losses_reports_rising_validation_loss() -> None:
    metrics = _metrics(
        epoch_history=(
            {"val_loss": 0.1},
            {"val_loss": 0.2},
            {"val_loss": 0.3},
        ),
    )
    reasons = check_losses(
        settings=DatasetValidatorSettings(),
        metrics=metrics,
        evaluation=_evaluation(),
    )
    assert ReleaseReason.VAL_LOSS_RISING in reasons


def test_check_losses_accepts_healthy_run() -> None:
    assert (
        check_losses(
            settings=DatasetValidatorSettings(),
            metrics=_metrics(),
            evaluation=_evaluation(),
        )
        == ()
    )


def test_ratio_reasons_handles_unbounded_train_loss() -> None:
    assert ratio_reasons(
        train_loss=0.0,
        test_loss=0.5,
        ratio_limit=2.0,
    ) == (ReleaseReason.TEST_TRAIN_RATIO_UNBOUNDED,)
    assert (
        ratio_reasons(
            train_loss=0.0,
            test_loss=0.0,
            ratio_limit=2.0,
        )
        == ()
    )
    assert ratio_reasons(
        train_loss=0.0,
        test_loss=0.5,
        ratio_limit=2.0,
        model=True,
    ) == (ReleaseReason.MODEL_TEST_TRAIN_RATIO_UNBOUNDED,)


def test_ratio_reasons_reports_exceeded_ratio() -> None:
    assert ratio_reasons(
        train_loss=1.0,
        test_loss=3.0,
        ratio_limit=2.0,
    ) == (ReleaseReason.TEST_TRAIN_RATIO_EXCEEDED,)
    assert ratio_reasons(
        train_loss=1.0,
        test_loss=3.0,
        ratio_limit=2.0,
        model=True,
    ) == (ReleaseReason.MODEL_TEST_TRAIN_RATIO_EXCEEDED,)
    assert (
        ratio_reasons(
            train_loss=1.0,
            test_loss=1.5,
            ratio_limit=2.0,
        )
        == ()
    )
    assert (
        ratio_reasons(
            train_loss=None,
            test_loss=1.5,
            ratio_limit=2.0,
        )
        == ()
    )
    assert (
        ratio_reasons(
            train_loss=1.0,
            test_loss=1.5,
            ratio_limit=None,
        )
        == ()
    )


def test_finite_accepts_only_present_finite_numbers() -> None:
    assert finite(1.0) is True
    assert finite("2.5") is True
    assert finite(None) is False
    assert finite(float("inf")) is False
    assert finite(float("nan")) is False
    assert finite(object()) is False


def test_val_loss_rises_detects_three_rising_values() -> None:
    assert (
        val_loss_rises(
            metrics=_metrics(
                epoch_history=(
                    {"val_loss": 0.1},
                    {"val_loss": 0.2},
                    {"val_loss": 0.3},
                ),
            )
        )
        is True
    )
    assert (
        val_loss_rises(
            metrics=_metrics(
                epoch_history=(
                    {"val_loss": 0.3},
                    {"val_loss": 0.2},
                    {"val_loss": 0.1},
                ),
            )
        )
        is False
    )
    assert (
        val_loss_rises(
            metrics=_metrics(
                epoch_history=(
                    {"val_loss": "0.1"},
                    {"val_loss": 0.2},
                    {"val_loss": 0.3},
                ),
            )
        )
        is True
    )
    assert (
        val_loss_rises(
            metrics=_metrics(epoch_history=()),
        )
        is False
    )


def test_check_sample_minimums_reports_low_splits() -> None:
    settings = DatasetValidatorSettings(
        min_total_samples=100,
        min_train_samples=50,
        min_val_samples=20,
        min_test_samples=20,
    )
    reasons = check_sample_minimums(
        total=10,
        train=6,
        val=2,
        test=2,
        settings=settings,
    )
    assert set(reasons) == {
        ReleaseReason.MIN_TOTAL_SAMPLES,
        ReleaseReason.MIN_TRAIN_SAMPLES,
        ReleaseReason.MIN_VAL_SAMPLES,
        ReleaseReason.MIN_TEST_SAMPLES,
    }
    assert (
        check_sample_minimums(
            total=100,
            train=50,
            val=20,
            test=20,
            settings=settings,
        )
        == ()
    )


def test_check_task_samples_reports_low_counts() -> None:
    assert check_task_samples(
        task_counts={"image_text_pair": 5},
        min_task_samples={"image_text_pair": 20},
    ) == (ReleaseReason.TASK_MIN_SAMPLES,)
    assert (
        check_task_samples(
            task_counts={"image_text_pair": 20},
            min_task_samples={"image_text_pair": 20},
        )
        == ()
    )
    assert (
        check_task_samples(
            task_counts={},
            min_task_samples={},
        )
        == ()
    )


def test_check_counts_combines_split_and_task_reasons() -> None:
    settings = DatasetValidatorSettings(
        min_total_samples=100,
        min_task_samples={"image_text_pair": 20},
    )
    reasons = check_counts(
        settings=settings,
        total=10,
        train=6,
        val=2,
        test=2,
        task_counts={"image_text_pair": 5},
    )
    assert ReleaseReason.MIN_TOTAL_SAMPLES in reasons
    assert ReleaseReason.TASK_MIN_SAMPLES in reasons


def test_coerce_task_metric_payload_normalizes_inputs() -> None:
    assert coerce_task_metric_payload(None) == {}
    assert coerce_task_metric_payload([1, 2]) == {}
    payload = coerce_task_metric_payload(
        {
            "Image_Text_Pair": {
                "recall_at_1": 0.9,
                "bad": "not-a-number",
                "bool": True,
                "none": None,
                "text": "0.5",
            },
            "broken": "not-a-dict",
        }
    )
    assert payload == {"image_text_pair": {"recall_at_1": 0.9, "text": 0.5}}


def test_check_tasks_reports_visual_task_metrics() -> None:
    settings = DatasetValidatorSettings(
        min_vqa_accuracy=0.8,
        max_visual_grounding_box_mse=1.0,
        max_ocr_character_error_rate=0.1,
        min_doc_qa_f1=0.7,
    )
    task_counts = {
        "vqa": 10,
        "ocr_parse": 10,
        "doc_qa": 10,
    }
    reasons = check_tasks(
        settings=settings,
        task_counts=task_counts,
        task_metrics={
            "vqa": {"vqa_accuracy": 0.5},
            "ocr_parse": {"character_error_rate": 0.5},
            "doc_qa": {"doc_qa_f1": 0.3},
        },
    )
    assert ReleaseReason.VQA_ACCURACY_LOW in reasons
    assert ReleaseReason.OCR_CHARACTER_ERROR_HIGH in reasons
    assert ReleaseReason.DOC_QA_F1_LOW in reasons


def test_check_tasks_reports_audio_task_metrics() -> None:
    settings = DatasetValidatorSettings(
        min_speech_translation_bleu=0.5,
        max_ocr_character_error_rate=0.1,
        min_emotion_f1=0.6,
        min_speaker_accuracy=0.9,
    )
    task_counts = {
        "speech_translation": 10,
        "speech_transcription": 10,
        "audio_emotion": 10,
        "speaker_id": 10,
    }
    reasons = check_tasks(
        settings=settings,
        task_counts=task_counts,
        task_metrics={
            "speech_translation": {"simple_bleu": 0.1},
            "speech_transcription": {
                "character_error_rate": 0.5,
                "word_error_rate": 0.5,
            },
            "audio_emotion": {"emotion_f1": 0.2},
            "speaker_id": {"speaker_accuracy": 0.3},
        },
    )
    assert ReleaseReason.TRANSLATION_BLEU_LOW in reasons
    assert ReleaseReason.TRANSCRIPTION_CHARACTER_ERROR_HIGH in reasons
    assert ReleaseReason.EMOTION_F1_LOW in reasons
    assert ReleaseReason.SPEAKER_ACCURACY_LOW in reasons


def test_check_tasks_reports_generation_task_metrics() -> None:
    settings = DatasetValidatorSettings(
        max_image_generation_mse=1.0,
        min_video_token_accuracy=0.8,
        max_image_editing_preservation_mse=1.0,
        max_visual_grounding_box_mse=1.0,
    )
    task_counts = {
        "text_to_image": 10,
        "text_to_video": 10,
        "image_editing": 10,
        "video_editing": 10,
    }
    reasons = check_tasks(
        settings=settings,
        task_counts=task_counts,
        task_metrics={
            "text_to_image": {"image_mse": 5.0},
            "text_to_video": {"video_token_accuracy": 0.1},
            "image_editing": {"edit_preservation_mse": 5.0},
            "video_editing": {"video_token_accuracy": 0.1},
        },
    )
    assert ReleaseReason.IMAGE_MSE_HIGH in reasons
    assert ReleaseReason.VIDEO_TOKEN_ACCURACY_LOW in reasons
    assert ReleaseReason.IMAGE_EDIT_MSE_HIGH in reasons


def test_check_tasks_accepts_healthy_metrics() -> None:
    settings = DatasetValidatorSettings(
        min_vqa_accuracy=0.8,
        max_ocr_character_error_rate=0.1,
    )
    assert (
        check_tasks(
            settings=settings,
            task_counts={"vqa": 10, "ocr_parse": 10},
            task_metrics={
                "vqa": {"vqa_accuracy": 0.9},
                "ocr_parse": {"character_error_rate": 0.05},
            },
        )
        == ()
    )


def test_check_supervision_reports_missing_supervised_metrics() -> None:
    reasons = check_supervision(
        settings=DatasetValidatorSettings(),
        evaluation=_evaluation(
            valid=False,
            labeled_sample_count=0,
        ).to_payload(),
        task_metrics={},
        effective_task_counts={"image_text_pair": 10},
    )
    assert ReleaseReason.EVALUATION_INVALID in reasons
    assert ReleaseReason.SUPERVISED_METRICS_MISSING in reasons
    assert any(
        reason.startswith(f"{ReleaseReason.TASK_METRIC_MISSING.value}:")
        for reason in reasons
    )


def test_check_supervision_reports_missing_retrieval_accuracy() -> None:
    reasons = check_supervision(
        settings=DatasetValidatorSettings(),
        evaluation=_evaluation().to_payload(),
        task_metrics={"multimodal_retrieval": {"recall_at_1": 0.5}},
        effective_task_counts={"multimodal_retrieval": 10},
    )
    assert ReleaseReason.RETRIEVAL_ACCURACY_MISSING in reasons


def test_check_supervision_accepts_unlabeled_evaluation_without_tasks() -> (
    None
):
    reasons = check_supervision(
        settings=DatasetValidatorSettings(),
        evaluation=_evaluation(
            valid=True,
            labeled_sample_count=0,
            evaluation_mode="unlabeled_loss_only",
        ).to_payload(),
        task_metrics={},
        effective_task_counts={},
    )
    assert reasons == ()


def test_check_supervision_accepts_healthy_run() -> None:
    reasons = check_supervision(
        settings=DatasetValidatorSettings(),
        evaluation=_evaluation().to_payload(),
        task_metrics={"image_text_pair": {"recall_at_1": 0.9}},
        effective_task_counts={"image_text_pair": 10},
    )
    assert reasons == ()


def test_task_present_matches_normalized_names() -> None:
    assert task_present(task_counts={"vqa": 5}, task_type="vqa") is True
    assert task_present(task_counts={"vqa": 5}, task_type="VQA") is True
    assert task_present(task_counts={}, task_type="vqa") is False


def test_first_metric_value_returns_first_finite_value() -> None:
    task_metrics = {
        "vqa": {
            "vqa_accuracy": 0.5,
            "exact_match": 0.9,
        }
    }
    assert (
        first_metric_value(
            task_metrics=task_metrics,
            task_type="vqa",
            metric_names=("vqa_accuracy", "exact_match"),
        )
        == 0.5
    )
    assert (
        first_metric_value(
            task_metrics={"vqa": {"exact_match": 0.9}},
            task_type="vqa",
            metric_names=("vqa_accuracy", "exact_match"),
        )
        == 0.9
    )
    assert (
        first_metric_value(
            task_metrics={},
            task_type="vqa",
            metric_names=("vqa_accuracy",),
        )
        is None
    )


def test_append_min_metric_reason_variants() -> None:
    reasons: list[str] = []
    append_min_metric_reason(
        reasons=reasons,
        task_metrics={"vqa": {"vqa_accuracy": 0.5}},
        task_type="vqa",
        metric_names=("vqa_accuracy",),
        minimum=0.8,
        reason=ReleaseReason.VQA_ACCURACY_LOW,
    )
    assert reasons == [ReleaseReason.VQA_ACCURACY_LOW]
    reasons.clear()
    append_min_metric_reason(
        reasons=reasons,
        task_metrics={},
        task_type="vqa",
        metric_names=("vqa_accuracy",),
        minimum=0.8,
        reason=ReleaseReason.VQA_ACCURACY_LOW,
    )
    assert reasons == [
        f"{ReleaseReason.TASK_METRIC_MISSING.value}:vqa:vqa_accuracy"
    ]
    reasons.clear()
    append_min_metric_reason(
        reasons=reasons,
        task_metrics={"vqa": {"vqa_accuracy": 0.9}},
        task_type="vqa",
        metric_names=("vqa_accuracy",),
        minimum=0.8,
        reason=ReleaseReason.VQA_ACCURACY_LOW,
    )
    assert reasons == []
    append_min_metric_reason(
        reasons=reasons,
        task_metrics={"vqa": {"vqa_accuracy": 0.5}},
        task_type="vqa",
        metric_names=("vqa_accuracy",),
        minimum=None,
        reason=ReleaseReason.VQA_ACCURACY_LOW,
    )
    assert reasons == []


def test_append_max_metric_reason_variants() -> None:
    reasons: list[str] = []
    append_max_metric_reason(
        reasons=reasons,
        task_metrics={"ocr_parse": {"character_error_rate": 0.5}},
        task_type="ocr_parse",
        metric_names=("character_error_rate",),
        maximum=0.1,
        reason=ReleaseReason.OCR_CHARACTER_ERROR_HIGH,
    )
    assert reasons == [ReleaseReason.OCR_CHARACTER_ERROR_HIGH]
    reasons.clear()
    append_max_metric_reason(
        reasons=reasons,
        task_metrics={},
        task_type="ocr_parse",
        metric_names=("character_error_rate",),
        maximum=0.1,
        reason=ReleaseReason.OCR_CHARACTER_ERROR_HIGH,
    )
    assert reasons == [
        f"{ReleaseReason.TASK_METRIC_MISSING.value}:ocr_parse:"
        "character_error_rate"
    ]
    reasons.clear()
    append_max_metric_reason(
        reasons=reasons,
        task_metrics={"ocr_parse": {"character_error_rate": 0.05}},
        task_type="ocr_parse",
        metric_names=("character_error_rate",),
        maximum=0.1,
        reason=ReleaseReason.OCR_CHARACTER_ERROR_HIGH,
    )
    assert reasons == []
    append_max_metric_reason(
        reasons=reasons,
        task_metrics={"ocr_parse": {"character_error_rate": 0.5}},
        task_type="ocr_parse",
        metric_names=("character_error_rate",),
        maximum=None,
        reason=ReleaseReason.OCR_CHARACTER_ERROR_HIGH,
    )
    assert reasons == []
