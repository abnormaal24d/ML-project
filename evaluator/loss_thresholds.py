"""Loss and overfitting checks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from schemas.release import ReleaseReason

if TYPE_CHECKING:
    from config.settings.datasets import DatasetValidatorSettings
    from evaluator.results import EvaluationResult


class TrainingMetricsView(Protocol):
    """Structural view of training metrics required for loss evaluation."""

    @property
    def average_loss(self) -> float | None: ...

    @property
    def last_epoch_loss(self) -> float | None: ...

    @property
    def train_loss(self) -> float | None: ...

    @property
    def epoch_history(self) -> Sequence[Mapping[str, object]]: ...


def check_loss_ratio(
    *,
    train_loss: float | None,
    test_loss: float | None,
    max_ratio: float,
) -> tuple[str, ...]:
    """Return a test/train loss-ratio reason when invalid."""

    if train_loss is None or test_loss is None:
        return (ReleaseReason.LOSS_VALUES_MISSING,)
    if train_loss <= 0:
        return (ReleaseReason.TRAIN_LOSS_INVALID_VALUE,)
    ratio = test_loss / train_loss
    if ratio > max_ratio:
        return (ReleaseReason.LOSS_RATIO_EXCEEDED,)
    return ()


def check_losses(
    *,
    settings: DatasetValidatorSettings,
    metrics: TrainingMetricsView,
    evaluation: EvaluationResult,
) -> tuple[str, ...]:
    """Return required-loss and overfitting reasons for a training run."""

    reasons: list[str] = []
    if evaluation.validation_loss is None:
        reasons.append(ReleaseReason.VAL_LOSS_MISSING)
    if evaluation.test_loss is None:
        reasons.append(ReleaseReason.TEST_LOSS_MISSING)
    if settings.require_finite_losses:
        reasons.extend(
            reason
            for value, reason in (
                (metrics.average_loss, ReleaseReason.AVERAGE_LOSS_NON_FINITE),
                (
                    metrics.last_epoch_loss,
                    ReleaseReason.LAST_EPOCH_LOSS_NON_FINITE,
                ),
                (metrics.train_loss, ReleaseReason.TRAIN_LOSS_NON_FINITE),
                (
                    evaluation.validation_loss,
                    ReleaseReason.VAL_LOSS_NON_FINITE,
                ),
                (evaluation.test_loss, ReleaseReason.TEST_LOSS_NON_FINITE),
            )
            if not finite(value)
        )
    if val_loss_rises(metrics=metrics):
        reasons.append(ReleaseReason.VAL_LOSS_RISING)
    return tuple(reasons)


def ratio_reasons(
    *,
    train_loss: float | None,
    test_loss: float | None,
    ratio_limit: float | None,
    model: bool = False,
) -> tuple[str, ...]:
    """Return a loss-ratio reason when the configured rules is exceeded."""

    if ratio_limit is None or not (finite(train_loss) and finite(test_loss)):
        return ()
    resolved_train_loss = float(train_loss or 0.0)
    resolved_test_loss = float(test_loss or 0.0)
    if resolved_train_loss <= 0.0:
        if resolved_test_loss <= 0.0:
            return ()
        reason = (
            ReleaseReason.MODEL_TEST_TRAIN_RATIO_UNBOUNDED
            if model
            else ReleaseReason.TEST_TRAIN_RATIO_UNBOUNDED
        )
        return (reason,)
    if resolved_test_loss / resolved_train_loss <= float(ratio_limit):
        return ()
    reason = (
        ReleaseReason.MODEL_TEST_TRAIN_RATIO_EXCEEDED
        if model
        else ReleaseReason.TEST_TRAIN_RATIO_EXCEEDED
    )
    return (reason,)


def finite(value: float | None) -> bool:
    """Return true only for a present finite number."""

    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def val_loss_rises(*, metrics: TrainingMetricsView) -> bool:
    """Return true when any three validation losses rise in sequence."""

    losses: list[float] = []
    for epoch in metrics.epoch_history:
        value = _optional_float(epoch.get("val_loss"))
        if value is not None:
            losses.append(value)
    return any(
        first < second < third
        for first, second, third in zip(
            losses,
            losses[1:],
            losses[2:],
            strict=False,
        )
    )


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None
