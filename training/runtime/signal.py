"""Track encoder gradients and parameter updates during training."""

from __future__ import annotations

import math

import torch

TRACKED_MODALITIES = ("text", "document", "image", "audio", "video")
_UPDATE_EPSILON = 1e-12


class TrainingSignalTracker:
    """Collect per-modality evidence that encoder parameters were trained."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        modalities: tuple[str, ...],
    ) -> None:
        requested = {
            str(modality).strip().lower()
            for modality in modalities
            if str(modality).strip()
        }
        self._modalities = tuple(
            modality
            for modality in TRACKED_MODALITIES
            if modality in requested
        )
        self._parameters = {
            modality: tuple(
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and _parameter_belongs_to_modality(
                    name=name,
                    modality=modality,
                )
            )
            for modality in self._modalities
        }
        self._initial_parameters = {
            modality: tuple(
                parameter.detach().cpu().clone() for parameter in parameters
            )
            for modality, parameters in self._parameters.items()
        }
        self._accumulators: dict[str, dict[str, float | int]] = {
            modality: {
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in self._parameters[modality]
                ),
                "backward_observations": 0,
                "gradient_observations": 0,
                "gradient_l2_sum": 0.0,
                "max_gradient_l2": 0.0,
            }
            for modality in self._modalities
        }

    def record_after_backward(self) -> None:
        """Record gradient norms after a backward pass."""

        for modality, parameters in self._parameters.items():
            accumulator = self._accumulators[modality]
            accumulator["backward_observations"] = (
                int(accumulator["backward_observations"]) + 1
            )
            gradient_l2 = _gradient_l2(parameters=parameters)
            if gradient_l2 <= 0.0:
                continue
            accumulator["gradient_observations"] = (
                int(accumulator["gradient_observations"]) + 1
            )
            accumulator["gradient_l2_sum"] = (
                float(accumulator["gradient_l2_sum"]) + gradient_l2
            )
            accumulator["max_gradient_l2"] = max(
                float(accumulator["max_gradient_l2"]),
                gradient_l2,
            )

    def to_payload(self) -> dict[str, dict[str, object]]:
        """Return JSON-serializable training-signal diagnostics."""

        payload: dict[str, dict[str, object]] = {}
        for modality in self._modalities:
            accumulator = self._accumulators[modality]
            parameter_delta_l2 = _parameter_delta_l2(
                current=self._parameters[modality],
                initial=self._initial_parameters[modality],
            )
            gradient_observations = int(accumulator["gradient_observations"])
            payload[modality] = {
                "trainable_parameter_count": int(
                    accumulator["trainable_parameter_count"]
                ),
                "backward_observations": int(
                    accumulator["backward_observations"]
                ),
                "gradient_observations": gradient_observations,
                "mean_gradient_l2": _rounded(
                    float(accumulator["gradient_l2_sum"])
                    / max(1, gradient_observations)
                ),
                "max_gradient_l2": _rounded(
                    float(accumulator["max_gradient_l2"])
                ),
                "parameter_delta_l2": _rounded(parameter_delta_l2),
                "gradient_detected": gradient_observations > 0,
                "updated": parameter_delta_l2 > _UPDATE_EPSILON,
            }
        return payload


def validate_effective_training_signal(
    *,
    modality_counts: dict[str, int],
    training_signal_by_modality: dict[str, dict[str, object]],
) -> None:
    """Raise if any effective input modality has no encoder train signal."""

    missing = _missing_training_signal(
        modality_counts=modality_counts,
        training_signal_by_modality=training_signal_by_modality,
    )
    if missing:
        raise ValueError(
            "effective_training_signal_missing:" + ";".join(missing)
        )


def _missing_training_signal(
    *,
    modality_counts: dict[str, int],
    training_signal_by_modality: dict[str, dict[str, object]],
) -> tuple[str, ...]:
    missing: list[str] = []
    for modality, count in sorted(modality_counts.items()):
        if int(count) <= 0 or modality == "unknown":
            continue
        signal = training_signal_by_modality.get(modality)
        if signal is None:
            missing.append(f"{modality}:missing")
            continue
        if _nonnegative_int(signal.get("trainable_parameter_count")) <= 0:
            missing.append(f"{modality}:parameters")
        if (
            _nonnegative_int(signal.get("gradient_observations")) <= 0
            or signal.get("gradient_detected") is not True
        ):
            missing.append(f"{modality}:gradient")
        if signal.get("updated") is not True:
            missing.append(f"{modality}:update")
    return tuple(missing)


def _parameter_belongs_to_modality(*, name: str, modality: str) -> bool:
    normalized = name.strip().lower()
    if f"encoders.{modality}." in normalized:
        return True
    if f"{modality}_encoder." in normalized:
        return True
    if modality == "document":
        return (
            "document_text_encoder." in normalized
            or "document_fallback_projection." in normalized
        )
    return False


def _gradient_l2(*, parameters: tuple[torch.nn.Parameter, ...]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        total += float(gradient.square().sum().cpu())
    return math.sqrt(total) if total > 0.0 else 0.0


def _parameter_delta_l2(
    *,
    current: tuple[torch.nn.Parameter, ...],
    initial: tuple[torch.Tensor, ...],
) -> float:
    total = 0.0
    for parameter, initial_value in zip(current, initial, strict=True):
        current_value = parameter.detach().cpu()
        total += float((current_value - initial_value).square().sum())
    return math.sqrt(total) if total > 0.0 else 0.0


def _rounded(value: float) -> float:
    return round(float(value), 10)


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(str(value).strip()))
    except (TypeError, ValueError):
        return 0
