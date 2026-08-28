"""CER/WER and layout error rate metric strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from evaluator.metric_contracts import SequenceStatistics

if TYPE_CHECKING:
    pass


class CERWERStrategy:
    """Stateless scoring strategy for CER/WER evaluation."""

    evaluation_method = "cer_wer"

    def accumulate(
        self,
        *,
        state: Any,
        batch: Any,
        outputs: Mapping[str, Any],
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        from evaluator.metrics.sequence.primitives import (
            _special_token_ids,
        )

        labels = batch.decoder_labels
        if labels is None:
            raise ValueError("sequence task batch is missing decoder_labels")
        if not torch.is_tensor(labels):
            raise TypeError("decoder_labels must be a tensor")
        if labels.ndim != 2:
            raise ValueError("decoder_labels must be [batch, tokens]")

        rows = _sequence_evaluation_rows(
            batch=batch,
            labels=labels,
            evaluation_method=self.evaluation_method,
        )
        if not rows:
            return

        special_tokens = _special_token_ids(tokenizer)
        generated = _generate_sequence_predictions(
            model=model,
            batch=batch,
            rows=rows,
            labels=labels,
        )

        for generated_index, batch_index in enumerate(rows):
            _accumulate_sequence_row_metrics(
                batch=batch,
                outputs=outputs,
                labels=labels,
                generated=generated,
                generated_index=generated_index,
                batch_index=batch_index,
                tokenizer=tokenizer,
                special_tokens=special_tokens,
                statistics=state.sequence_statistics,
                method="cer_wer",
            )

    def synchronize(
        self,
        *,
        state: Any,
        device: Any,
    ) -> None:
        from evaluator.distributed.reductions import reduce_sequence_statistics

        reduce_sequence_statistics(
            statistics=state.sequence_statistics,
            device=device,
            evaluation_method=self.evaluation_method,
        )

    def finalize(
        self,
        *,
        state: Any,
    ) -> dict[str, dict[str, float]]:
        task_metrics: dict[str, dict[str, float]] = {}

        from multimodal.tasks.registry import get_task

        for task_type, values in state.sequence_statistics.items():
            definition = get_task(task_type)
            if definition is None:
                continue
            if definition.evaluation_method != "cer_wer":
                continue

            sample_count = values.samples
            if sample_count <= 0.0:
                continue

            metrics = {
                "character_error_rate": (
                    values.character_edits / values.character_reference_length
                ),
                "word_error_rate": (
                    values.word_edits / values.word_reference_length
                ),
            }
            task_metrics[task_type] = metrics
        return task_metrics


class CERWERLayoutStrategy:
    """Stateless scoring strategy for CER/WER with layout evaluation."""

    evaluation_method = "cer_wer_layout"

    def accumulate(
        self,
        *,
        state: Any,
        batch: Any,
        outputs: Mapping[str, Any],
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        from evaluator.metrics.sequence.primitives import (
            _special_token_ids,
        )

        labels = batch.decoder_labels
        if labels is None:
            raise ValueError("sequence task batch is missing decoder_labels")
        if not torch.is_tensor(labels):
            raise TypeError("decoder_labels must be a tensor")
        if labels.ndim != 2:
            raise ValueError("decoder_labels must be [batch, tokens]")

        rows = _sequence_evaluation_rows(
            batch=batch,
            labels=labels,
            evaluation_method=self.evaluation_method,
        )
        if not rows:
            return

        special_tokens = _special_token_ids(tokenizer)
        generated = _generate_sequence_predictions(
            model=model,
            batch=batch,
            rows=rows,
            labels=labels,
        )

        for generated_index, batch_index in enumerate(rows):
            _accumulate_sequence_row_metrics(
                batch=batch,
                outputs=outputs,
                labels=labels,
                generated=generated,
                generated_index=generated_index,
                batch_index=batch_index,
                tokenizer=tokenizer,
                special_tokens=special_tokens,
                statistics=state.sequence_statistics,
                method="cer_wer_layout",
            )

    def synchronize(
        self,
        *,
        state: Any,
        device: Any,
    ) -> None:
        from evaluator.distributed.reductions import reduce_sequence_statistics

        reduce_sequence_statistics(
            statistics=state.sequence_statistics,
            device=device,
            evaluation_method=self.evaluation_method,
        )

    def finalize(
        self,
        *,
        state: Any,
    ) -> dict[str, dict[str, float]]:
        task_metrics: dict[str, dict[str, float]] = {}

        from multimodal.tasks.registry import get_task

        for task_type, values in state.sequence_statistics.items():
            definition = get_task(task_type)
            if definition is None:
                continue
            if definition.evaluation_method != "cer_wer_layout":
                continue

            sample_count = values.samples
            if sample_count <= 0.0:
                continue

            metrics = {
                "character_error_rate": (
                    values.character_edits / values.character_reference_length
                ),
                "word_error_rate": (
                    values.word_edits / values.word_reference_length
                ),
            }
            if values.layout_element_count > 0.0:
                metrics["layout_box_mse"] = (
                    values.layout_squared_error / values.layout_element_count
                )
            task_metrics[task_type] = metrics
        return task_metrics


def _sequence_evaluation_rows(
    *,
    batch: Any,
    labels: Any,
    evaluation_method: str,
) -> list[int]:
    from multimodal.tasks.registry import get_task

    rows: list[int] = []
    for index, task_type in enumerate(batch.task_types):
        if (
            definition := get_task(task_type)
        ) is not None and definition.evaluation_method == evaluation_method:
            if not labels[index].ne(-100).any():
                raise ValueError(
                    f"sequence evaluation row {index} has no target labels"
                )
            rows.append(index)
    return rows


def _generate_sequence_predictions(
    *,
    model: Any,
    batch: Any,
    rows: list[int],
    labels: Any,
) -> Any:
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise TypeError("sequence task evaluation requires model.generate")
    maximum_target_length = max(
        int(labels[index].ne(-100).sum().item()) for index in rows
    )
    from mmcrawler_datasets.collation.multimodal import select_batch_rows

    generation_batch = select_batch_rows(batch=batch, rows=rows)
    generated = generate(
        generation_batch,
        max_new_tokens=maximum_target_length,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )
    if not torch.is_tensor(generated) or generated.ndim != 2:
        raise TypeError("model.generate must return a [batch, tokens] tensor")
    if generated.shape[0] != len(rows):
        raise ValueError("generated token batch does not match selected rows")
    return generated


def _accumulate_sequence_row_metrics(
    *,
    batch: Any,
    outputs: Mapping[str, Any],
    labels: Any,
    generated: Any,
    generated_index: int,
    batch_index: int,
    tokenizer: Any,
    special_tokens: tuple[int, int, int],
    statistics: dict[str, Any],
    method: str,
) -> None:
    from evaluator.metrics.sequence.primitives import (
        _clean_token_ids,
        _decode_evaluation_text,
        _edit_distance,
        _normalize_evaluation_text,
    )

    task_type = batch.task_types[batch_index]
    definition = None
    from multimodal.tasks.registry import get_task

    definition = get_task(task_type)
    if definition is None:
        return

    pad_id, bos_id, eos_id = special_tokens
    target_ids = _clean_token_ids(
        labels[batch_index],
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        ignore_label=-100,
    )
    prediction_ids = _clean_token_ids(
        generated[generated_index],
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        ignore_label=None,
    )
    target_text = _decode_evaluation_text(
        token_ids=target_ids,
        tokenizer=tokenizer,
    )
    prediction_text = _decode_evaluation_text(
        token_ids=prediction_ids,
        tokenizer=tokenizer,
    )
    normalized_target = _normalize_evaluation_text(target_text)
    normalized_prediction = _normalize_evaluation_text(prediction_text)

    values = statistics.setdefault(task_type, SequenceStatistics())
    values.samples += 1.0
    values.exact += float(normalized_prediction == normalized_target)

    if method in ("cer_wer", "cer_wer_layout"):
        values.character_edits += float(
            _edit_distance(
                prediction=list(normalized_prediction),
                target=list(normalized_target),
            )
        )
        values.character_reference_length += float(
            max(len(normalized_target), 1)
        )
        values.word_edits += float(
            _edit_distance(
                prediction=normalized_prediction.split(),
                target=normalized_target.split(),
            )
        )
        values.word_reference_length += float(
            max(len(normalized_target.split()), 1)
        )
        if method == "cer_wer_layout":
            _accumulate_layout_error(
                batch=batch,
                outputs=outputs,
                row_index=batch_index,
                values=values,
            )


def _accumulate_layout_error(
    *,
    batch: Any,
    outputs: Mapping[str, Any],
    row_index: int,
    values: Any,
) -> None:
    prediction = outputs.get("layout_box_prediction")
    target = batch.layout_box_targets
    if (
        not torch.is_tensor(prediction)
        or not torch.is_tensor(target)
        or row_index >= prediction.shape[0]
        or row_index >= target.shape[0]
        or prediction[row_index].shape != target[row_index].shape
    ):
        return
    squared_error = (
        prediction[row_index].float() - target[row_index].float()
    ).square()
    values.layout_squared_error += float(squared_error.sum().item())
    values.layout_element_count += float(squared_error.numel())


__all__ = [
    "CERWERStrategy",
    "CERWERLayoutStrategy",
]
