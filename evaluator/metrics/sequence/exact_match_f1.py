"""Exact match F1 metric strategy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from evaluator.metric_contracts import SequenceStatistics

if TYPE_CHECKING:
    pass


class ExactMatchF1Strategy:
    """Stateless scoring strategy for exact match F1 evaluation."""

    evaluation_method = "exact_match_f1"

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
                method="exact_match_f1",
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
            if definition.evaluation_method != "exact_match_f1":
                continue

            sample_count = values.samples
            if sample_count <= 0.0:
                continue

            exact_match = values.exact / sample_count
            metrics = {
                "sequence_exact_match": exact_match,
                "sequence_token_accuracy": (
                    values.token_correct / values.token_total
                ),
            }
            metrics["exact_match"] = exact_match
            metrics["token_f1"] = values.token_f1_sum / sample_count
            if task_type == "doc_qa":
                metrics["doc_qa_f1"] = metrics["token_f1"]
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
        _normalize_evaluation_text,
        _token_f1,
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
    values.token_correct += float(
        sum(
            predicted == target
            for predicted, target in zip(
                prediction_ids, target_ids, strict=False
            )
        )
    )
    values.token_total += float(max(len(prediction_ids), len(target_ids), 1))

    values.token_f1_sum += _token_f1(
        prediction=prediction_ids,
        target=target_ids,
    )


__all__ = [
    "ExactMatchF1Strategy",
]
