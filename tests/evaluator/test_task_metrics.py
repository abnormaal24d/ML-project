from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
import torch

from evaluator.metric_contracts import PAIR_TASKS
from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL
from multimodal.model.contracts import CollatedBatch


def _task_batch(
    *,
    sample_ids: list[str],
    task_types: list[str],
    **overrides: object,
) -> CollatedBatch:
    size = len(sample_ids)
    return CollatedBatch(
        sample_ids=sample_ids,
        text=torch.zeros((size, 2)),
        image=torch.zeros((size, 2)),
        audio=torch.zeros((size, 2)),
        video=torch.zeros((size, 2)),
        modality_mask=torch.ones((size, 4), dtype=torch.bool),
        labels=None,
        task_types=task_types,
        **overrides,
    )


class _StoredOutputsModel(torch.nn.Module):
    def __init__(self, outputs: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.outputs = dict(outputs)

    def forward(self, batch: object) -> dict[str, torch.Tensor]:
        return self.outputs


class _SequentialOutputsModel(torch.nn.Module):
    def __init__(self, outputs: Sequence[Mapping[str, torch.Tensor]]) -> None:
        super().__init__()
        self.outputs = list(outputs)

    def forward(self, batch: object) -> dict[str, torch.Tensor]:
        return dict(self.outputs.pop(0))


def test_validation_and_test_metric_calls_remain_independent() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    validation_batch = _task_batch(
        sample_ids=["validation-a", "validation-b"],
        task_types=["image_text_pair", "image_text_pair"],
        text_mlm_targets=None,
    )
    test_batch = _task_batch(
        sample_ids=["test-a", "test-b"],
        task_types=["image_text_pair", "image_text_pair"],
        text_mlm_targets=None,
    )

    validation_metrics = evaluate_task_metrics(
        model=_StoredOutputsModel(
            {
                "text_embedding": torch.eye(2),
                "image_embedding": torch.eye(2),
                "contrastive_row_indices": torch.tensor([0, 1]),
            }
        ),
        loader=[validation_batch],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )
    test_metrics = evaluate_task_metrics(
        model=_StoredOutputsModel(
            {
                "text_embedding": torch.eye(2),
                "image_embedding": torch.flip(torch.eye(2), dims=(0,)),
                "contrastive_row_indices": torch.tensor([0, 1]),
            }
        ),
        loader=[test_batch],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    assert validation_metrics["image_text_pair"]["recall_at_1"] == 1.0
    assert test_metrics["image_text_pair"]["recall_at_1"] == 0.0


def test_runtime_variant_measures_latency_and_peak_memory() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import (
        evaluate_task_metrics_with_runtime,
    )

    batch = _task_batch(
        sample_ids=["runtime-a", "runtime-b"],
        task_types=["image_text_pair", "image_text_pair"],
        text_mlm_targets=None,
    )
    model = _StoredOutputsModel(
        {
            "text_embedding": torch.eye(2),
            "image_embedding": torch.eye(2),
            "contrastive_row_indices": torch.tensor([0, 1]),
        }
    )

    metrics, max_latency_ms, peak_memory_mb = (
        evaluate_task_metrics_with_runtime(
            model=model,
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )
    )

    assert metrics["image_text_pair"]["recall_at_1"] == 1.0
    assert max_latency_ms is not None and max_latency_ms >= 0.0
    assert peak_memory_mb is None or peak_memory_mb >= 0.0


def test_runtime_variant_keeps_the_highest_observed_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext

    import evaluator.runtime_metrics as runtime_metrics
    import evaluator.task_metrics as task_metrics

    peaks = iter((10.0, 100.0, 20.0))
    monkeypatch.setattr(
        runtime_metrics,
        "_peak_memory_mb",
        lambda device: next(peaks),
    )
    batches = [
        _task_batch(
            sample_ids=[f"runtime-{index}"],
            task_types=["image_text_pair"],
            text_mlm_targets=None,
        )
        for index in range(2)
    ]
    outputs = [
        {
            "text_embedding": torch.tensor([[1.0, 0.0]]),
            "image_embedding": torch.tensor([[1.0, 0.0]]),
            "contrastive_row_indices": torch.tensor([0]),
        }
        for _ in batches
    ]

    _, _, peak_memory_mb = task_metrics.evaluate_task_metrics_with_runtime(
        model=_SequentialOutputsModel(outputs),
        loader=batches,
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    assert peak_memory_mb == 100.0


def test_runtime_cuda_measurement_resets_peak_and_synchronizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evaluator.runtime_metrics as runtime_metrics

    calls: list[tuple[str, torch.device]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *, device: calls.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda *, device: calls.append(("reset", device)),
    )

    device = torch.device("cuda:0")
    runtime_metrics._reset_cuda_peak_memory(device=device)
    runtime_metrics._synchronize_cuda(device=device)

    assert calls == [("reset", device), ("synchronize", device)]


def test_runtime_variant_requires_loader() -> None:
    from evaluator.task_metrics import (
        evaluate_task_metrics_with_runtime,
    )

    with pytest.raises(RuntimeError, match="evaluation loader is required"):
        evaluate_task_metrics_with_runtime(
            model=_StoredOutputsModel({}),
            loader=None,
            device=torch.device("cpu"),
            autocast_factory=lambda: __import__("contextlib").nullcontext(),
        )


def test_retrieval_metrics_use_all_batches_in_the_split() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    text_embeddings = torch.eye(6)
    media_embeddings = torch.ones((6, 6)) - torch.eye(6)
    batches = [
        _task_batch(
            sample_ids=[f"sample-{index}"],
            task_types=["image_text_pair"],
            text_mlm_targets=None,
        )
        for index in range(6)
    ]
    outputs = [
        {
            "text_embedding": text_embeddings[index : index + 1],
            "image_embedding": media_embeddings[index : index + 1],
            "contrastive_row_indices": torch.tensor([0]),
        }
        for index in range(6)
    ]

    metrics = evaluate_task_metrics(
        model=_SequentialOutputsModel(outputs),
        loader=batches,
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )["image_text_pair"]

    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_5"] == 0.0
    assert metrics["embedding_similarity_mean"] == 0.0


def test_pair_task_order_is_deterministic() -> None:
    from evaluator.metric_contracts import (
        PAIR_TASK_ORDER,
        PAIR_TASKS,
    )

    assert isinstance(PAIR_TASK_ORDER, tuple)
    assert PAIR_TASKS == frozenset(PAIR_TASK_ORDER)
    assert list(PAIR_TASK_ORDER) == [
        "document_text_pair",
        "pdf_text_pair",
        "image_text_pair",
        "audio_text_pair",
        "video_text_pair",
        "multimodal_retrieval",
        "cross_modal_consistency",
    ]


def test_sequence_task_metrics_use_deterministic_generation() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    class ExactGenerationModel(torch.nn.Module):
        def forward(self, batch: CollatedBatch) -> dict[str, torch.Tensor]:
            return {}

        def generate(
            self, batch: CollatedBatch, **kwargs: object
        ) -> torch.Tensor:
            assert kwargs["temperature"] == 0.0
            return torch.tensor([[10, 11, 3]], device=batch.text.device)

    batch = _task_batch(
        sample_ids=["doc-qa"],
        task_types=["doc_qa"],
        text_mlm_targets=None,
        decoder_input_ids=torch.tensor([[2, 9]]),
        decoder_attention_mask=torch.tensor([[True, True]]),
        decoder_labels=torch.tensor([[IGNORE_LABEL, 10, 11, 3]]),
        prompt_token_count=[2],
    )

    metrics = evaluate_task_metrics(
        model=ExactGenerationModel(),
        loader=[batch],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )["doc_qa"]

    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["doc_qa_f1"] == 1.0


def test_cer_wer_sequence_metrics_accumulate_edit_counts() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    class ExactGenerationModel(torch.nn.Module):
        def forward(self, batch: CollatedBatch) -> dict[str, torch.Tensor]:
            return {}

        def generate(
            self, batch: CollatedBatch, **kwargs: object
        ) -> torch.Tensor:
            del kwargs
            return torch.tensor([[10, 11, 3]], device=batch.text.device)

    batch = _task_batch(
        sample_ids=["speech-1"],
        task_types=["speech_transcription"],
        text_mlm_targets=None,
        decoder_input_ids=torch.tensor([[2, 9]]),
        decoder_attention_mask=torch.tensor([[True, True]]),
        decoder_labels=torch.tensor([[IGNORE_LABEL, 10, 11, 3]]),
        prompt_token_count=[2],
    )

    metrics = evaluate_task_metrics(
        model=ExactGenerationModel(),
        loader=[batch],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )["speech_transcription"]

    assert metrics["character_error_rate"] == 0.0
    assert metrics["word_error_rate"] == 0.0


def test_causal_language_modeling_uses_its_own_finalizer() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    logits = torch.full((1, 4, 16), -100.0)
    logits[0, 0, 10] = 100.0
    logits[0, 1, 11] = 100.0
    logits[0, 2, 3] = 100.0
    batch = _task_batch(
        sample_ids=["causal-1"],
        task_types=["causal_text_pretrain"],
        text_mlm_targets=None,
        decoder_labels=torch.tensor([[IGNORE_LABEL, 10, 11, 3]]),
    )

    metrics = evaluate_task_metrics(
        model=_StoredOutputsModel({"sequence_logits": logits}),
        loader=[batch],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )["causal_text_pretrain"]

    assert metrics["next_token_accuracy"] == 1.0
    assert metrics["perplexity"] == pytest.approx(1.0)
    assert "sequence_exact_match" not in metrics


def test_task_metrics_use_sparse_contrastive_row_indices() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["pair-a", "pair-b", "pair-c", "pair-d"],
        task_types=[
            "image_text_pair",
            "image_text_pair",
            "image_text_pair",
            "image_text_pair",
        ],
        text_mlm_targets=None,
    )
    model = _StoredOutputsModel(
        {
            "text_embedding": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.99, 0.1],
                    [0.1, 0.99],
                ]
            ),
            "image_embedding": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.99, 0.1],
                    [0.1, 0.99],
                ]
            ),
            "contrastive_row_indices": torch.tensor([0, 1, 2, 3]),
        }
    )

    metrics = evaluate_task_metrics(
        model=model,
        loader=[batch],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    image_metrics = metrics["image_text_pair"]
    assert image_metrics["recall_at_1"] == 1.0
    assert abs(image_metrics["embedding_similarity_mean"] - 1.0) < 1e-2


def test_single_pair_omits_undefined_negative_margin() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["audio-pair"],
        task_types=["audio_text_pair"],
        text_mlm_targets=None,
    )
    model = _StoredOutputsModel(
        {
            "text_embedding": torch.tensor([[1.0, 0.0]]),
            "audio_embedding": torch.tensor([[1.0, 0.0]]),
            "contrastive_row_indices": torch.tensor([0]),
        }
    )

    metrics = evaluate_task_metrics(
        model=model,
        loader=[batch],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    audio_metrics = metrics["audio_text_pair"]
    assert audio_metrics["recall_at_1"] == 1.0
    assert "positive_negative_margin" not in audio_metrics


def test_mlm_metrics_count_only_text_pretrain_and_ignore_ignore_label() -> (
    None
):
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    targets = torch.tensor(
        [
            [IGNORE_LABEL, 5, 5, 5],
            [7, 7, 7, 7],
            [4, IGNORE_LABEL, 4, 4],
        ]
    )
    predictions = torch.tensor(
        [
            [0, 6, 5, 5],
            [7, 7, 7, 7],
            [4, 0, 4, 4],
        ]
    )
    vocabulary_size = 16
    logits = torch.zeros((3, 4, vocabulary_size))
    logits.scatter_(2, predictions.unsqueeze(-1), 1.0)

    batch = _task_batch(
        sample_ids=["pretrain-a", "pretrain-b", "pretrain-c"],
        task_types=["text_pretrain", "text_pretrain", "text_pretrain"],
        text_mlm_targets=targets,
    )

    metrics = evaluate_task_metrics(
        model=_StoredOutputsModel({"text_mlm_logits": logits}),
        loader=[batch],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    # Valid positions: row 0 has 3 valid, row 1 has 4 valid, row 2 has 3 valid = 10 total
    # Correct: row 0 has 2 correct (positions 2,3), row 1 has 4 correct, row 2 has 3 correct = 9 total
    assert metrics["text_pretrain"]["masked_token_accuracy"] == 9 / 10


def test_partial_row_selection_keeps_batch_fields_aligned() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    seen_batches: list[CollatedBatch] = []

    class SelectingGenerationModel(torch.nn.Module):
        def forward(self, batch: CollatedBatch) -> dict[str, torch.Tensor]:
            return {}

        def generate(
            self, batch: CollatedBatch, **kwargs: object
        ) -> torch.Tensor:
            seen_batches.append(batch)
            assert kwargs["temperature"] == 0.0
            assert batch.sample_ids == ["question-row"]
            assert batch.text.shape == (1, 2)
            assert batch.layout_box_targets is not None
            assert batch.layout_box_targets.shape == (1, 2, 4)
            return torch.tensor([[10, 11, 3]])

    batch = _task_batch(
        sample_ids=["question-row"],
        task_types=["doc_qa"],
        text_mlm_targets=None,
        decoder_labels=torch.tensor(
            [
                [IGNORE_LABEL, 10, 11, 3],
            ]
        ),
        layout_box_targets=torch.zeros((1, 2, 4)),
        prompt_token_count=[1],
    )

    metrics = evaluate_task_metrics(
        model=SelectingGenerationModel(),
        loader=[batch],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    assert seen_batches
    assert metrics["doc_qa"]["exact_match"] == 1.0


def test_invalid_generated_batch_shape_fails() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    class WrongShapeModel(torch.nn.Module):
        def forward(self, batch: CollatedBatch) -> dict[str, torch.Tensor]:
            return {}

        def generate(
            self, batch: CollatedBatch, **kwargs: object
        ) -> torch.Tensor:
            return torch.tensor([[10, 11, 3], [1, 2, 3]])

    batch = _task_batch(
        sample_ids=["question-row"],
        task_types=["doc_qa"],
        text_mlm_targets=None,
        decoder_labels=torch.tensor([[IGNORE_LABEL, 10, 11, 3]]),
    )

    with pytest.raises(ValueError, match="does not match selected rows"):
        evaluate_task_metrics(
            model=WrongShapeModel(),
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_sequence_generation_must_return_tensor() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    class BadReturnModel(torch.nn.Module):
        def forward(self, batch: CollatedBatch) -> dict[str, torch.Tensor]:
            return {}

        def generate(
            self, batch: CollatedBatch, **kwargs: object
        ) -> list[list[int]]:
            return [[10, 11, 3]]

    batch = _task_batch(
        sample_ids=["question-row"],
        task_types=["doc_qa"],
        text_mlm_targets=None,
        decoder_labels=torch.tensor([[IGNORE_LABEL, 10, 11, 3]]),
    )

    with pytest.raises(TypeError, match="must return a"):
        evaluate_task_metrics(
            model=BadReturnModel(),
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_select_batch_rows_rejects_out_of_range_rows() -> None:
    from mmcrawler_datasets.collation.multimodal import select_batch_rows

    batch = _task_batch(
        sample_ids=["a", "b"],
        task_types=["doc_qa", "doc_qa"],
        text_mlm_targets=None,
    )

    with pytest.raises(IndexError, match="outside the batch"):
        select_batch_rows(batch=batch, rows=[0, 2])


def test_mlm_logits_rows_must_match_batch() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["pretrain-a", "pretrain-b", "pretrain-c"],
        task_types=["text_pretrain", "text_pretrain", "text_pretrain"],
        text_mlm_targets=torch.zeros((3, 4), dtype=torch.long),
    )
    model = _StoredOutputsModel({"text_mlm_logits": torch.zeros((2, 4, 16))})

    with pytest.raises(ValueError, match="must match the batch"):
        evaluate_task_metrics(
            model=model,
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_mlm_targets_rows_must_match_batch() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["pretrain-a", "pair", "pretrain-b"],
        task_types=["text_pretrain", "image_text_pair", "text_pretrain"],
        text_mlm_targets=torch.zeros((2, 4), dtype=torch.long),
    )
    model = _StoredOutputsModel({"text_mlm_logits": torch.zeros((3, 4, 16))})

    with pytest.raises(ValueError, match="must match the batch"):
        evaluate_task_metrics(
            model=model,
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_task_metric_batch_must_be_collated_batch() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    with pytest.raises(TypeError, match="CollatedBatch"):
        evaluate_task_metrics(
            model=_StoredOutputsModel({}),
            loader=[
                {
                    "task_types": ["image_text_pair"],
                    "text_embedding": torch.eye(1),
                }
            ],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_task_metric_batch_must_not_be_empty() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_task_metrics(
            model=_StoredOutputsModel({}),
            loader=[
                _task_batch(
                    sample_ids=[],
                    task_types=[],
                    text_mlm_targets=None,
                )
            ],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_task_metric_batch_requires_aligned_task_types() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    with pytest.raises(ValueError, match="align with sample_ids"):
        evaluate_task_metrics(
            model=_StoredOutputsModel({}),
            loader=[
                _task_batch(
                    sample_ids=["a", "b"],
                    task_types=["image_text_pair"],
                    text_mlm_targets=None,
                )
            ],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_task_metric_model_outputs_must_be_mapping() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    class TensorOutputModel(torch.nn.Module):
        def forward(self, batch: object) -> torch.Tensor:
            return torch.zeros(1)

    with pytest.raises(TypeError, match="mapping"):
        evaluate_task_metrics(
            model=TensorOutputModel(),
            loader=[
                _task_batch(
                    sample_ids=["pair"],
                    task_types=["image_text_pair"],
                    text_mlm_targets=None,
                )
            ],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_task_metric_evaluation_requires_loader() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    with pytest.raises(RuntimeError, match="evaluation loader is required"):
        evaluate_task_metrics(
            model=_StoredOutputsModel({}),
            loader=None,
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_task_metric_evaluation_restores_training_mode() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    model = _StoredOutputsModel(
        {
            "text_embedding": torch.eye(1),
            "image_embedding": torch.eye(1),
            "contrastive_row_indices": torch.tensor([0]),
        }
    )
    model.train()

    evaluate_task_metrics(
        model=model,
        loader=[
            _task_batch(
                sample_ids=["pair"],
                task_types=["image_text_pair"],
                text_mlm_targets=None,
            )
        ],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    assert model.training is True


def test_task_metric_evaluation_restores_training_mode_after_exception() -> (
    None
):
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    class ExplodingModel(torch.nn.Module):
        def forward(self, batch: object) -> dict[str, torch.Tensor]:
            raise RuntimeError("boom")

    model = ExplodingModel()
    model.train()

    with pytest.raises(RuntimeError, match="boom"):
        evaluate_task_metrics(
            model=model,
            loader=[
                _task_batch(
                    sample_ids=["pair"],
                    task_types=["image_text_pair"],
                    text_mlm_targets=None,
                )
            ],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )

    assert model.training is True


class _ScriptedDist:
    """Two-rank torch.distributed stand-in with scripted remote inputs."""

    def __init__(
        self,
        *,
        world_size: int,
        remote_inputs: Sequence[torch.Tensor],
    ) -> None:
        self.world_size = world_size
        self.remote_inputs = list(remote_inputs)
        self.all_gather_calls: list[torch.Tensor] = []
        self.all_reduce_calls: list[torch.Tensor] = []

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def get_world_size(self) -> int:
        return self.world_size

    def get_backend(self) -> str:
        return "gloo"

    def all_gather(
        self, tensor_list: list[torch.Tensor], tensor: torch.Tensor
    ) -> None:
        self.all_gather_calls.append(tensor)
        for index, target in enumerate(tensor_list):
            if index == 0:
                target.copy_(tensor)
            else:
                target.copy_(self.remote_inputs.pop(0))

    def all_reduce(self, tensor: torch.Tensor, op: object) -> None:
        self.all_reduce_calls.append(tensor)
        tensor.mul_(self.world_size)

    class ReduceOp:
        SUM = 0


def _remote_meta(count: int, dim: int, valid: int = 1) -> torch.Tensor:
    return torch.tensor([count, dim, valid], dtype=torch.long)


def test_rouge_l_returns_zero_without_overlap() -> None:
    from evaluator.metrics.sequence.primitives import _rouge_l_f1

    assert _rouge_l_f1(prediction=["cat"], target=["dog"]) == 0.0


def test_disjoint_sequence_tokens_do_not_crash() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    class DisjointGenerationModel(torch.nn.Module):
        def forward(self, batch: CollatedBatch) -> dict[str, torch.Tensor]:
            return {}

        def generate(
            self, batch: CollatedBatch, **kwargs: object
        ) -> torch.Tensor:
            return torch.tensor([[20, 21]])

    batch = _task_batch(
        sample_ids=["doc-qa"],
        task_types=["doc_qa"],
        text_mlm_targets=None,
        decoder_labels=torch.tensor([[IGNORE_LABEL, 10, 11]]),
    )

    metrics = evaluate_task_metrics(
        model=DisjointGenerationModel(),
        loader=[batch],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )["doc_qa"]

    assert metrics["exact_match"] == 0.0
    assert metrics["token_f1"] == 0.0


def test_decoder_labels_rows_must_match_batch() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["a", "b"],
        task_types=["doc_qa", "doc_qa"],
        text_mlm_targets=None,
        decoder_labels=torch.tensor([[IGNORE_LABEL, 10, 11]]),
    )

    with pytest.raises(ValueError, match="must match the batch"):
        evaluate_task_metrics(
            model=_StoredOutputsModel({}),
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_pair_gather_reconstructs_aligned_pairs_across_ranks() -> None:
    from evaluator.distributed.collectives import gather_pair_embeddings

    local_text = [torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])]
    local_media = [torch.tensor([0.5, 0.5]), torch.tensor([0.1, 0.9])]
    remote_text_rows = torch.tensor([[3.0, 0.0]] * 5)
    remote_media_rows = torch.tensor([[0.2, 0.8]] * 5)
    remote_inputs = [
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        _remote_meta(5, 2),
        remote_text_rows,
        remote_media_rows,
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        _remote_meta(0, 0),
    ]
    fake = _ScriptedDist(world_size=2, remote_inputs=remote_inputs)

    text, media = gather_pair_embeddings(
        text_embeddings_by_task={"image_text_pair": local_text},
        media_embeddings_by_task={"image_text_pair": local_media},
        device=None,
        dist_module=fake,
    )

    gathered_text = text["image_text_pair"]
    gathered_media = media["image_text_pair"]
    assert sum(tensor.shape[0] for tensor in gathered_text) == 7
    assert sum(tensor.shape[0] for tensor in gathered_media) == 7
    assert all(
        left.shape == right.shape
        for left, right in zip(gathered_text, gathered_media, strict=True)
    )
    assert gathered_text[0][0].tolist() == [1.0, 0.0]


def test_pair_gather_runs_fixed_collectives_with_uneven_task_presence() -> (
    None
):
    from evaluator.distributed.collectives import gather_pair_embeddings

    local_text = [torch.tensor([1.0, 0.0])]
    local_media = [torch.tensor([0.5, 0.5])]
    remote_inputs = [
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        torch.zeros((1, 2)),
        torch.zeros((1, 2)),
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        _remote_meta(0, 0),
    ]
    fake = _ScriptedDist(world_size=2, remote_inputs=remote_inputs)

    text, media = gather_pair_embeddings(
        text_embeddings_by_task={"image_text_pair": local_text},
        media_embeddings_by_task={"image_text_pair": local_media},
        device=None,
        dist_module=fake,
    )

    assert len(text["image_text_pair"]) == 1
    assert len(media["image_text_pair"]) == 1
    assert len(fake.all_gather_calls) == 9


def test_pair_gather_rejects_rank_dimension_mismatch() -> None:
    from evaluator.distributed.collectives import gather_pair_embeddings

    local_text = [torch.tensor([1.0, 0.0])]
    local_media = [torch.tensor([0.5, 0.5])]
    remote_inputs = [
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        _remote_meta(1, 256),
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        _remote_meta(0, 0),
    ]
    fake = _ScriptedDist(world_size=2, remote_inputs=remote_inputs)

    with pytest.raises(ValueError, match="dimension mismatch"):
        gather_pair_embeddings(
            text_embeddings_by_task={"image_text_pair": local_text},
            media_embeddings_by_task={"image_text_pair": local_media},
            device=None,
            dist_module=fake,
        )


def test_pair_gather_rejects_invalid_remote_rank_metadata() -> None:
    from evaluator.distributed.collectives import gather_pair_embeddings

    local_text = [torch.tensor([1.0, 0.0])]
    local_media = [torch.tensor([0.5, 0.5])]
    remote_inputs = [
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        _remote_meta(0, 0, valid=0),
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        _remote_meta(0, 0),
        _remote_meta(0, 0),
    ]
    fake = _ScriptedDist(world_size=2, remote_inputs=remote_inputs)

    with pytest.raises(ValueError, match="invalid local pair embedding"):
        gather_pair_embeddings(
            text_embeddings_by_task={"image_text_pair": local_text},
            media_embeddings_by_task={"image_text_pair": local_media},
            device=None,
            dist_module=fake,
        )


def test_pair_output_contracts_are_fail_closed() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["pair"],
        task_types=["image_text_pair"],
        text_mlm_targets=None,
    )

    cases = [
        (
            {
                "text_embedding": torch.tensor([1.0, 0.0]),
                "contrastive_row_indices": torch.tensor([0]),
            },
            ValueError,
        ),
        (
            {
                "text_embedding": torch.eye(1),
                "contrastive_row_indices": torch.tensor([[0]]),
            },
            ValueError,
        ),
        (
            {
                "text_embedding": torch.eye(1),
                "contrastive_row_indices": torch.tensor([0.0]),
            },
            TypeError,
        ),
        (
            {
                "text_embedding": torch.eye(1),
                "image_embedding": torch.tensor([1.0, 0.0]),
                "contrastive_row_indices": torch.tensor([0]),
            },
            ValueError,
        ),
    ]
    for outputs, error_type in cases:
        with pytest.raises(error_type):
            evaluate_task_metrics(
                model=_StoredOutputsModel(outputs),
                loader=[batch],
                device=torch.device("cpu"),
                autocast_factory=nullcontext,
            )


def test_sequence_statistics_reduce_sums_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evaluator.distributed.reductions as reductions
    from evaluator.metric_contracts import (
        SEQUENCE_STAT_FIELDS,
        SequenceStatistics,
    )

    fake = _ScriptedDist(world_size=2, remote_inputs=[])
    monkeypatch.setattr(reductions, "dist", fake)

    statistics = {
        "doc_qa": SequenceStatistics(
            **{field: 1.0 for field in SEQUENCE_STAT_FIELDS}
        )
    }
    reductions.reduce_sequence_statistics(
        statistics=statistics,
        device=torch.device("cpu"),
        evaluation_method="exact_match_f1",
    )

    assert all(
        getattr(statistics["doc_qa"], field) == 2.0
        for field in SEQUENCE_STAT_FIELDS
    )
    assert len(fake.all_reduce_calls) == 1


class _PairIdentityModel(torch.nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, batch: CollatedBatch) -> dict[str, torch.Tensor]:
        count = len(batch.task_types)
        embedding = torch.eye(self.embedding_dim)[:count].float()
        pair_rows = [
            index
            for index, task_type in enumerate(batch.task_types)
            if task_type in PAIR_TASKS
        ]
        return {
            "text_embedding": embedding,
            "image_embedding": embedding,
            "audio_embedding": embedding,
            "contrastive_row_indices": torch.tensor(
                pair_rows, dtype=torch.long
            ),
        }


def _run_task_metric_ddp_rank(
    rank: int,
    *,
    world_size: int,
    port: int,
    rank_task_types: Sequence[str],
    output: object,
) -> None:
    import os

    os.environ["USE_LIBUV"] = "0"

    from contextlib import nullcontext

    import torch.distributed as dist

    from evaluator.task_metrics import evaluate_task_metrics

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        batch = _task_batch(
            sample_ids=[f"rank-{rank}-{index}" for index in rank_task_types],
            task_types=list(rank_task_types),
            text_mlm_targets=None,
        )
        metrics = evaluate_task_metrics(
            model=_PairIdentityModel(embedding_dim=3),
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )
        output.put((rank, metrics))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "rank_one_task_types",
    [["audio_text_pair"]],
)
def test_distributed_ranks_finish_with_matching_metrics(
    rank_one_task_types: list[str],
) -> None:
    import multiprocessing as mp
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    queue = mp.get_context("spawn").Queue()
    rank_task_types = {
        0: ["image_text_pair", "image_text_pair"],
        1: rank_one_task_types,
    }
    processes = [
        mp.get_context("spawn").Process(
            target=_run_task_metric_ddp_rank,
            args=(),
            kwargs={
                "rank": rank,
                "world_size": 2,
                "port": port,
                "rank_task_types": rank_task_types[rank],
                "output": queue,
            },
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(240)

    assert all(process.exitcode == 0 for process in processes), [
        process.exitcode for process in processes
    ]
    results = dict(queue.get() for _ in range(2))
    assert results[0] == results[1]
    assert results[0]["image_text_pair"]["recall_at_1"] == 1.0


def test_pair_coverage_requires_all_pair_rows_evaluated() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["pair-0", "pair-1"],
        task_types=["image_text_pair", "image_text_pair"],
        text_mlm_targets=None,
    )
    model = _StoredOutputsModel(
        {
            "text_embedding": torch.eye(2),
            "image_embedding": torch.eye(2),
            "contrastive_row_indices": torch.tensor([0]),
        }
    )

    with pytest.raises(
        ValueError, match="pair metric coverage does not match"
    ):
        evaluate_task_metrics(
            model=model,
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_pair_coverage_rejects_duplicate_row_indices() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["pair-0", "pair-1"],
        task_types=["image_text_pair", "image_text_pair"],
        text_mlm_targets=None,
    )
    model = _StoredOutputsModel(
        {
            "text_embedding": torch.eye(2),
            "image_embedding": torch.eye(2),
            "contrastive_row_indices": torch.tensor([0, 0]),
        }
    )

    with pytest.raises(ValueError, match="duplicate rows"):
        evaluate_task_metrics(
            model=model,
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_pair_task_requires_resolvable_modality() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["pair-unknown"],
        task_types=["multimodal_retrieval"],
        text_mlm_targets=None,
    )
    model = _StoredOutputsModel(
        {
            "text_embedding": torch.eye(1),
            "contrastive_row_indices": torch.tensor([0]),
        }
    )

    with pytest.raises(ValueError, match="unable to resolve modality"):
        evaluate_task_metrics(
            model=model,
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_pair_task_requires_media_embedding() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["pair-img"],
        task_types=["image_text_pair"],
        text_mlm_targets=None,
    )
    model = _StoredOutputsModel(
        {
            "text_embedding": torch.eye(1),
            "contrastive_row_indices": torch.tensor([0]),
        }
    )

    with pytest.raises(ValueError, match="missing image_embedding"):
        evaluate_task_metrics(
            model=model,
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_sequence_task_rejects_all_ignore_label_rows() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["doc-qa-0", "doc-qa-1"],
        task_types=["doc_qa", "doc_qa"],
        text_mlm_targets=None,
        decoder_labels=torch.tensor(
            [
                [IGNORE_LABEL, 10, 11, 3],
                [IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL],
            ]
        ),
    )

    class GenerateModel(torch.nn.Module):
        def forward(self, batch: CollatedBatch) -> dict[str, torch.Tensor]:
            return {}

        def generate(
            self, batch: CollatedBatch, **kwargs: object
        ) -> torch.Tensor:
            return torch.tensor([[10, 11, 3]])

    with pytest.raises(ValueError, match="no target labels"):
        evaluate_task_metrics(
            model=GenerateModel(),
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_sequence_task_requires_model_generate() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["doc-qa"],
        task_types=["doc_qa"],
        text_mlm_targets=None,
        decoder_labels=torch.tensor([[IGNORE_LABEL, 10, 11, 3]]),
    )

    class NoGenerateModel(torch.nn.Module):
        def forward(self, batch: CollatedBatch) -> dict[str, torch.Tensor]:
            return {}

    with pytest.raises(TypeError, match="requires model.generate"):
        evaluate_task_metrics(
            model=NoGenerateModel(),
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_pair_task_requires_text_embedding() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["pair"],
        task_types=["image_text_pair"],
        text_mlm_targets=None,
    )
    model = _StoredOutputsModel(
        {
            "contrastive_row_indices": torch.tensor([0]),
            "image_embedding": torch.eye(1),
        }
    )

    with pytest.raises(
        ValueError, match="pair task batch is missing text_embedding"
    ):
        evaluate_task_metrics(
            model=model,
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_pair_task_requires_contrastive_row_indices() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["pair"],
        task_types=["image_text_pair"],
        text_mlm_targets=None,
    )
    model = _StoredOutputsModel(
        {
            "text_embedding": torch.eye(1),
            "image_embedding": torch.eye(1),
        }
    )

    with pytest.raises(
        ValueError, match="pair task batch is missing contrastive_row_indices"
    ):
        evaluate_task_metrics(
            model=model,
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_sequence_task_requires_decoder_labels() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["doc-qa"],
        task_types=["doc_qa"],
        text_mlm_targets=None,
        decoder_labels=None,
    )

    class GenerateModel(torch.nn.Module):
        def forward(self, batch: CollatedBatch) -> dict[str, torch.Tensor]:
            return {}

        def generate(
            self, batch: CollatedBatch, **kwargs: object
        ) -> torch.Tensor:
            return torch.tensor([[10, 11, 3]])

    with pytest.raises(
        ValueError, match="sequence task batch is missing decoder_labels"
    ):
        evaluate_task_metrics(
            model=GenerateModel(),
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_text_pretrain_requires_mlm_targets() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["pretrain"],
        task_types=["text_pretrain"],
        text_mlm_targets=None,
    )
    model = _StoredOutputsModel({})

    with pytest.raises(
        ValueError, match="text_pretrain evaluation requires text_mlm_targets"
    ):
        evaluate_task_metrics(
            model=model,
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_text_pretrain_requires_mlm_logits() -> None:
    from contextlib import nullcontext

    from evaluator.task_metrics import evaluate_task_metrics

    batch = _task_batch(
        sample_ids=["pretrain"],
        task_types=["text_pretrain"],
        text_mlm_targets=torch.zeros((1, 4), dtype=torch.long),
    )
    model = _StoredOutputsModel({})

    with pytest.raises(
        ValueError, match="text_pretrain evaluation requires text_mlm_logits"
    ):
        evaluate_task_metrics(
            model=model,
            loader=[batch],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_release_tasks_metrics_exist_in_evaluator_output() -> None:
    """Verify that every metric in release.tasks exists in evaluator output."""
    from config.load import load_settings

    settings = load_settings(
        "prod",
        env={
            "DATA_ENGINE_PROJECT_ROOT": "C:/Users/abnor/Downloads/pro (2)",
            "APP_OVERRIDE__preprocessing.transcription.model_name": "/models/whisper",
            "APP_OVERRIDE__preprocessing.transcription.model_revision": "test",
            "APP_OVERRIDE__preprocessing.transcription.model_artifact_hash": "0"
            * 64,
            "APP_OVERRIDE__preprocessing.transcription.backend_version": "test",
        },
    )

    # Expected evaluator output metrics per task (from evaluator/task_metrics.py)
    evaluator_metrics = {
        "text_pretrain": {"masked_token_accuracy"},
        "instruction_following": {
            "token_f1",
            "exact_match",
            "sequence_exact_match",
            "sequence_token_accuracy",
        },
        "document_text_pair": {
            "recall_at_1",
            "embedding_similarity_mean",
            "recall_at_5",
            "positive_negative_margin",
        },
        "pdf_text_pair": {
            "recall_at_1",
            "embedding_similarity_mean",
            "recall_at_5",
            "positive_negative_margin",
        },
        "doc_qa": {
            "doc_qa_f1",
            "token_f1",
            "exact_match",
            "sequence_token_accuracy",
        },
        "image_text_pair": {
            "recall_at_1",
            "embedding_similarity_mean",
            "recall_at_5",
            "positive_negative_margin",
        },
        "ocr_parse": {
            "character_error_rate",
            "word_error_rate",
            "layout_box_mse",
        },
        "vqa": {"vqa_accuracy", "exact_match"},
        "audio_text_pair": {
            "recall_at_1",
            "embedding_similarity_mean",
            "recall_at_5",
            "positive_negative_margin",
        },
        "audio_qa": {
            "token_f1",
            "exact_match",
            "sequence_exact_match",
            "sequence_token_accuracy",
        },
        "video_text_pair": {
            "recall_at_1",
            "embedding_similarity_mean",
            "recall_at_5",
            "positive_negative_margin",
        },
        "video_qa": {
            "token_f1",
            "exact_match",
            "sequence_exact_match",
            "sequence_token_accuracy",
        },
        "multimodal_retrieval": {
            "recall_at_1",
            "embedding_similarity_mean",
            "recall_at_5",
            "positive_negative_margin",
        },
        "cross_modal_consistency": {
            "recall_at_1",
            "embedding_similarity_mean",
            "recall_at_5",
            "positive_negative_margin",
        },
    }

    for task in settings.release.tasks:
        task_name = task.name
        for metric in task.metrics:
            assert metric.name in evaluator_metrics[task_name], (
                f"Metric {metric.name} for task {task_name} is not produced by evaluator. "
                f"Available: {evaluator_metrics[task_name]}"
            )

    # Verify OCR uses max, others use min
    for task in settings.release.tasks:
        for metric in task.metrics:
            if metric.name in ("character_error_rate", "word_error_rate"):
                assert metric.max is not None, (
                    f"OCR metric {metric.name} should use max"
                )
                assert metric.min is None, (
                    f"OCR metric {metric.name} should not use min"
                )
            else:
                assert metric.min is not None, (
                    f"Metric {metric.name} should use min"
                )
                assert metric.max is None, (
                    f"Metric {metric.name} should not use max"
                )
