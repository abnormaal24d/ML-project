from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch

from augmentation.text.text_field_augmenter import TextFieldAugmenter
from augmentation.text.text_variant_assembler import TextVariantAssembler
from config.augmentation.augmentation_settings import AugmentationSettings
from config.multimodal.encoder_settings import EncoderSettings
from config.multimodal.model_settings import ModelSettings
from config.multimodal.training_head_settings import DecoderSettings
from config.multimodal.training_settings import TrainingSettings
from datachecker.workflow_decision import (
    WorkflowAction,
    WorkflowDecisionReason,
    WorkflowExecutionPlan,
)
from mmcrawler_datasets.dataloader import build_dataloader
from mmcrawler_datasets.schema import DatasetSplit
from mmcrawler_datasets.training_samples.snapshot_mapping import (
    build_snapshot_sample,
    serialize_snapshot_sample,
)
from multimodal.tokenization.text import VocabularyTokenizer
from multimodal.tokenization.training import train_vocabulary_tokenizer
from orchestration.bootstrap.workflow_executor import (
    EXIT_SUCCESS,
    WorkflowPhaseExecutor,
)
from orchestration.composition.runtime.training import (
    build_model,
    build_training_loss,
)
from orchestration.workflow.phase import (
    PhaseOutcome,
    PhaseStatus,
)
from preprocessing.text.text_preparation import normalize_text
from training.runtime.optimization import build_lr_scheduler, build_optimizer
from training.runtime.trainer import MultimodalTrainer


class _Logger:
    def debug(self, *_args: object, **_kwargs: object) -> None:
        return None

    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def error(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None


class _Checker:
    def __init__(self, actions: Iterable[WorkflowAction]) -> None:
        self._actions = iter(actions)

    def check(self, *, timeout_seconds: float) -> WorkflowExecutionPlan:
        assert timeout_seconds > 0
        action = next(self._actions)
        reason = {
            WorkflowAction.CRAWL: WorkflowDecisionReason.CRAWL_OUTPUT_MISSING,
            WorkflowAction.PREPROCESS: (
                WorkflowDecisionReason.PREPROCESSING_OUTPUT_MISSING
            ),
            WorkflowAction.AUGMENT: (
                WorkflowDecisionReason.AUGMENTATION_OUTPUT_MISSING
            ),
            WorkflowAction.TRAIN: (
                WorkflowDecisionReason.TRAINING_OUTPUT_MISSING
            ),
            WorkflowAction.NOOP: WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
        }[action]
        return WorkflowExecutionPlan(
            action=action,
            reason=reason,
            details=(),
        )


class _PhaseRunner:
    def __init__(self, *, action: WorkflowAction, root: Path) -> None:
        self._action = action
        self._root = root

    async def __call__(self, plan: WorkflowExecutionPlan) -> PhaseOutcome:
        if plan.action is not self._action:
            raise AssertionError(
                f"runner {self._action} received action {plan.action}"
            )
        artifact = self._root / f"{self._action.value}.complete"
        artifact.write_text(self._action.value + "\n", encoding="utf-8")
        return PhaseOutcome(status=PhaseStatus.SUCCEEDED)


@pytest.mark.asyncio
async def test_workflow_router_completes_all_planned_actions(
    tmp_path: Path,
) -> None:
    runners = {
        WorkflowAction.CRAWL: _PhaseRunner(
            action=WorkflowAction.CRAWL,
            root=tmp_path,
        ),
        WorkflowAction.PREPROCESS: _PhaseRunner(
            action=WorkflowAction.PREPROCESS,
            root=tmp_path,
        ),
        WorkflowAction.AUGMENT: _PhaseRunner(
            action=WorkflowAction.AUGMENT,
            root=tmp_path,
        ),
        WorkflowAction.TRAIN: _PhaseRunner(
            action=WorkflowAction.TRAIN,
            root=tmp_path,
        ),
    }

    executor = WorkflowPhaseExecutor(
        check=partial(
            _Checker(
                (
                    WorkflowAction.CRAWL,
                    WorkflowAction.PREPROCESS,
                    WorkflowAction.AUGMENT,
                    WorkflowAction.TRAIN,
                    WorkflowAction.NOOP,
                )
            ).check,
            timeout_seconds=5.0,
        ),
        runners=runners,
        max_iterations=5,
        iteration_pause_seconds=0.0,
        logger=_Logger(),
    )

    exit_code = await executor.execute()

    assert exit_code == EXIT_SUCCESS
    assert (tmp_path / "crawl.complete").is_file()
    assert (tmp_path / "preprocess.complete").is_file()
    assert (tmp_path / "augment.complete").is_file()
    assert (tmp_path / "train.complete").is_file()


def test_local_fixture_runs_preprocess_augment_snapshot_and_dense_training(
    tmp_path: Path,
) -> None:
    raw_text = "  Alpha   beta gamma delta.  " * 4
    normalized_text = normalize_text(text=raw_text)
    assert normalized_text == (
        "Alpha beta gamma delta. " * 3 + "Alpha beta gamma delta."
    )

    source_record = {
        "schema_version": "3.0",
        "sample_id": "local-source",
        "record_id": "local-record",
        "modality": "text",
        "text": normalized_text,
        "title": "Offline fixture",
        "objects": [],
        "task_target": {
            "task_type": "text_pretrain",
            "task_family": "text",
            "output_modalities": ["text"],
        },
    }
    source_sample = build_snapshot_sample(
        payload=source_record,
        dataset_root=tmp_path,
        source_path=tmp_path / "ingested.jsonl",
        line_number=1,
    )

    augmentation_settings = AugmentationSettings(
        cache_enabled=False,
        text={
            "minimum_text_length": 1,
            "maximum_text_length": 512,
            "max_variants_per_sample": 1,
            "title_prefix_enabled": True,
            "context_prefix_enabled": False,
            "text_span_focus_enabled": False,
        },
    )
    variant_assembler = TextVariantAssembler(
        settings=augmentation_settings,
        logger=_Logger(),
    )
    variants = TextFieldAugmenter(
        settings=augmentation_settings,
        variant_assembler=variant_assembler,
        logger=_Logger(),
    ).augment(sample=source_sample)
    assert len(variants) == 1
    variant = variants[0][1]
    assert variant.text is not None
    assert variant.text.startswith("Offline fixture")
    assert variant.metadata["augmentation_type"] == "text_field"

    tokenizer_path = tmp_path / "tokenizer.json"
    train_vocabulary_tokenizer(
        output_path=tokenizer_path,
        vocab_size=269,
        texts=(variant.text, "alpha beta gamma delta"),
        snapshot_id="local-e2e",
    )
    tokenizer = VocabularyTokenizer.load(tokenizer_path, max_tokens=32)
    tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()

    dataset_root = tmp_path / "dataset"
    splits = dataset_root / "splits"
    tensors = dataset_root / "training_tensors"
    splits.mkdir(parents=True)
    tensors.mkdir(parents=True)
    text_tensor_path = tensors / "local-source.pt"
    torch.save(
        torch.tensor(tokenizer.encode(variant.text), dtype=torch.long),
        text_tensor_path,
    )
    materialized = replace(
        variant,
        text_tokens_path=text_tensor_path,
    )
    serialized = serialize_snapshot_sample(
        sample=materialized,
        dataset_root=dataset_root,
    )
    assert serialized["task_target"]["text_tokens_path"] == (
        "training_tensors/local-source.pt"
    )

    for split in ("train", "val", "test"):
        split_record = dict(serialized)
        split_record["sample_id"] = f"{materialized.sample_id}-{split}"
        split_record["record_id"] = f"{materialized.record_id}-{split}"
        (splits / f"{split}.jsonl").write_text(
            json.dumps(split_record, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (dataset_root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "splits": {"train": 1, "val": 1, "test": 1},
                "paths": {
                    "splits": {
                        "train": "splits/train.jsonl",
                        "val": "splits/val.jsonl",
                        "test": "splits/test.jsonl",
                    }
                },
                "outputs": {
                    "jsonl": True,
                    "shards": False,
                    "shard_format": None,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    dataset_manifest_sha256 = hashlib.sha256(
        (dataset_root / "dataset_manifest.json").read_bytes()
    ).hexdigest()

    encoder = EncoderSettings(
        input_dim=16,
        hidden_dim=16,
        output_dim=16,
        dropout=0.0,
    )
    model_settings = ModelSettings(
        text=encoder,
        document=encoder,
        image=encoder,
        audio=encoder,
        video=encoder,
        fusion_dim=16,
        projection_dim=16,
        raw_text_vocab_size=269,
        raw_text_max_tokens=32,
        enabled_modalities=("text",),
        output_modalities=("text",),
        text_decoder=DecoderSettings(
            enabled=True,
            vocab_size=269,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_target_tokens=32,
            max_context_tokens=32,
            max_text_context_tokens=16,
            max_document_context_tokens=4,
            max_image_context_tokens=4,
            max_audio_context_tokens=4,
            max_video_context_tokens=4,
        ),
    )
    training_settings = TrainingSettings(
        run_mode="smoke",
        release_stage="learning_candidate",
        training_backend="dense_transformer",
        text_tokenizer_max_tokens=32,
        text_tokenizer_path=str(tokenizer_path),
        text_tokenizer_sha256=tokenizer_sha256,
        text_tokenizer_vocab_size=269,
        tasks=("text_pretrain",),
        curriculum_schedule=("text_pretrain",),
        task_sampling_weights={"text_pretrain": 1.0},
        min_task_samples={"text_pretrain": 1},
        disable_undercovered_tasks=False,
        dynamic_sampling=False,
        task_aware_batching=False,
        batch_size=1,
        epochs=1,
        lr_scheduler="none",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        device="cpu",
        precision="fp32",
        distributed_strategy="none",
        export_artifacts=False,
        materialized_tensors_enabled=True,
        min_split_items=1,
        deterministic=True,
        offline=True,
        mlm_probability=1.0,
        mlm_loss_weight=1.0,
        language_modeling_loss_weight=0.0,
        image_patch_loss_weight=0.0,
        audio_masked_loss_weight=0.0,
        video_temporal_loss_weight=0.0,
        use_hard_negative_sampler=False,
    )
    _dataset, loader = build_dataloader(
        dataset_root=dataset_root,
        split=DatasetSplit.TRAIN,
        model_settings=model_settings,
        training_settings=training_settings,
        tokenizer=tokenizer,
        logger=_Logger(),
        distributed=False,
    )
    collated = next(iter(loader))
    assert collated.task_types == ["text_pretrain"]
    assert collated.text_mlm_targets is not None
    assert collated.decoder_input_ids.shape == (1, 0)

    result = MultimodalTrainer(
        model_settings=model_settings,
        training_settings=training_settings,
        tokenizer=tokenizer,
        model_exporter=lambda **_kwargs: {},
        logger=_Logger(),
        training_backend="dense_transformer",
        model_factory=build_model,
        loss_factory=build_training_loss,
        optimizer_factory=build_optimizer,
        scheduler_factory=build_lr_scheduler,
        project_root=Path.cwd(),
    ).train(
        dataset_root=dataset_root,
        checkpoint_path=tmp_path / "local-e2e.pt",
        export_directory=tmp_path / "exports",
        dataset_manifest_sha256=dataset_manifest_sha256,
    )

    assert result.metrics.batches == 1
    assert result.metrics.effective_task_counts == {"text_pretrain": 1}
    assert (
        result.metrics.training_signal_by_modality["text"]["updated"] is True
    )
