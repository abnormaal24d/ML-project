from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from config.multimodal.encoder_settings import EncoderSettings
from config.multimodal.model_settings import ModelSettings
from config.multimodal.training_head_settings import DecoderSettings
from config.multimodal.training_settings import TrainingSettings
from multimodal.tokenization.text import VocabularyTokenizer
from multimodal.tokenization.training import train_vocabulary_tokenizer
from orchestration.composition.runtime.training import (
    build_model,
    build_training_loss,
)
from training.runtime.checkpoint.contract import (
    CheckpointContract,
    checkpoint_headers_present,
)
from training.runtime.checkpoint.io import (
    checkpoint_checksum_path,
    safe_torch_load,
)
from training.runtime.optimization import build_lr_scheduler, build_optimizer
from training.runtime.trainer import MultimodalTrainer


class _Logger:
    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def debug(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None


def _write_training_dataset(
    *, root: Path, tokenizer: VocabularyTokenizer
) -> None:
    splits = root / "splits"
    tensors = root / "training_tensors"
    splits.mkdir(parents=True)
    tensors.mkdir(parents=True)

    for split in ("train", "val", "test"):
        tensor_path = tensors / f"{split}-text.pt"
        torch.save(
            torch.tensor(
                tokenizer.encode(f"golden path {split} sample"),
                dtype=torch.long,
            ),
            tensor_path,
        )
        record = {
            "schema_version": "3.0",
            "sample_id": f"sample-{split}",
            "record_id": f"record-{split}",
            "text": f"golden path {split} sample",
            "objects": [],
            "task_target": {
                "task_type": "text_pretrain",
                "task_family": "text",
                "text_tokens_path": tensor_path.relative_to(root).as_posix(),
                "output_modalities": [],
            },
        }
        (splits / f"{split}.jsonl").write_text(
            json.dumps(record, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    (root / "dataset_manifest.json").write_text(
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


def test_real_cpu_training_runtime_writes_resumable_checkpoint(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    train_vocabulary_tokenizer(
        output_path=tokenizer_path,
        vocab_size=269,
        texts=("golden path training corpus",),
        snapshot_id="runtime-smoke",
    )
    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer_sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()
    tokenizer = VocabularyTokenizer.load(tokenizer_path, max_tokens=16)

    dataset_root = tmp_path / "dataset"
    _write_training_dataset(root=dataset_root, tokenizer=tokenizer)

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
        raw_text_max_tokens=16,
        raw_image_size=16,
        raw_audio_num_samples=512,
        raw_video_frames=2,
        enabled_modalities=("text",),
        output_modalities=("text",),
    )
    training_settings = TrainingSettings(
        run_mode="smoke",
        release_stage="pipeline_smoke",
        text_tokenizer_max_tokens=16,
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

    trainer = MultimodalTrainer(
        model_settings=model_settings,
        training_settings=training_settings,
        tokenizer=tokenizer,
        model_exporter=lambda **_kwargs: {},
        logger=_Logger(),
        training_backend=training_settings.training_backend,
        model_factory=build_model,
        loss_factory=build_training_loss,
        optimizer_factory=build_optimizer,
        scheduler_factory=build_lr_scheduler,
        project_root=Path.cwd(),
    )
    checkpoint_path = tmp_path / "checkpoints" / "model.pt"
    result = trainer.train(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path,
        export_directory=tmp_path / "exports",
        dataset_manifest_sha256="a" * 64,
    )

    metrics = result.metrics
    assert metrics.epochs == 1
    assert metrics.batches == 1
    assert metrics.samples == 1
    assert metrics.effective_task_counts == {"text_pretrain": 1}
    assert metrics.effective_modality_counts == {"text": 1}
    assert metrics.training_signal_by_modality["text"]["updated"] is True
    assert checkpoint_path.is_file()
    assert checkpoint_checksum_path(checkpoint_path).is_file()

    checkpoint = safe_torch_load(checkpoint_path)
    assert isinstance(checkpoint, dict)
    assert isinstance(checkpoint["optimizer_state"], dict)
    assert checkpoint["training_state"]["epoch"] == 1
    assert checkpoint["training_state"]["global_step"] == 1


def _write_dense_candidate_dataset(
    *,
    root: Path,
    tokenizer: VocabularyTokenizer,
    tasks: tuple[str, ...],
) -> None:
    splits = root / "splits"
    tensors = root / "training_tensors"
    splits.mkdir(parents=True)
    tensors.mkdir(parents=True)

    for split in ("train", "val", "test"):
        records: list[dict[str, object]] = []
        for task in tasks:
            for index, text in enumerate(("alpha beta", "gamma delta")):
                sample_id = f"{split}-{task}-{index}"
                text_path = tensors / f"{sample_id}-text.pt"
                torch.save(
                    torch.tensor(tokenizer.encode(text), dtype=torch.long),
                    text_path,
                )
                target: dict[str, object] = {
                    "task_type": task,
                    "task_family": (
                        "text"
                        if task == "text_pretrain"
                        else task.split("_")[0]
                    ),
                    "target_text": text,
                    "text_tokens_path": text_path.relative_to(root).as_posix(),
                    "output_modalities": (
                        ["text"] if task == "text_pretrain" else ["embedding"]
                    ),
                    "alignment_score": 1.0,
                }
                objects: list[dict[str, object]] = []
                if task == "document_text_pair":
                    objects = [
                        {
                            "object_id": f"{sample_id}-document",
                            "object_url": "https://example.test/source.pdf",
                            "object_mime_type": "application/pdf",
                            "role": "document",
                        }
                    ]
                elif task == "image_text_pair":
                    tensor_path = tensors / f"{sample_id}-image.pt"
                    torch.save(torch.rand(3, 16, 16), tensor_path)
                    target["image_tensor_path"] = tensor_path.relative_to(
                        root
                    ).as_posix()
                    objects = [
                        {
                            "object_id": f"{sample_id}-image",
                            "object_url": "https://example.test/source.png",
                            "object_mime_type": "image/png",
                            "role": "image",
                        }
                    ]
                elif task == "audio_text_pair":
                    tensor_path = tensors / f"{sample_id}-audio.pt"
                    torch.save(torch.rand(1, 512), tensor_path)
                    target["audio_tensor_path"] = tensor_path.relative_to(
                        root
                    ).as_posix()
                    objects = [
                        {
                            "object_id": f"{sample_id}-audio",
                            "object_url": "https://example.test/source.wav",
                            "object_mime_type": "audio/wav",
                            "role": "audio",
                        }
                    ]
                elif task == "video_text_pair":
                    tensor_path = tensors / f"{sample_id}-video.pt"
                    torch.save(torch.rand(2, 3, 16, 16), tensor_path)
                    target["video_tensor_path"] = tensor_path.relative_to(
                        root
                    ).as_posix()
                    objects = [
                        {
                            "object_id": f"{sample_id}-video",
                            "object_url": "https://example.test/source.mp4",
                            "object_mime_type": "video/mp4",
                            "role": "video",
                        }
                    ]
                records.append(
                    {
                        "schema_version": "3.0",
                        "sample_id": sample_id,
                        "record_id": f"record-{sample_id}",
                        "text": text,
                        "objects": objects,
                        "task_target": target,
                    }
                )
        (splits / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(record, sort_keys=True) + "\n" for record in records
            ),
            encoding="utf-8",
        )

    split_size = len(tasks) * 2
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "splits": {
                    "train": split_size,
                    "val": split_size,
                    "test": split_size,
                },
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


def test_dense_backend_trains_core_multimodal_tasks_from_prod_candidate_policy(
    tmp_path: Path,
    production_whisper_env: None,
) -> None:
    from config.load import load_settings

    production = load_settings(
        "prod",
        project_root=Path(__file__).resolve().parents[2],
        environment="prod",
    )
    tasks = (
        "text_pretrain",
        "document_text_pair",
        "image_text_pair",
        "audio_text_pair",
        "video_text_pair",
    )
    assert production.training.release_stage == "candidate"
    assert production.training.training_backend == "dense_transformer"
    assert set(tasks).issubset(production.training.tasks)

    tokenizer_path = tmp_path / "tokenizer.json"
    train_vocabulary_tokenizer(
        output_path=tokenizer_path,
        vocab_size=269,
        texts=("alpha beta gamma delta",),
        snapshot_id="dense-candidate-smoke",
    )
    tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    tokenizer = VocabularyTokenizer.load(tokenizer_path, max_tokens=16)
    dataset_root = tmp_path / "dataset"
    _write_dense_candidate_dataset(
        root=dataset_root,
        tokenizer=tokenizer,
        tasks=tasks,
    )

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
        raw_text_max_tokens=16,
        raw_image_size=16,
        raw_audio_num_samples=512,
        raw_video_frames=2,
        enabled_modalities=("text", "document", "image", "audio", "video"),
        output_modalities=("text", "embedding"),
        text_decoder=DecoderSettings(
            enabled=True,
            vocab_size=269,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_target_tokens=16,
            max_context_tokens=32,
            max_text_context_tokens=8,
            max_document_context_tokens=6,
            max_image_context_tokens=6,
            max_audio_context_tokens=6,
            max_video_context_tokens=6,
        ),
    )
    training_settings = TrainingSettings(
        run_mode="smoke",
        release_stage="learning_candidate",
        training_backend="dense_transformer",
        text_tokenizer_max_tokens=16,
        text_tokenizer_path=str(tokenizer_path),
        text_tokenizer_sha256=tokenizer_sha256,
        text_tokenizer_vocab_size=269,
        tasks=tasks,
        approved_beta_tasks=(
            "image_text_pair",
            "audio_text_pair",
            "video_text_pair",
        ),
        curriculum_schedule=tasks,
        task_sampling_weights={task: 1.0 for task in tasks},
        min_task_samples={task: 2 for task in tasks},
        disable_undercovered_tasks=False,
        dynamic_sampling=False,
        task_aware_batching=True,
        batch_size=2,
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
    trainer = MultimodalTrainer(
        model_settings=model_settings,
        training_settings=training_settings,
        tokenizer=tokenizer,
        model_exporter=lambda **_kwargs: {},
        logger=_Logger(),
        training_backend=training_settings.training_backend,
        model_factory=build_model,
        loss_factory=build_training_loss,
        optimizer_factory=build_optimizer,
        scheduler_factory=build_lr_scheduler,
        project_root=Path.cwd(),
        checkpoint_contract=CheckpointContract(checkpoint_headers=True),
    )

    checkpoint_path = tmp_path / "dense-candidate.pt"
    result = trainer.train(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path,
        export_directory=tmp_path / "exports",
        dataset_manifest_sha256="a" * 64,
    )

    assert checkpoint_headers_present(checkpoint_path)
    assert result.metrics.batches == len(tasks)
    assert result.metrics.effective_task_counts == {
        task: 2 for task in sorted(tasks)
    }
    for modality in ("text", "document", "image", "audio", "video"):
        signal = result.metrics.training_signal_by_modality[modality]
        assert signal["gradient_detected"] is True
        assert signal["updated"] is True
