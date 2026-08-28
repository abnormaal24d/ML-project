"""Compose MultimodalJsonlDataset with sampling into a PyTorch DataLoader."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import torch
from torch.utils.data import DataLoader

from config.multimodal.training_settings import IMPLEMENTED_TRAINING_BACKENDS
from mmcrawler_datasets.dataset import (
    MultimodalJsonlDataset,
    resolve_split_paths,
)
from mmcrawler_datasets.sampling import (
    apply_distributed_sampling,
    select_sampling,
)
from mmcrawler_datasets.schema import DatasetSplit
from multimodal.tokenization.text import VocabularyTokenizer
from schemas.multimodal_tasks import canonical_task_name

if TYPE_CHECKING:
    from config.multimodal.model_settings import ModelSettings
    from config.multimodal.training_settings import TrainingSettings
    from logger.project_logger import ProjectLogger


def build_dataloader(
    *,
    dataset_root: Path,
    split: DatasetSplit,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    tokenizer: VocabularyTokenizer,
    logger: ProjectLogger,
    distributed: bool = True,
) -> tuple[MultimodalJsonlDataset, DataLoader[Any]]:
    """Build one validated dataset and DataLoader for a split."""

    from mmcrawler_datasets.collation.multimodal import MultimodalCollator

    if not isinstance(split, DatasetSplit):
        raise TypeError(f"split must be DatasetSplit, got {type(split)!r}")

    _validate_collator_backend_compatibility(
        backend=training_settings.training_backend,
    )

    resolved_root = Path(dataset_root).resolve()
    paths = resolve_split_paths(dataset_root=resolved_root, split=split)

    enabled = frozenset(
        canonical_task_name(task)
        for task in training_settings.tasks
        if task.strip()
    )
    minimums = {
        canonical_task_name(task): max(0, int(minimum))
        for task, minimum in training_settings.effective_min_task_samples().items()
    }
    if not training_settings.disable_undercovered_tasks:
        minimums = {}

    dataset = MultimodalJsonlDataset(
        dataset_root=resolved_root,
        split=split,
        record_paths=paths,
        max_samples=training_settings.max_samples,
        logger=logger,
        enabled_tasks=enabled,
        minimums_by_task=minimums,
        drop_samples_with_invalid_targets=(
            training_settings.drop_samples_with_invalid_targets
        ),
        require_materialized_tensors=(
            training_settings.materialized_tensors_enabled
        ),
    )
    if len(dataset) == 0:
        raise ValueError(
            f"dataset split {split.value!r} has no usable samples"
        )

    min_items = max(
        0,
        int(
            training_settings.min_split_items_by_split.get(
                split.value, training_settings.min_split_items
            )
        ),
    )
    if len(dataset) < min_items:
        raise ValueError(
            f"dataset split {split.value!r} contains {len(dataset)} "
            f"usable samples; minimum required is {min_items}"
        )

    feature_dimensions = model_settings.feature_dimensions
    collator = MultimodalCollator(
        tokenizer=tokenizer,
        text_dim=feature_dimensions["text"],
        image_dim=feature_dimensions["image"],
        audio_dim=feature_dimensions["audio"],
        video_dim=feature_dimensions["video"],
        training_backend=training_settings.training_backend,
        raw_text_max_tokens=model_settings.raw_text_max_tokens,
        raw_text_vocab_size=model_settings.raw_text_vocab_size,
        raw_image_size=model_settings.raw_image_size,
        raw_audio_num_samples=model_settings.raw_audio_num_samples,
        raw_video_frames=model_settings.raw_video_frames,
        video_generation_frames=model_settings.video_generator.frames,
        audio_token_codec=model_settings.audio_tokenizer.codec,
        mlm_probability=training_settings.mlm_probability,
        image_mask_probability=training_settings.image_mask_probability,
        audio_mask_probability=training_settings.audio_mask_probability,
        modality_dropout=(
            training_settings.modality_dropout
            if split is DatasetSplit.TRAIN
            else None
        ),
        materialized_dataset_root=(
            resolved_root
            if training_settings.materialized_tensors_enabled
            else None
        ),
        materialized_tensors_enabled=(
            training_settings.materialized_tensors_enabled
        ),
        base_seed=int(training_settings.seed),
    )

    batch_size = _effective_batch_size(
        configured_batch_size=int(training_settings.batch_size),
        dataset_size=len(dataset),
    )
    shuffle_training = _should_shuffle(
        split=split,
        training_settings=training_settings,
    )
    sampler, batch_sampler = select_sampling(
        dataset=dataset,
        split=split,
        settings=training_settings,
        shuffle_training=shuffle_training,
        batch_size=batch_size,
    )
    if distributed:
        sampler, batch_sampler = apply_distributed_sampling(
            dataset=dataset,
            sampler=sampler,
            batch_sampler=batch_sampler,
            shuffle=shuffle_training,
            drop_last=(
                bool(training_settings.drop_last)
                if split is DatasetSplit.TRAIN
                else False
            ),
            seed=int(training_settings.seed),
        )

    num_workers = _resolved_num_workers(training_settings=training_settings)
    pin_memory = _should_pin_memory(training_settings=training_settings)
    persistent_workers = num_workers > 0 and bool(
        training_settings.persistent_workers
    )
    generator = torch.Generator()
    generator.manual_seed(int(training_settings.seed))

    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if batch_sampler is not None:
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs.update(
            {
                "batch_size": batch_size,
                "drop_last": (
                    bool(training_settings.drop_last)
                    if split is DatasetSplit.TRAIN
                    else False
                ),
                "generator": generator,
            }
        )
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
            loader_kwargs["shuffle"] = False
        else:
            loader_kwargs["shuffle"] = shuffle_training

    if num_workers > 0:
        loader_kwargs["worker_init_fn"] = _seed_worker
        if training_settings.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = max(
                1, int(training_settings.prefetch_factor)
            )

    loader = DataLoader(dataset, **loader_kwargs)
    logger.info(
        "multimodal_dataloader_built",
        dataset_root=resolved_root.as_posix(),
        split=split.value,
        batch_size=batch_size,
        sample_count=len(dataset),
        num_workers=num_workers,
    )
    return dataset, loader


def load_configured_vocabulary_tokenizer(
    *,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    project_root: Path,
) -> VocabularyTokenizer:
    """Load and verify the configured tokenizer before dataset construction."""

    configured_vocab_size = int(training_settings.text_tokenizer_vocab_size)
    model_vocab_size = int(model_settings.raw_text_vocab_size)
    if configured_vocab_size != model_vocab_size:
        raise ValueError(
            "text_tokenizer_vocab_size must match model raw_text_vocab_size: "
            f"tokenizer_config={configured_vocab_size}, model={model_vocab_size}"
        )

    configured_max_tokens = int(training_settings.text_tokenizer_max_tokens)
    model_max_tokens = int(model_settings.raw_text_max_tokens)
    if configured_max_tokens != model_max_tokens:
        raise ValueError(
            "text_tokenizer_max_tokens must match model raw_text_max_tokens: "
            f"tokenizer_config={configured_max_tokens}, model={model_max_tokens}"
        )

    expected_sha256 = training_settings.text_tokenizer_sha256
    if expected_sha256 is None:
        raise ValueError(
            "text_tokenizer_sha256 is required before training can open a "
            "dataset"
        )

    tokenizer_path = _resolve_tokenizer_artifact_path(
        training_settings.text_tokenizer_path,
        project_root=project_root,
    )
    if not tokenizer_path.is_file():
        raise ValueError(
            "configured text tokenizer artifact does not exist: "
            f"{tokenizer_path}"
        )

    artifact_bytes = tokenizer_path.read_bytes()
    observed_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "text tokenizer SHA-256 mismatch: "
            f"expected={expected_sha256}, observed={observed_sha256}, "
            f"path={tokenizer_path}"
        )

    try:
        payload = json.loads(artifact_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid text tokenizer artifact JSON: {tokenizer_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("text tokenizer artifact root must be an object")

    expected_version = training_settings.text_tokenizer_artifact_version
    observed_version = str(payload.get("tokenizer_type", ""))
    if observed_version != expected_version:
        raise ValueError(
            "text tokenizer artifact version mismatch: "
            f"expected={expected_version!r}, observed={observed_version!r}"
        )

    declared_vocab_size = int(payload.get("vocab_size", -1))
    if declared_vocab_size != configured_vocab_size:
        raise ValueError(
            "text tokenizer declared vocab_size mismatch: "
            f"configured={configured_vocab_size}, "
            f"artifact={declared_vocab_size}"
        )

    token_to_id_raw = payload.get("token_to_id")
    if not isinstance(token_to_id_raw, dict):
        raise ValueError(
            "text tokenizer artifact token_to_id must be an object"
        )
    token_to_id = {
        str(token): int(token_id)
        for token, token_id in token_to_id_raw.items()
    }
    if len(token_to_id) != configured_vocab_size:
        raise ValueError(
            "text tokenizer vocabulary length mismatch: "
            f"configured={configured_vocab_size}, artifact={len(token_to_id)}"
        )
    if set(token_to_id.values()) != set(range(configured_vocab_size)):
        raise ValueError(
            "text tokenizer ids must be contiguous in "
            f"[0, {configured_vocab_size})"
        )

    expected_special_tokens = training_settings.text_tokenizer_special_tokens
    mismatched_special_tokens = {
        token: {
            "expected": int(expected_id),
            "observed": token_to_id.get(token),
        }
        for token, expected_id in expected_special_tokens.items()
        if token_to_id.get(token) != int(expected_id)
    }
    if mismatched_special_tokens:
        raise ValueError(
            "text tokenizer special-token mapping mismatch: "
            f"{mismatched_special_tokens}"
        )

    tokenizer = VocabularyTokenizer.load(
        tokenizer_path,
        max_tokens=model_max_tokens,
    )
    if len(tokenizer.token_to_id) != model_vocab_size:
        raise ValueError(
            "raw_text_vocab_size must match the loaded tokenizer: "
            f"model={model_vocab_size}, "
            f"tokenizer={len(tokenizer.token_to_id)}"
        )
    return tokenizer


def _validate_collator_backend_compatibility(
    *,
    backend: str,
) -> None:
    if backend in IMPLEMENTED_TRAINING_BACKENDS:
        return
    raise ValueError(
        "unsupported training_backend for the current collator/model "
        "pipeline: "
        f"{backend!r}. Implemented backends are "
        f"{sorted(IMPLEMENTED_TRAINING_BACKENDS)!r}."
    )


def _resolved_num_workers(*, training_settings: TrainingSettings) -> int:
    configured = max(0, int(training_settings.num_workers))
    if configured <= 0:
        return 0
    device = training_settings.device.strip().lower()
    if device == "cpu":
        return 0
    if device == "auto" and not torch.cuda.is_available():
        return 0
    return configured


def _should_shuffle(
    *,
    split: DatasetSplit,
    training_settings: TrainingSettings,
) -> bool:
    return split is DatasetSplit.TRAIN and bool(
        training_settings.shuffle_training_split
    )


def _should_pin_memory(*, training_settings: TrainingSettings) -> bool:
    if not bool(training_settings.pin_memory):
        return False
    device = training_settings.device.strip().lower()
    if device == "cpu":
        return False
    if device == "auto":
        return torch.cuda.is_available()
    return device.startswith("cuda")


def _effective_batch_size(
    *,
    configured_batch_size: int,
    dataset_size: int,
) -> int:
    if dataset_size <= 0:
        return max(1, configured_batch_size)
    return max(1, min(configured_batch_size, dataset_size))


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    worker_seed = (worker_seed + int(worker_id)) % 2**32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(worker_seed)


def _resolve_tokenizer_artifact_path(
    configured_path: str,
    *,
    project_root: Path,
) -> Path:
    raw_value = configured_path.strip()
    if not raw_value:
        raise ValueError("text_tokenizer_path must not be empty")

    if re.match(r"^[A-Za-z]:[\\/]", raw_value):
        candidate = Path(raw_value)
    else:
        parsed = urlparse(raw_value)
        if parsed.scheme and parsed.scheme != "file":
            raise ValueError(
                "text_tokenizer_path must be a local path or file URI during "
                f"offline training, got scheme {parsed.scheme!r}"
            )
        if parsed.scheme == "file":
            path_text = unquote(parsed.path)
            if re.match(r"^/[A-Za-z]:/", path_text):
                path_text = path_text[1:]
            candidate = Path(path_text)
        else:
            candidate = Path(raw_value).expanduser()

    if not candidate.is_absolute():
        candidate = Path(project_root) / candidate
    return candidate.resolve()
