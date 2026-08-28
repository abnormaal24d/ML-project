"""Orchestrate modality collators into unified training batches."""

from __future__ import annotations

import math
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from config.multimodal.training_settings import IMPLEMENTED_TRAINING_BACKENDS
from mmcrawler_datasets.collation.audio import AudioCollator
from mmcrawler_datasets.collation.document import DocumentCollator
from mmcrawler_datasets.collation.image import ImageCollator
from mmcrawler_datasets.collation.tensor_ops import (
    sample_generator,
    stack_feature_matrix,
)
from mmcrawler_datasets.collation.text import TextCollator
from mmcrawler_datasets.collation.video import VideoCollator
from mmcrawler_datasets.schema import ProsodyFeatures
from mmcrawler_datasets.tensors import (
    MaterializedTensorSource,
    SampleTensorSource,
)
from mmcrawler_datasets.training_samples.targets import (
    validate_conversation_turns,
)
from multimodal.model.contracts import CollatedBatch
from multimodal.tokenization.text import VocabularyTokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mmcrawler_datasets.schema import MultimodalSample


class EmptyBatchError(ValueError):
    """Raised when collation receives no samples."""


class UnsupportedFeatureBackendError(ValueError):
    """Raised when an unsupported feature backend is requested."""


def positive_dim(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def require_non_empty_batch(samples: Sequence[MultimodalSample]) -> None:
    if not samples:
        raise EmptyBatchError("cannot collate an empty sample batch")


def validate_text_tokenizer_schema(
    *,
    tokenizer: VocabularyTokenizer,
    raw_text_vocab_size: int,
    raw_text_max_tokens: int,
) -> tuple[int, int]:
    """Validate one canonical tokenizer against the raw tensor schema."""

    if not isinstance(tokenizer, VocabularyTokenizer):
        raise TypeError("tokenizer must be a VocabularyTokenizer")
    tokenizer_vocab_size = len(tokenizer.token_to_id)
    if set(tokenizer.token_to_id.values()) != set(range(tokenizer_vocab_size)):
        raise ValueError(
            "tokenizer token ids must be contiguous in "
            f"[0, {tokenizer_vocab_size})"
        )
    if int(raw_text_vocab_size) != tokenizer_vocab_size:
        raise ValueError(
            "raw_text_vocab_size must match the injected tokenizer: "
            f"configured={int(raw_text_vocab_size)}, "
            f"tokenizer={tokenizer_vocab_size}"
        )
    tokenizer_max_tokens = int(tokenizer.max_tokens)
    if int(raw_text_max_tokens) != tokenizer_max_tokens:
        raise ValueError(
            "raw_text_max_tokens must match the injected tokenizer: "
            f"configured={int(raw_text_max_tokens)}, "
            f"tokenizer={tokenizer_max_tokens}"
        )
    return tokenizer_vocab_size, tokenizer_max_tokens


class MultimodalCollator:
    """Collate multimodal samples into fixed-size tensor batches."""

    def __init__(
        self,
        *,
        tokenizer: VocabularyTokenizer,
        text_dim: int,
        image_dim: int,
        audio_dim: int,
        video_dim: int,
        training_backend: str,
        raw_text_max_tokens: int,
        raw_text_vocab_size: int,
        raw_image_size: int,
        raw_audio_num_samples: int,
        raw_video_frames: int,
        video_generation_frames: int,
        audio_token_codec: str,
        mlm_probability: float,
        image_mask_probability: float,
        audio_mask_probability: float,
        modality_dropout: dict[str, float] | None = None,
        materialized_dataset_root: Path | None = None,
        materialized_tensors_enabled: bool,
        base_seed: int,
    ) -> None:
        """Configure modality collators and shared collation settings."""
        positive_dim("text_dim", text_dim)
        positive_dim("image_dim", image_dim)
        positive_dim("audio_dim", audio_dim)
        positive_dim("video_dim", video_dim)
        if training_backend not in IMPLEMENTED_TRAINING_BACKENDS:
            raise UnsupportedFeatureBackendError(
                "unsupported training_backend for explicit MultimodalModel "
                "training: "
                f"{training_backend!r}. Implemented backends are "
                f"{sorted(IMPLEMENTED_TRAINING_BACKENDS)!r}."
            )
        validate_text_tokenizer_schema(
            tokenizer=tokenizer,
            raw_text_vocab_size=raw_text_vocab_size,
            raw_text_max_tokens=raw_text_max_tokens,
        )
        self.raw_video_frames = int(raw_video_frames)
        self.video_generation_frames = int(video_generation_frames)
        if self.video_generation_frames <= 0:
            raise ValueError(
                "video_generation_frames must be greater than zero"
            )
        self.modality_dropout = {
            str(modality).strip().lower(): float(probability)
            for modality, probability in (modality_dropout or {}).items()
            if float(probability) > 0.0
        }
        invalid_dropout = {
            name: probability
            for name, probability in self.modality_dropout.items()
            if probability > 1.0
        }
        if invalid_dropout:
            raise ValueError(
                f"modality dropout probabilities must be <= 1: {invalid_dropout}"
            )
        self._base_seed = int(base_seed)
        self._epoch = 0

        normalized_audio_codec = str(audio_token_codec).strip().lower()
        if normalized_audio_codec not in {"continuous", "discrete", "none"}:
            raise ValueError(
                "audio_token_codec must be 'continuous', 'discrete', or 'none'"
            )

        if (
            not materialized_tensors_enabled
            or materialized_dataset_root is None
        ):
            raise ValueError(
                "training requires a fully materialized dataset; native media "
                "decoding belongs to preprocessing"
            )
        tensor_source: SampleTensorSource = MaterializedTensorSource(
            dataset_root=Path(materialized_dataset_root),
            image_size=int(raw_image_size),
            audio_num_samples=int(raw_audio_num_samples),
            video_frames=int(raw_video_frames),
        )
        self._tensor_source = tensor_source
        self._text = TextCollator(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            materialized_dataset_root=materialized_dataset_root,
            use_materialized_text_tokens=True,
            base_seed=self._base_seed,
        )
        self._image = ImageCollator(
            image_size=int(raw_image_size),
            dataset_root=Path(materialized_dataset_root),
            tensor_source=tensor_source,
            image_mask_probability=image_mask_probability,
            base_seed=self._base_seed,
        )
        self._audio = AudioCollator(
            dataset_root=Path(materialized_dataset_root),
            tensor_source=tensor_source,
            audio_mask_probability=audio_mask_probability,
            audio_token_codec=normalized_audio_codec,
            base_seed=self._base_seed,
        )
        self._video = VideoCollator(
            video_frames=int(raw_video_frames),
            image_size=int(raw_image_size),
            tensor_source=tensor_source,
            base_seed=self._base_seed,
        )
        self._document = DocumentCollator(
            tokenizer=tokenizer,
        )

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to deterministic masking RNGs."""

        self._epoch = int(epoch)
        self._text.set_epoch(self._epoch)
        self._image.set_epoch(self._epoch)
        self._audio.set_epoch(self._epoch)
        self._video.set_epoch(self._epoch)

    def __call__(self, samples: Sequence[MultimodalSample]) -> CollatedBatch:
        """Collate samples from every modality into one batch."""
        require_non_empty_batch(samples)

        text, text_mlm_targets = self._text.collate_batch(samples)
        image, image_reconstruction_target, image_reconstruction_mask = (
            self._image.collate_batch(samples)
        )
        audio, audio_reconstruction_target, audio_reconstruction_mask = (
            self._audio.collate_batch(samples)
        )
        video, video_temporal_labels = self._video.collate_batch(samples)
        document = self._document.collate_batch(samples)

        return assemble_collated_batch(
            samples=samples,
            text=text,
            text_mlm_targets=text_mlm_targets,
            image=image,
            image_reconstruction_target=image_reconstruction_target,
            image_reconstruction_mask=image_reconstruction_mask,
            audio=audio,
            audio_reconstruction_target=audio_reconstruction_target,
            audio_reconstruction_mask=audio_reconstruction_mask,
            video=video,
            video_temporal_labels=video_temporal_labels,
            document=document,
            text_collator=self._text,
            image_collator=self._image,
            audio_collator=self._audio,
            video_collator=self._video,
            modality_dropout=self.modality_dropout,
            base_seed=self._base_seed,
            epoch=self._epoch,
            video_generation_frames=self.video_generation_frames,
        )


_UNSUPPORTED_PROSODY_KEYS = frozenset(
    {"pitch_mean", "energy_mean", "speaking_rate"}
)


def sample_attrs(samples: Sequence[MultimodalSample], name: str) -> list[Any]:
    return [getattr(sample, name) for sample in samples]


def modality_mask(
    *,
    samples: Sequence[MultimodalSample],
    name: str,
) -> torch.Tensor:
    return torch.tensor(
        [bool(getattr(sample, f"has_{name}")) for sample in samples],
        dtype=torch.bool,
    )


def apply_dropout(
    *,
    masks: dict[str, torch.Tensor],
    probabilities: dict[str, float],
    sample_ids: Sequence[str],
    base_seed: int,
    epoch: int,
    task_types: Sequence[str] | None = None,
    disable_dropout_task_types: frozenset[str] = frozenset(
        {"multimodal_evidence_qa"}
    ),
) -> dict[str, torch.Tensor]:
    if not probabilities:
        return dict(masks)

    updated = {name: mask.clone() for name, mask in masks.items()}
    for modality, probability in probabilities.items():
        if modality not in updated or probability <= 0.0:
            continue
        active = updated[modality]
        other_active = torch.stack(
            [mask for name, mask in updated.items() if name != modality],
            dim=0,
        ).any(dim=0)
        draws = torch.tensor(
            [
                torch.rand(
                    (),
                    generator=sample_generator(
                        base_seed=base_seed,
                        epoch=epoch,
                        sample_id=sample_id,
                        operation=f"modality_dropout:{modality}",
                    ),
                ).item()
                for sample_id in sample_ids
            ],
            dtype=torch.float32,
            device=active.device,
        )
        drop = active & other_active & draws.lt(probability)

        # Disable dropout for specific task types
        if task_types is not None:
            for i, task_type in enumerate(task_types):
                if task_type in disable_dropout_task_types:
                    drop[i] = False

        updated[modality] = active & ~drop
    return updated


def first_speaker_id(segments: Sequence[object]) -> str | None:
    for segment in segments:
        speaker_id = getattr(segment, "speaker_id", None)
        if speaker_id:
            return str(speaker_id)
    return None


def _validate_conversation_samples(
    *,
    samples: Sequence[MultimodalSample],
) -> None:
    """Validate conversation structure before collation.

    Every conversation sample must expose at least one assistant answer
    target (>=1 label token) and may not reference missing multimodal
    objects. Task output must correspond to a target type.
    """

    for sample in samples:
        if not sample.has_conversation:
            continue
        answer = sample.answer_text
        if not answer:
            raise ValueError(
                "conversation sample requires an assistant answer target: "
                f"{sample.sample_id!r} task={sample.task_type!r}"
            )
        if sample.conversation_turns:
            validate_conversation_turns(
                sample.conversation_turns,
                sample_id=str(sample.sample_id),
            )
        for turn in sample.conversation_turns:
            if (
                turn.role == "tool"
                and not turn.text
                and not turn.tool_result_json
            ):
                raise ValueError(
                    "tool turn requires text or tool_result_json: "
                    f"{sample.sample_id!r}"
                )


def assemble_collated_batch(
    *,
    samples: Sequence[MultimodalSample],
    text: torch.Tensor,
    text_mlm_targets: torch.Tensor,
    image: torch.Tensor,
    image_reconstruction_target: torch.Tensor,
    image_reconstruction_mask: torch.Tensor,
    audio: torch.Tensor,
    audio_reconstruction_target: torch.Tensor,
    audio_reconstruction_mask: torch.Tensor,
    video: torch.Tensor,
    video_temporal_labels: torch.Tensor,
    document: torch.Tensor,
    text_collator: TextCollator,
    image_collator: ImageCollator,
    audio_collator: AudioCollator,
    video_collator: VideoCollator,
    modality_dropout: dict[str, float],
    base_seed: int,
    epoch: int,
    video_generation_frames: int,
) -> CollatedBatch:
    """Merge per-modality collator outputs into one batch schema."""
    sample_ids = [sample.sample_id for sample in samples]
    attrs = sample_attrs

    text_mask = modality_mask(samples=samples, name="text")
    image_mask = modality_mask(samples=samples, name="image")
    audio_mask = modality_mask(samples=samples, name="audio")
    video_mask = modality_mask(samples=samples, name="video")
    document_mask = modality_mask(samples=samples, name="document")
    layout_mask = modality_mask(samples=samples, name="layout")
    edit_mask_input_mask = modality_mask(samples=samples, name="mask")
    dropped_masks = apply_dropout(
        masks={
            "text": text_mask,
            "document": document_mask,
            "image": image_mask,
            "audio": audio_mask,
            "video": video_mask,
        },
        probabilities=modality_dropout,
        sample_ids=sample_ids,
        base_seed=base_seed,
        epoch=epoch,
    )
    text_mask = dropped_masks["text"]
    document_mask = dropped_masks["document"]
    image_mask = dropped_masks["image"]
    audio_mask = dropped_masks["audio"]
    video_mask = dropped_masks["video"]
    modality_mask_tensor = stack_feature_matrix(
        [text_mask, document_mask, image_mask, audio_mask, video_mask]
    ).T

    _validate_conversation_samples(samples=samples)
    decoder_tensors = text_collator.collate_decoder_tensors(samples)
    preference_tensors = text_collator.collate_preference_tensors(samples)
    safety_targets, safety_target_mask = text_collator.collate_safety_targets(
        samples
    )

    layout_inputs = DocumentCollator.collate_layout_inputs(
        [sample.layout_boxes for sample in samples],
    )
    audio_token_ids, audio_token_attention_mask = (
        audio_collator.collate_audio_token_targets(samples)
    )
    video_token_result = video_collator.collate_video_token_targets(
        attrs(samples, "target_video_tokens_path"),
        frame_count=video_generation_frames,
    )
    video_token_targets = (
        video_token_result[0] if video_token_result is not None else None
    )
    video_token_attention_mask = (
        video_token_result[1] if video_token_result is not None else None
    )

    return CollatedBatch(
        sample_ids=sample_ids,
        text=text,
        image=image,
        audio=audio,
        video=video,
        modality_mask=modality_mask_tensor,
        labels=TextCollator.collate_labels(samples),
        document=document,
        target_texts=[sample.generative_target_text for sample in samples],
        positive_ids=[sample.positive_id for sample in samples],
        negative_ids=[sample.negative_ids for sample in samples],
        task_types=[sample.task_type for sample in samples],
        task_ids=TextCollator.collate_task_ids(samples),
        instructions=[sample.instruction for sample in samples],
        questions=[sample.question for sample in samples],
        answers=[sample.answer for sample in samples],
        output_modalities=[sample.output_modalities for sample in samples],
        target_audio_tokens_paths=attrs(samples, "target_audio_tokens_path"),
        target_audio_token_ids=audio_token_ids,
        target_audio_token_attention_mask=audio_token_attention_mask,
        # Use full sequence for generative audio training (not just first token)
        audio_token_targets=audio_token_ids,
        target_image_tensor_paths=attrs(samples, "target_image_tensor_path"),
        target_video_tensor_paths=attrs(samples, "target_video_tensor_path"),
        target_video_tokens_paths=attrs(samples, "target_video_tokens_path"),
        source_image_tensor_paths=attrs(
            samples,
            "source_image_tensor_path",
        ),
        edit_mask_tensor_paths=attrs(samples, "edit_mask_tensor_path"),
        target_codes=[sample.target_code for sample in samples],
        code_languages=[sample.code_language for sample in samples],
        question_token_ids=text_collator.collate_token_ids(
            [
                sample.question or sample.instruction or ""
                for sample in samples
            ],
        ),
        target_token_ids=text_collator.collate_token_ids(
            [sample.generative_target_text or "" for sample in samples],
        ),
        target_attention_mask=text_collator.collate_target_attention_mask(
            [sample.generative_target_text or "" for sample in samples],
        ),
        decoder_input_ids=decoder_tensors["decoder_input_ids"],
        decoder_labels=decoder_tensors["decoder_labels"],
        decoder_attention_mask=decoder_tensors["decoder_attention_mask"],
        chosen_input_ids=preference_tensors["chosen_input_ids"],
        chosen_labels=preference_tensors["chosen_labels"],
        chosen_attention_mask=preference_tensors["chosen_attention_mask"],
        rejected_input_ids=preference_tensors["rejected_input_ids"],
        rejected_labels=preference_tensors["rejected_labels"],
        rejected_attention_mask=preference_tensors["rejected_attention_mask"],
        safety_targets=safety_targets,
        safety_target_mask=safety_target_mask,
        prompt_token_count=decoder_tensors["prompt_token_count"],
        answer_token_count=decoder_tensors["answer_token_count"],
        conversation_flags=[
            bool(sample.has_conversation) for sample in samples
        ],
        layout_boxes=[sample.layout_boxes for sample in samples],
        ui_elements=[sample.ui_elements for sample in samples],
        geometry_annotations=attrs(samples, "geometry_annotations"),
        object_boxes=[sample.object_boxes for sample in samples],
        speaker_segments=[sample.speaker_segments for sample in samples],
        target_image_tensor=image_collator.collate_optional_images(
            paths=attrs(samples, "target_image_tensor_path"),
        ),
        target_video_tensor=video_collator.collate_optional_videos(
            paths=attrs(samples, "target_video_tensor_path"),
        ),
        source_image_tensor=image_collator.collate_optional_images(
            paths=attrs(samples, "source_image_tensor_path"),
        ),
        edit_mask_tensor=image_collator.collate_optional_edit_masks(
            paths=attrs(samples, "edit_mask_tensor_path"),
        ),
        layout_tensor=layout_inputs["document_layout_boxes"],
        table_tensor=DocumentCollator.collate_table_inputs(samples),
        layout_box_targets=DocumentCollator.collate_box_targets(
            [sample.layout_boxes for sample in samples]
        ),
        object_box_targets=DocumentCollator.collate_box_targets(
            [sample.object_boxes for sample in samples]
        ),
        emotion_label_ids=TextCollator.collate_optional_label_ids(
            [sample.emotion_label for sample in samples]
        ),
        speaker_label_ids=TextCollator.collate_optional_label_ids(
            [
                sample.speaker_label
                or first_speaker_id(sample.speaker_segments)
                for sample in samples
            ]
        ),
        prosody_targets=_collate_prosody_targets(samples),
        prosody_mask=_collate_prosody_mask(samples),
        video_token_targets=video_token_targets,
        video_token_attention_mask=video_token_attention_mask,
        document_mask=document_mask,
        layout_mask=layout_mask,
        edit_mask_input_mask=edit_mask_input_mask,
        **layout_inputs,
        text_mask=text_mask,
        image_mask=image_mask,
        audio_mask=audio_mask,
        video_mask=video_mask,
        alignment_scores=TextCollator.collate_alignment_scores(samples),
        text_mlm_targets=text_mlm_targets,
        image_reconstruction_target=image_reconstruction_target,
        image_reconstruction_mask=image_reconstruction_mask,
        audio_reconstruction_target=audio_reconstruction_target,
        audio_reconstruction_mask=audio_reconstruction_mask,
        video_temporal_labels=video_temporal_labels,
    )


def _collate_prosody_targets(
    samples: Sequence[MultimodalSample],
) -> torch.Tensor | None:
    """Collate pitch_hz, energy, tempo, and pause_ratio into [B, 4]."""

    prosody_list: list[list[float]] = []
    has_any = False
    for sample in samples:
        values = _prosody_values(getattr(sample, "prosody", None))
        if values is None:
            prosody_list.append([0.0, 0.0, 0.0, 0.0])
            continue
        has_values = any(value is not None for value in values)
        has_any = has_any or has_values
        prosody_list.append(
            [value if value is not None else 0.0 for value in values]
        )
    if not has_any:
        return None
    return torch.tensor(prosody_list, dtype=torch.float32)


def _collate_prosody_mask(
    samples: Sequence[MultimodalSample],
) -> torch.Tensor | None:
    """Return one for samples with at least one canonical prosody value."""

    mask: list[float] = []
    for sample in samples:
        values = _prosody_values(getattr(sample, "prosody", None))
        mask.append(
            1.0
            if values is not None
            and any(value is not None for value in values)
            else 0.0
        )
    if all(m == 0.0 for m in mask):
        return None
    return torch.tensor(mask, dtype=torch.float32)


def _prosody_values(
    value: object,
) -> tuple[float | None, float | None, float | None, float | None] | None:
    if value is None:
        return None
    if isinstance(value, ProsodyFeatures):
        raw_values = (
            value.pitch_hz,
            value.energy,
            value.tempo,
            value.pause_ratio,
        )
    elif isinstance(value, dict):
        unsupported_keys = sorted(
            _UNSUPPORTED_PROSODY_KEYS.intersection(value)
        )
        if unsupported_keys:
            fields = ", ".join(unsupported_keys)
            raise ValueError(
                f"unsupported prosody field(s) {fields}; "
                "use pitch_hz, energy, and tempo"
            )
        raw_values = (
            value.get("pitch_hz"),
            value.get("energy"),
            value.get("tempo"),
            value.get("pause_ratio"),
        )
    else:
        raise TypeError("prosody must be ProsodyFeatures or a canonical dict")
    return (
        _safe_float(raw_values[0]),
        _safe_float(raw_values[1]),
        _safe_float(raw_values[2]),
        _safe_float(raw_values[3]),
    )


def _safe_float(v: object) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float, str, bytes, bytearray)):
        try:
            result = float(v)
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return None


def move_batch_to_device(
    *,
    batch: "CollatedBatch",
    device: "torch.device",
) -> "CollatedBatch":
    """Move tensor fields in a collated batch onto the target device.
    (moved from runtime_helpers.py - batch transformation, not runtime helper)
    """
    updates: dict[str, Any] = {}
    for batch_field in fields(batch):
        value = getattr(batch, batch_field.name)
        if isinstance(value, torch.Tensor):
            updates[batch_field.name] = value.to(device)
    return replace(batch, **updates)


def select_batch_rows(
    *,
    batch: "CollatedBatch",
    rows: list[int],
) -> "CollatedBatch":
    """Select a subset of rows while keeping every field row-aligned.

    Tensors with a leading batch dimension and lists aligned with
    sample_ids are sliced together so downstream consumers observe a
    consistent batch.
    """
    batch_size = len(batch.sample_ids)
    for row in rows:
        if row < 0 or row >= batch_size:
            raise IndexError(
                f"row index {row} is outside the batch of size {batch_size}"
            )
    updates: dict[str, Any] = {}
    for batch_field in fields(batch):
        value = getattr(batch, batch_field.name)
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == batch_size
        ):
            indices = torch.tensor(rows, dtype=torch.long, device=value.device)
            updates[batch_field.name] = value.index_select(0, indices)
        elif isinstance(value, list) and len(value) == batch_size:
            updates[batch_field.name] = [value[index] for index in rows]
    return replace(batch, **updates)
