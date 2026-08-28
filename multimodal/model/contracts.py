"""Model-owned contracts: modality, token-sequence, and output definitions.

This file is the authoritative source for model contracts. Do not import
or reference orchestration values from the model package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

MODALITY_ORDER = ("text", "document", "image", "audio", "video")
PHYSICAL_MODALITIES = frozenset(MODALITY_ORDER)
CONDITIONING_MODALITIES = ("document", "layout", "mask")
LOGICAL_TO_PHYSICAL_MODALITIES = {
    "document": ("document",),
    "layout": (),
    "mask": (),
    "code": ("text",),
    "json": ("text",),
}

MODALITY_TOKEN_IDS = {
    "text": 0,
    "document": 1,
    "image": 2,
    "audio": 3,
    "video": 4,
    "layout": 5,
    "mask": 6,
}


@dataclass(frozen=True, slots=True)
class CollatedBatch:
    """Canonical tensor batch consumed by the multimodal model."""

    sample_ids: list[str]
    text: torch.Tensor
    image: torch.Tensor
    audio: torch.Tensor
    video: torch.Tensor
    modality_mask: torch.Tensor
    labels: torch.Tensor | None
    document: torch.Tensor | None = None
    target_texts: list[str | None] = field(default_factory=list)
    positive_ids: list[str | None] = field(default_factory=list)
    negative_ids: list[tuple[str, ...]] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)

    task_ids: torch.Tensor | None = None
    instructions: list[str | None] = field(default_factory=list)
    questions: list[str | None] = field(default_factory=list)
    answers: list[str | None] = field(default_factory=list)
    output_modalities: list[tuple[str, ...]] = field(default_factory=list)
    target_audio_tokens_paths: list[Path | None] = field(default_factory=list)
    target_audio_token_ids: torch.Tensor | None = None
    target_audio_token_attention_mask: torch.Tensor | None = None
    target_image_tensor_paths: list[Path | None] = field(default_factory=list)
    target_video_tensor_paths: list[Path | None] = field(default_factory=list)
    target_video_tokens_paths: list[Path | None] = field(default_factory=list)
    source_image_tensor_paths: list[Path | None] = field(default_factory=list)
    edit_mask_tensor_paths: list[Path | None] = field(default_factory=list)
    target_codes: list[str | None] = field(default_factory=list)
    code_languages: list[str | None] = field(default_factory=list)
    question_token_ids: torch.Tensor | None = None
    target_token_ids: torch.Tensor | None = None
    target_attention_mask: torch.Tensor | None = None
    decoder_input_ids: torch.Tensor | None = None
    decoder_labels: torch.Tensor | None = None
    decoder_attention_mask: torch.Tensor | None = None
    chosen_input_ids: torch.Tensor | None = None
    chosen_labels: torch.Tensor | None = None
    chosen_attention_mask: torch.Tensor | None = None
    rejected_input_ids: torch.Tensor | None = None
    rejected_labels: torch.Tensor | None = None
    rejected_attention_mask: torch.Tensor | None = None
    safety_targets: torch.Tensor | None = None
    safety_target_mask: torch.Tensor | None = None
    prompt_token_count: list[int] = field(default_factory=list)
    answer_token_count: list[int] = field(default_factory=list)
    conversation_flags: list[bool] = field(default_factory=list)
    layout_boxes: list[object] = field(default_factory=list)
    ui_elements: list[object] = field(default_factory=list)
    geometry_annotations: list[object] = field(default_factory=list)
    object_boxes: list[object] = field(default_factory=list)
    speaker_segments: list[object] = field(default_factory=list)
    target_image_tensor: torch.Tensor | None = None
    target_image_mask: torch.Tensor | None = None
    target_video_tensor: torch.Tensor | None = None
    source_image_tensor: torch.Tensor | None = None
    edit_mask_tensor: torch.Tensor | None = None
    layout_tensor: torch.Tensor | None = None
    table_tensor: torch.Tensor | None = None
    layout_box_targets: torch.Tensor | None = None
    object_box_targets: torch.Tensor | None = None
    emotion_label_ids: torch.Tensor | None = None
    speaker_label_ids: torch.Tensor | None = None
    prosody_targets: torch.Tensor | None = None
    prosody_mask: torch.Tensor | None = None
    audio_token_targets: torch.Tensor | None = None
    video_token_targets: torch.Tensor | None = None
    video_token_attention_mask: torch.Tensor | None = None
    document_mask: torch.Tensor | None = None
    layout_mask: torch.Tensor | None = None
    edit_mask_input_mask: torch.Tensor | None = None
    document_layout_boxes: torch.Tensor | None = None
    document_page_ids: torch.Tensor | None = None
    document_layout_attention_mask: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    image_mask: torch.Tensor | None = None
    audio_mask: torch.Tensor | None = None
    video_mask: torch.Tensor | None = None
    alignment_scores: torch.Tensor | None = None
    text_mlm_targets: torch.Tensor | None = None
    image_reconstruction_target: torch.Tensor | None = None
    image_reconstruction_mask: torch.Tensor | None = None
    audio_reconstruction_target: torch.Tensor | None = None
    audio_reconstruction_mask: torch.Tensor | None = None
    video_temporal_labels: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class ModalityTokenSequence:
    """One padded modality token sequence consumed by dense fusion."""

    tokens: torch.Tensor
    attention_mask: torch.Tensor
    modality_ids: torch.Tensor
    temporal_positions: torch.Tensor | None = None
    spatial_positions: torch.Tensor | None = None
    separator_mask: torch.Tensor | None = None

    def validate(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError("modality tokens must be [batch, tokens, hidden]")
        if self.attention_mask.shape != self.tokens.shape[:2]:
            raise ValueError("modality attention mask must match token axes")
        if self.modality_ids.shape != self.tokens.shape[:2]:
            raise ValueError("modality ids must match token axes")
        if self.temporal_positions is not None and (
            self.temporal_positions.shape != self.tokens.shape[:2]
        ):
            raise ValueError("temporal positions must match token axes")
        if self.spatial_positions is not None and (
            self.spatial_positions.shape != (*self.tokens.shape[:2], 2)
        ):
            raise ValueError("spatial positions must be [batch, tokens, 2]")
        if self.separator_mask is not None and (
            self.separator_mask.shape != self.tokens.shape[:2]
        ):
            raise ValueError("separator mask must match token axes")


@dataclass(frozen=True, slots=True)
class DenseDecoderOutput:
    """Autoregressive decoder output for training or generation."""

    hidden_states: torch.Tensor
    logits: torch.Tensor

    def validate(self) -> None:
        if self.hidden_states.ndim != 3:
            raise ValueError(
                "decoder hidden states must be [batch, tokens, hidden]"
            )
        if self.logits.ndim != 3:
            raise ValueError(
                "decoder logits must be [batch, tokens, vocabulary]"
            )
        if self.hidden_states.shape[:2] != self.logits.shape[:2]:
            raise ValueError("decoder hidden/logit token axes must match")


ALL_OUTPUT_HEADS = frozenset(
    {
        "classifier",
        "projection",
        "modality_embeddings",
        "sequence",
        "emotion",
        "speaker",
        "layout",
        "object",
        "audio_token",
        "prosody",
        "image_latents",
        "generated_image",
        "text_mlm",
        "image_reconstruction",
        "audio_reconstruction",
        "video_temporal",
        "video_generation",
    }
)

ENCODER_OUTPUT_HEADS = {
    "text_mlm_logits": "text_mlm",
    "image_reconstruction": "image_reconstruction",
    "image_region_features": "image_reconstruction",
    "audio_reconstruction": "audio_reconstruction",
    "video_temporal_logits": "video_temporal",
}
