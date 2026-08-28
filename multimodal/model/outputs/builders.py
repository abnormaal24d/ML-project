"""Build supervised and generation outputs from fused representations."""

from __future__ import annotations

import torch
from torch import nn

from multimodal.model.contracts import CollatedBatch
from multimodal.model.outputs.batch import (
    head_requested,
    include_encoder_output,
    modality_embedding_outputs,
)
from multimodal.model.outputs.projection import ProjectionHeads


class SupervisedOutputBuilder:
    """Build loss and diagnostic outputs from injected projection heads."""

    def __init__(
        self,
        *,
        heads: ProjectionHeads,
        enabled_modalities: tuple[str, ...],
    ) -> None:
        self._heads = heads
        self._enabled_modalities = enabled_modalities

    def build(
        self,
        *,
        fused: torch.Tensor,
        resolved_heads: frozenset[str],
        encoded_by_modality: dict[str, dict[str, torch.Tensor]],
        batch_size: int,
        batch: CollatedBatch,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        if head_requested(resolved_heads, "classifier"):
            logits = self._heads.classifier(fused)
            outputs.update({"logits": logits, "label_logits": logits})
        if head_requested(resolved_heads, "projection"):
            projected = nn.functional.normalize(
                self._heads.projection(fused),
                dim=-1,
            )
            outputs.update({"embedding": projected, "projection": projected})
        if head_requested(resolved_heads, "emotion"):
            outputs["emotion_logits"] = self._heads.emotion_head(fused)
        if head_requested(resolved_heads, "speaker"):
            outputs["speaker_embedding"] = nn.functional.normalize(
                self._heads.speaker_projection(fused),
                dim=-1,
            )
        if head_requested(resolved_heads, "layout"):
            outputs["layout_box_prediction"] = self._heads.layout_box_head(
                fused
            ).sigmoid()
        if head_requested(resolved_heads, "object"):
            outputs["object_box_prediction"] = self._heads.object_box_head(
                fused
            ).sigmoid()
        if head_requested(resolved_heads, "prosody"):
            outputs["prosody_prediction"] = self._heads.prosody_head(fused)
        if getattr(batch, "safety_targets", None) is not None:
            outputs["safety_logits"] = self._heads.safety_head(fused)
        if head_requested(resolved_heads, "modality_embeddings"):
            modality_outputs = modality_embedding_outputs(
                encoded_by_modality=encoded_by_modality,
                enabled_modalities=self._enabled_modalities,
                batch_size=batch_size,
                fusion_dim=self._heads.fusion_dim,
                device=fused.device,
            )
            outputs.update(modality_outputs)
            outputs.update(
                _build_contrastive_outputs(
                    batch=batch,
                    modality_outputs=modality_outputs,
                )
            )
        for encoded in encoded_by_modality.values():
            outputs.update(
                {
                    key: value
                    for key, value in encoded.items()
                    if key != "embedding"
                    and include_encoder_output(
                        key=key,
                        output_heads=resolved_heads,
                    )
                }
            )
        return outputs


class GenerationOutputBuilder:
    """Build requested generation tensors from injected heads."""

    def __init__(self, *, heads: ProjectionHeads) -> None:
        self._heads = heads

    def build(
        self,
        *,
        fused: torch.Tensor,
        resolved_heads: frozenset[str],
        decoder_hidden_states: torch.Tensor | None = None,
        sequence_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        if "sequence" in resolved_heads:
            if sequence_logits is not None:
                outputs["sequence_logits"] = sequence_logits
                if decoder_hidden_states is not None:
                    outputs["decoder_hidden_states"] = decoder_hidden_states
            else:
                text_head = self._heads.text_generation_head
                if text_head is None:
                    raise RuntimeError(
                        "legacy sequence output requires the pipeline-smoke "
                        "text generation head"
                    )
                position_embedding = self._heads.text_position_embedding
                if position_embedding is None:
                    raise RuntimeError(
                        "legacy sequence output requires text positions"
                    )
                positions = torch.arange(
                    self._heads.text_sequence_length,
                    device=fused.device,
                )
                sequence_hidden = fused.unsqueeze(1) + position_embedding(
                    positions
                ).unsqueeze(0)
                outputs["sequence_logits"] = text_head(sequence_hidden)
        if "image_latents" in resolved_heads:
            image_latent_head = self._heads.image_latent_head
            if image_latent_head is None:
                raise RuntimeError("image generation heads are disabled")
            outputs["image_latents"] = image_latent_head(fused)
        if "generated_image" in resolved_heads:
            image_pixel_head = self._heads.image_pixel_head
            if image_pixel_head is None:
                raise RuntimeError("image generation heads are disabled")
            batch_size = fused.shape[0]
            outputs["generated_image"] = image_pixel_head(fused).reshape(
                batch_size,
                self._heads.image_output_channels,
                self._heads.image_size,
                self._heads.image_size,
            )
        if "audio_token" in resolved_heads:
            audio_codebook_embedding = self._heads.audio_codebook_embedding
            audio_position_embedding = self._heads.audio_position_embedding
            audio_generation_head = self._heads.audio_generation_head
            if (
                audio_codebook_embedding is None
                or audio_position_embedding is None
                or audio_generation_head is None
            ):
                raise RuntimeError("audio generation heads are disabled")
            batch_size = fused.shape[0]
            device = fused.device
            codebook_positions = torch.arange(
                self._heads.audio_n_codebooks, device=device
            )
            time_positions = torch.arange(
                self._heads.audio_generation_frames, device=device
            )
            hidden = (
                fused[:, None, None, :]
                + audio_codebook_embedding(codebook_positions)[
                    None, :, None, :
                ]
                + audio_position_embedding(time_positions)[None, None, :, :]
            )
            outputs["audio_token_logits"] = audio_generation_head(hidden)
        if "video_generation" in resolved_heads:
            video_position_embedding = self._heads.video_position_embedding
            video_row_embedding = self._heads.video_row_embedding
            video_column_embedding = self._heads.video_column_embedding
            video_generation_head = self._heads.video_generation_head
            if (
                video_position_embedding is None
                or video_row_embedding is None
                or video_column_embedding is None
                or video_generation_head is None
            ):
                raise RuntimeError("video generation heads are disabled")
            batch_size = fused.shape[0]
            device = fused.device
            time_positions = torch.arange(
                self._heads.video_generation_frames, device=device
            )
            row_positions = torch.arange(
                self._heads.video_grid_height, device=device
            )
            column_positions = torch.arange(
                self._heads.video_grid_width, device=device
            )
            hidden = (
                fused[:, None, None, None, :]
                + video_position_embedding(time_positions)[
                    None, :, None, None, :
                ]
                + video_row_embedding(row_positions)[None, None, :, None, :]
                + video_column_embedding(column_positions)[
                    None, None, None, :, :
                ]
            )
            outputs["video_token_logits"] = video_generation_head(hidden)
        return outputs


_PAIR_TASK_MODALITY: dict[str, str] = {
    "document_text_pair": "document",
    "pdf_text_pair": "document",
    "image_text_pair": "image",
    "audio_text_pair": "audio",
    "video_text_pair": "video",
}
_PAIRWISE_TASKS = frozenset(
    {
        *_PAIR_TASK_MODALITY,
        "multimodal_retrieval",
        "cross_modal_consistency",
    }
)


def _build_contrastive_outputs(
    *,
    batch: CollatedBatch,
    modality_outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Build a square text-to-modality similarity matrix for valid pair rows."""

    text_embedding = modality_outputs.get("text_embedding")
    if text_embedding is None:
        return {}

    valid_indices: list[int] = []
    right_embeddings: list[torch.Tensor] = []
    task_types = batch.task_types or []
    for index, task_type in enumerate(task_types):
        if task_type not in _PAIRWISE_TASKS:
            continue
        modality = _PAIR_TASK_MODALITY.get(task_type)
        if modality is None:
            modality = _available_target_modality(
                batch=batch,
                index=index,
                modality_outputs=modality_outputs,
            )
        if modality is None:
            continue
        embedding = modality_outputs.get(f"{modality}_embedding")
        if embedding is None or index >= embedding.shape[0]:
            continue
        if not _row_has_modality(batch=batch, modality=modality, index=index):
            continue
        valid_indices.append(index)
        right_embeddings.append(embedding[index])

    if not valid_indices:
        return {}

    index_tensor = torch.tensor(
        valid_indices,
        device=text_embedding.device,
        dtype=torch.long,
    )
    left = nn.functional.normalize(
        text_embedding.index_select(0, index_tensor),
        dim=-1,
    )
    right = nn.functional.normalize(torch.stack(right_embeddings), dim=-1)
    return {
        "contrastive_logits": left @ right.transpose(0, 1),
        "contrastive_row_indices": index_tensor,
    }


def _available_target_modality(
    *,
    batch: CollatedBatch,
    index: int,
    modality_outputs: dict[str, torch.Tensor],
) -> str | None:
    for modality in ("document", "image", "audio", "video"):
        if f"{modality}_embedding" not in modality_outputs:
            continue
        if _row_has_modality(batch=batch, modality=modality, index=index):
            return modality
    return None


def _row_has_modality(
    *,
    batch: CollatedBatch,
    modality: str,
    index: int,
) -> bool:
    mask = getattr(batch, f"{modality}_mask", None)
    if mask is None or not torch.is_tensor(mask) or index >= mask.shape[0]:
        return False
    return bool(mask[index].item())
