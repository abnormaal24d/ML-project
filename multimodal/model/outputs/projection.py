"""Projection heads and modality projection execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from multimodal.model.contracts import CollatedBatch
from multimodal.model.encoders.runner import (
    required_embedding,
    text_sequence_length,
    text_vocab_size,
    video_generation_vocab_size,
)
from multimodal.model.outputs.batch import optional_batch_mask

if TYPE_CHECKING:
    from config.multimodal.model_settings import ModelSettings
    from multimodal.model.encoders.runner import ModalityEncoderRunner


class ProjectionHeads(nn.Module):
    """Own all supervised, conditioning, and generation heads."""

    def __init__(self, *, config: ModelSettings) -> None:
        super().__init__()
        self.fusion_dim = int(config.fusion_dim)
        self.enabled_modalities = tuple(config.enabled_modalities)
        self.image_size = int(config.raw_image_size)
        self.image_output_channels = int(config.image_codec.output_channels)

        self.classifier = nn.Linear(config.fusion_dim, config.num_classes)
        self.projection = nn.Linear(
            config.fusion_dim,
            config.projection_dim,
        )
        self.emotion_head = nn.Linear(config.fusion_dim, 64)
        self.speaker_projection = nn.Linear(
            config.fusion_dim,
            config.projection_dim,
        )
        self.layout_box_head = nn.Linear(config.fusion_dim, 4)
        self.object_box_head = nn.Linear(config.fusion_dim, 4)
        self.document_fallback_projection = nn.Linear(
            config.fusion_dim,
            config.fusion_dim,
        )
        self.layout_condition_projection = nn.Linear(4, config.fusion_dim)
        self.mask_condition_projection = nn.Sequential(
            nn.Linear(2, config.fusion_dim),
            nn.GELU(),
            nn.Linear(config.fusion_dim, config.fusion_dim),
        )
        self.audio_token_head = nn.Linear(
            config.fusion_dim,
            config.audio_tokenizer.codebook_size,
        )

        configured_audio_codebooks = int(config.audio_tokenizer.n_codebooks)
        if configured_audio_codebooks != 1:
            raise ValueError(
                "audio generation head requires exactly one tokenizer codebook"
            )
        self.audio_n_codebooks = 1
        self.audio_generation_frames = max(
            1,
            int(config.generation.audio_codec_rate)
            * max(
                1,
                int(
                    config.raw_audio_num_samples
                    // config.audio_tokenizer.sample_rate
                ),
            ),
        )
        if config.audio_tokenizer.enabled:
            self.audio_position_embedding: nn.Embedding | None = nn.Embedding(
                self.audio_generation_frames,
                config.fusion_dim,
            )
            self.audio_codebook_embedding: nn.Embedding | None = nn.Embedding(
                self.audio_n_codebooks,
                config.fusion_dim,
            )
            self.audio_generation_head: nn.Linear | None = nn.Linear(
                config.fusion_dim,
                config.audio_tokenizer.codebook_size,
            )
        else:
            self.audio_position_embedding = None
            self.audio_codebook_embedding = None
            self.audio_generation_head = None
        self.prosody_head = nn.Linear(config.fusion_dim, 4)
        self.safety_head = nn.Linear(config.fusion_dim, 7)
        if config.image_generator.enabled or config.image_decoder.enabled:
            self.image_latent_head: nn.Linear | None = nn.Linear(
                config.fusion_dim,
                config.image_codec.latent_channels,
            )
            self.image_pixel_head: nn.Linear | None = nn.Linear(
                config.fusion_dim,
                self.image_output_channels
                * config.raw_image_size
                * config.raw_image_size,
            )
        else:
            self.image_latent_head = None
            self.image_pixel_head = None

        # Image generation heads (new in Step 6)
        if config.image_generator.enabled:
            # Image codec heads
            self.image_codec_encoder_head: nn.Linear | None = nn.Linear(
                config.fusion_dim,
                config.image_codec.hidden_dim * 2,  # for encoder features
            )
            self.image_codec_decoder_head: nn.Linear | None = nn.Linear(
                config.fusion_dim,
                config.image_codec.latent_channels
                * (
                    config.image_codec.input_resolution
                    // config.image_codec.downsample_factor
                )
                ** 2,
            )
            # Diffusion model heads
            self.diffusion_timestep_embedding: nn.Embedding | None = (
                nn.Embedding(
                    config.image_generator.num_train_timesteps,
                    config.fusion_dim,
                )
            )
            self.diffusion_text_projection: nn.Linear | None = nn.Linear(
                config.fusion_dim,
                config.image_generator.hidden_dim,
            )
            self.diffusion_noise_head: nn.Sequential | None = nn.Sequential(
                nn.LayerNorm(config.fusion_dim),
                nn.Linear(
                    config.fusion_dim, config.image_codec.latent_channels
                ),
            )
        else:
            self.image_codec_encoder_head = None
            self.image_codec_decoder_head = None
            self.diffusion_timestep_embedding = None
            self.diffusion_text_projection = None
            self.diffusion_noise_head = None

        self.text_sequence_length = text_sequence_length(config=config)
        self.text_vocab_size = text_vocab_size(config=config)
        if config.text_decoder.enabled:
            self.text_position_embedding: nn.Embedding | None = None
            self.text_generation_head: nn.Linear | None = None
        else:
            self.text_position_embedding = nn.Embedding(
                self.text_sequence_length,
                config.fusion_dim,
            )
            self.text_generation_head = nn.Linear(
                config.fusion_dim,
                self.text_vocab_size,
            )

        self.video_generation_frames = max(
            1,
            int(config.video_generator.frames),
        )
        self.video_generation_vocab_size = video_generation_vocab_size(
            config=config
        )
        self.video_grid_height = int(
            getattr(config.video_generator, "grid_height", 16)
        )
        self.video_grid_width = int(
            getattr(config.video_generator, "grid_width", 16)
        )
        if config.video_generator.enabled or config.video_decoder.enabled:
            self.video_position_embedding: nn.Embedding | None = nn.Embedding(
                self.video_generation_frames,
                config.fusion_dim,
            )
            self.video_row_embedding: nn.Embedding | None = nn.Embedding(
                self.video_grid_height,
                config.fusion_dim,
            )
            self.video_column_embedding: nn.Embedding | None = nn.Embedding(
                self.video_grid_width,
                config.fusion_dim,
            )
            self.video_generation_head: nn.Linear | None = nn.Linear(
                config.fusion_dim,
                self.video_generation_vocab_size,
            )
        else:
            self.video_position_embedding = None
            self.video_row_embedding = None
            self.video_column_embedding = None
            self.video_generation_head = None


def build_projection_heads(*, config: ModelSettings) -> ProjectionHeads:
    """Build one self-contained output-head collection."""

    return ProjectionHeads(config=config)


@dataclass(frozen=True, slots=True)
class ProjectedModalities:
    encoded_by_modality: dict[str, dict[str, torch.Tensor]]
    modality_embeddings: list[torch.Tensor]
    modality_masks: list[torch.Tensor]
    active_names: list[str]


class ProjectionRunner:
    def __init__(
        self,
        *,
        enabled_modalities: tuple[str, ...],
        projection_heads: ProjectionHeads,
        encoder_runner: ModalityEncoderRunner,
    ) -> None:
        self._enabled_modalities = enabled_modalities
        self._projection_heads = projection_heads
        self._encoder_runner = encoder_runner

    def project(
        self,
        batch: CollatedBatch,
        *,
        encoded_by_modality: dict[str, dict[str, torch.Tensor]],
        modality_row_masks: dict[str, torch.Tensor],
    ) -> ProjectedModalities:
        modality_embeddings: list[torch.Tensor] = []
        modality_masks: list[torch.Tensor] = []
        active_names: list[str] = []
        encoded_by_modality = dict(encoded_by_modality)

        for modality in self._enabled_modalities:
            if modality == "document":
                continue
            row_mask = modality_row_masks.get(modality)
            modality_encoded = encoded_by_modality.get(modality)
            if (
                row_mask is None
                or modality_encoded is None
                or not bool(row_mask.any().item())
            ):
                continue
            embedding = required_embedding(
                encoded=modality_encoded,
                source=f"{modality} encoder",
            )
            modality_embeddings.append(embedding)
            modality_masks.append(row_mask.to(embedding.device))
            active_names.append(modality)

        document_mask = modality_row_masks.get("document")
        document_encoded = encoded_by_modality.get("document")
        if (
            "document" in self._enabled_modalities
            and document_mask is not None
            and document_encoded is not None
            and bool(document_mask.any().item())
        ):
            embedding = required_embedding(
                encoded=document_encoded,
                source="document encoder",
            )
            modality_embeddings.append(embedding)
            modality_masks.append(document_mask.to(embedding.device))
            active_names.append("document")

        for name, item_encoded, row_mask in self._conditioning_embeddings(
            batch=batch,
            encoded_by_modality=encoded_by_modality,
        ):
            embedding = required_embedding(
                encoded=item_encoded,
                source=f"{name} conditioning",
            )
            encoded_by_modality[name] = item_encoded
            modality_embeddings.append(embedding)
            modality_masks.append(row_mask.to(embedding.device))
            active_names.append(name)

        return ProjectedModalities(
            encoded_by_modality=encoded_by_modality,
            modality_embeddings=modality_embeddings,
            modality_masks=modality_masks,
            active_names=active_names,
        )

    def _conditioning_embeddings(
        self,
        *,
        batch: CollatedBatch,
        encoded_by_modality: dict[str, dict[str, torch.Tensor]],
    ) -> list[tuple[str, dict[str, torch.Tensor], torch.Tensor]]:
        text_encoded = encoded_by_modality.get("text")
        if text_encoded is None:
            return []

        text_embedding = required_embedding(
            encoded=text_encoded,
            source="text conditioning",
        )
        conditionings: list[
            tuple[str, dict[str, torch.Tensor], torch.Tensor]
        ] = []

        document_mask = optional_batch_mask(
            batch=batch,
            name="document_mask",
            reference=text_embedding,
        )
        if "document" not in self._enabled_modalities and bool(
            document_mask.any().item()
        ):
            document_embedding = self._encoder_runner.document_embedding(
                batch=batch,
                text_embedding=text_embedding,
                document_mask=document_mask,
            )
            conditionings.append(
                (
                    "document",
                    {"embedding": document_embedding},
                    document_mask,
                )
            )

        layout_mask = optional_batch_mask(
            batch=batch,
            name="layout_mask",
            reference=text_embedding,
        )
        layout_targets = getattr(batch, "layout_box_targets", None)
        if layout_targets is not None and bool(layout_mask.any().item()):
            layout_embedding = (
                self._projection_heads.layout_condition_projection(
                    layout_targets.to(
                        device=text_embedding.device,
                        dtype=text_embedding.dtype,
                    )
                )
            )
            conditionings.append(
                (
                    "layout",
                    {
                        "embedding": layout_embedding
                        * layout_mask.unsqueeze(-1)
                    },
                    layout_mask,
                )
            )

        mask_input_mask = optional_batch_mask(
            batch=batch,
            name="edit_mask_input_mask",
            reference=text_embedding,
        )
        edit_mask = getattr(batch, "edit_mask_tensor", None)
        if edit_mask is not None and bool(mask_input_mask.any().item()):
            mask_tensor = edit_mask.to(
                device=text_embedding.device,
                dtype=text_embedding.dtype,
            )
            coverage = mask_tensor.flatten(1).mean(dim=1)
            variance = mask_tensor.flatten(1).var(dim=1, unbiased=False)
            mask_features = torch.stack((coverage, variance), dim=1)
            mask_embedding = self._projection_heads.mask_condition_projection(
                mask_features
            )
            conditionings.append(
                (
                    "mask",
                    {
                        "embedding": mask_embedding
                        * mask_input_mask.unsqueeze(-1)
                    },
                    mask_input_mask,
                )
            )
        return conditionings
