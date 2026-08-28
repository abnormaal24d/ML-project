"""Latent diffusion model for image generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from config.multimodal.generation_settings import ImageGeneratorSettings


@dataclass(frozen=True, slots=True)
class ImageDiffusionOutput:
    """Output from the latent diffusion model."""

    # Depending on ``prediction_type`` this is epsilon, v, or the clean
    # sample prediction.
    prediction: torch.Tensor  # [B, C, H, W] predicted diffusion target
    timestep: torch.Tensor  # [B] timestep used
    target: torch.Tensor | None = None  # target matching ``prediction_type``
    noise: torch.Tensor | None = None  # sampled epsilon used for training
    noisy_latents: torch.Tensor | None = None


class ImageLatentDiffusion(nn.Module):
    """Latent diffusion model for image generation.

    Predicts noise in the latent space of a pretrained image codec.
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor

    def __init__(
        self,
        *,
        config: "ImageGeneratorSettings",
        latent_channels: int,
        conditioning_dim: int,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.conditioning_dim = int(conditioning_dim)
        if self.latent_channels <= 0 or self.conditioning_dim <= 0:
            raise ValueError(
                "latent_channels and conditioning_dim must be greater than zero"
            )
        self.hidden_dim = int(config.hidden_dim)
        self.num_layers = int(config.num_layers)
        self.num_heads = int(config.num_heads)
        self.num_train_timesteps = int(config.num_train_timesteps)
        self.prediction_type = str(config.prediction_type)
        if self.prediction_type not in {"epsilon", "v_prediction", "sample"}:
            raise ValueError(
                "image diffusion prediction_type must be epsilon, v_prediction, "
                "or sample"
            )

        # Latent patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=self.latent_channels,
            out_channels=config.hidden_dim,
            kernel_size=1,
        )

        # Timestep embedding
        self.timestep_embedding = nn.Embedding(
            config.num_train_timesteps,
            config.hidden_dim,
        )

        # Text conditioning projection
        self.text_projection = nn.Linear(
            self.conditioning_dim,
            config.hidden_dim,
        )

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Output noise prediction head
        self.noise_prediction_head = nn.Sequential(
            nn.GroupNorm(1, config.hidden_dim),
            nn.Conv2d(
                config.hidden_dim,
                self.latent_channels,
                kernel_size=1,
            ),
        )

        # Diffusion schedule
        self.register_buffer(
            "betas",
            self._make_beta_schedule(config.num_train_timesteps),
        )
        self.register_buffer(
            "alphas",
            1.0 - self.betas,
        )
        self.register_buffer(
            "alphas_cumprod",
            torch.cumprod(self.alphas, dim=0),
        )
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(self.alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - self.alphas_cumprod),
        )

    def _make_beta_schedule(self, num_timesteps: int) -> torch.Tensor:
        """Create linear beta schedule."""
        return torch.linspace(0.0001, 0.02, num_timesteps)

    def forward(
        self,
        latents: torch.Tensor,  # [B, C, H, W] clean latents
        text_embedding: torch.Tensor,  # [B, D] global text embedding
        timestep: torch.Tensor | None = None,  # [B] or None for random
        noise: torch.Tensor | None = None,
    ) -> ImageDiffusionOutput:
        """Forward pass for training.

        Args:
            latents: [B, C, H, W] clean latents from image codec
            text_embedding: [B, D] global text embedding
            timestep: [B] timestep indices, or None for random sampling

        Returns:
            ImageDiffusionOutput with predicted noise and timestep used
        """
        batch_size = latents.shape[0]
        device = latents.device

        # Sample random timesteps if not provided
        if timestep is None:
            timestep = torch.randint(
                0,
                self.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            )
        timestep = self._validated_timestep(
            timestep=timestep,
            batch_size=batch_size,
            device=device,
        )

        if noise is None:
            noise = torch.randn_like(latents)
        elif noise.shape != latents.shape:
            raise ValueError("diffusion noise must match latent shape")
        noisy_latents = self.q_sample(
            latents=latents,
            timestep=timestep,
            noise=noise,
        )
        prediction = self.predict_noise(
            latents=noisy_latents,
            text_embedding=text_embedding,
            timestep=timestep,
        )

        return ImageDiffusionOutput(
            prediction=prediction,
            timestep=timestep,
            target=self.training_target(
                latents=latents,
                noise=noise,
                timestep=timestep,
            ),
            noise=noise,
            noisy_latents=noisy_latents,
        )

    def training_target(
        self,
        *,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Return the training target selected by ``prediction_type``.

        All three target parameterizations describe the same noisy sample.  By
        making the conversion explicit here, training no longer silently trains
        an epsilon predictor when configuration requested v- or x0-prediction.
        """

        self._validate_latents(latents)
        if noise.shape != latents.shape:
            raise ValueError("diffusion noise must match latent shape")
        timestep = self._validated_timestep(
            timestep=timestep,
            batch_size=latents.shape[0],
            device=latents.device,
        )
        sqrt_alpha, sqrt_one_minus_alpha = self._schedule_terms(
            timestep=timestep,
            dtype=latents.dtype,
            device=latents.device,
        )
        if self.prediction_type == "epsilon":
            return noise
        if self.prediction_type == "sample":
            return latents
        return sqrt_alpha * noise - sqrt_one_minus_alpha * latents

    def prediction_to_original_and_noise(
        self,
        *,
        model_prediction: torch.Tensor,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a configured model prediction to x0 and epsilon.

        Sampling always needs both terms for the DDIM update.  The conversion
        is deliberately separate from ``predict_noise`` so sampling cannot add
        fresh noise or reinterpret a configured parameterization as epsilon.
        """

        self._validate_latents(noisy_latents)
        if model_prediction.shape != noisy_latents.shape:
            raise ValueError(
                "diffusion model prediction must match noisy latent shape"
            )
        if model_prediction.device != noisy_latents.device:
            raise ValueError(
                "diffusion model prediction and noisy latents must share a device"
            )
        timestep = self._validated_timestep(
            timestep=timestep,
            batch_size=noisy_latents.shape[0],
            device=noisy_latents.device,
        )
        sqrt_alpha, sqrt_one_minus_alpha = self._schedule_terms(
            timestep=timestep,
            dtype=noisy_latents.dtype,
            device=noisy_latents.device,
        )

        if self.prediction_type == "epsilon":
            predicted_noise = model_prediction
            predicted_original = (
                noisy_latents - sqrt_one_minus_alpha * predicted_noise
            ) / sqrt_alpha
            return predicted_original, predicted_noise
        if self.prediction_type == "sample":
            predicted_original = model_prediction
            predicted_noise = (
                noisy_latents - sqrt_alpha * predicted_original
            ) / sqrt_one_minus_alpha
            return predicted_original, predicted_noise

        predicted_original = (
            sqrt_alpha * noisy_latents
            - sqrt_one_minus_alpha * model_prediction
        )
        predicted_noise = (
            sqrt_alpha * model_prediction
            + sqrt_one_minus_alpha * noisy_latents
        )
        return predicted_original, predicted_noise

    def q_sample(
        self,
        *,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Add caller-provided training noise to clean codec latents."""

        self._validate_latents(latents)
        timestep = self._validated_timestep(
            timestep=timestep,
            batch_size=latents.shape[0],
            device=latents.device,
        )
        if noise.shape != latents.shape:
            raise ValueError("diffusion noise must match latent shape")
        sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t = (
            self._schedule_terms(
                timestep=timestep,
                dtype=latents.dtype,
                device=latents.device,
            )
        )
        return (
            sqrt_alphas_cumprod_t * latents
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def predict_noise(
        self,
        *,
        latents: torch.Tensor,
        text_embedding: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise for already-noisy latents without adding new noise."""

        self._validate_latents(latents)
        batch_size, _, height, width = latents.shape
        timestep = self._validated_timestep(
            timestep=timestep,
            batch_size=batch_size,
            device=latents.device,
        )
        if text_embedding.shape != (batch_size, self.conditioning_dim):
            raise ValueError(
                "text_embedding must have shape "
                f"[{batch_size}, {self.conditioning_dim}]"
            )
        if text_embedding.device != latents.device:
            raise ValueError("text_embedding and latents must share a device")

        # Patch embedding
        x = self.patch_embedding(latents)  # [B, hidden_dim, H, W]

        # Flatten spatial dimensions for transformer
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, hidden_dim]

        # Timestep embedding
        t_emb = self.timestep_embedding(timestep)  # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, hidden_dim]

        # Text conditioning
        text_cond = self.text_projection(text_embedding)  # [B, hidden_dim]
        text_cond = text_cond.unsqueeze(1)  # [B, 1, hidden_dim]

        # Combine with timestep and text conditioning
        x = x + t_emb + text_cond

        # Transformer encoding
        x = self.transformer(x)

        # Reshape back to spatial
        x = x.transpose(1, 2).reshape(
            batch_size,
            self.hidden_dim,
            height,
            width,
        )

        # Noise prediction
        return cast(torch.Tensor, self.noise_prediction_head(x))

    def _validate_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 4:
            raise ValueError("diffusion latents must have shape [B, C, H, W]")
        if latents.shape[1] != self.latent_channels:
            raise ValueError(
                "diffusion latent channel count must match the image codec: "
                f"expected={self.latent_channels}, got={latents.shape[1]}"
            )

    def _schedule_terms(
        self,
        *,
        timestep: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return broadcastable sqrt(alpha) terms on the active tensor device."""

        # Derive from the canonical cumulative schedule rather than relying on
        # cached square-root buffers.  This keeps all conversions coherent when
        # an inference schedule is restored or adjusted in-place.
        alpha = self.alphas_cumprod[timestep].to(
            device=device,
            dtype=dtype,
        )
        sqrt_alpha = torch.sqrt(alpha).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = torch.sqrt((1.0 - alpha).clamp_min(0.0)).view(
            -1, 1, 1, 1
        )
        return sqrt_alpha, sqrt_one_minus_alpha

    def _validated_timestep(
        self,
        *,
        timestep: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if timestep.ndim != 1 or timestep.shape[0] != batch_size:
            raise ValueError(
                "diffusion timestep must have shape [B] matching latents"
            )
        if timestep.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("diffusion timestep must use an integer dtype")
        timestep = timestep.to(device=device, dtype=torch.long)
        if bool(
            ((timestep < 0) | (timestep >= self.num_train_timesteps)).any()
        ):
            raise ValueError(
                "diffusion timestep is outside the training schedule"
            )
        return timestep
