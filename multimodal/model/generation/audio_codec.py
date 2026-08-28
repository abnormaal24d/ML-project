"""Scratch-trained audio codec with residual vector quantization."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from config.multimodal.generation_settings import AudioCodecSettings


_AUDIO_CODEC_STAGES = 4


def _codec_stride_factors(config: "AudioCodecSettings") -> tuple[int, ...]:
    """Derive one symmetric, exact stride plan from codec configuration.

    The encoder and decoder must use the same factors in opposite order.  The
    prior hard-coded plans reduced by 32 but reconstructed by 16, while also
    ignoring ``downsample_factor`` entirely.  Factoring the configured value
    keeps the temporal contract explicit for all supported codec settings.
    """

    factor = int(config.downsample_factor)
    if factor <= 0:
        raise ValueError(
            "audio codec downsample_factor must be greater than zero"
        )

    factors: list[int] = []
    remaining = factor
    divisor = 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)

    # Distribute prime factors over a fixed number of stages, keeping individual
    # kernels practical while preserving the exact configured product.
    strides = [1] * _AUDIO_CODEC_STAGES
    for prime in sorted(factors, reverse=True):
        target = min(
            range(_AUDIO_CODEC_STAGES), key=lambda index: strides[index]
        )
        strides[target] *= prime
    result = tuple(strides)
    if prod(result) != factor:
        raise ValueError(
            "audio codec stride plan does not match downsample_factor"
        )
    return result


@dataclass(frozen=True, slots=True)
class AudioCodecOutput:
    """Output from the audio codec encoder."""

    tokens: torch.Tensor  # [B, T, n_codebooks] quantized token indices
    latents: torch.Tensor  # [B, D, T] continuous latents before quantization
    quantized: torch.Tensor  # [B, T, D] quantized latents
    commitment_loss: torch.Tensor  # scalar commitment loss
    codebook_loss: torch.Tensor  # scalar codebook loss
    global_embedding: torch.Tensor  # [B, D] global embedding


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with multiple codebooks."""

    def __init__(self, *, config: "AudioCodecSettings") -> None:
        super().__init__()
        self.n_codebooks = int(config.n_codebooks)
        self.codebook_size = int(config.codebook_size)
        self.latent_dim = int(config.latent_dim)
        self.token_dim = int(config.token_dim)
        self.commitment_weight = float(config.commitment_weight)

        # Input projection to token_dim
        self.input_projection = nn.Linear(config.latent_dim, config.token_dim)

        # Codebooks
        self.codebooks = nn.ModuleList(
            [
                nn.Embedding(config.codebook_size, config.token_dim)
                for _ in range(config.n_codebooks)
            ]
        )

        # Output projection back to latent_dim
        self.output_projection = nn.Linear(config.token_dim, config.latent_dim)

    def forward(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize latents using residual vector quantization.

        Args:
            latents: [B, D, T] continuous latents from encoder

        Returns:
            quantized: [B, T, D] quantized latents
            tokens: [B, T, n_codebooks] token indices
            commitment_loss: scalar
            codebook_loss: scalar
        """
        if latents.ndim != 3:
            raise ValueError("audio codec latents must have shape [B, D, T]")
        _batch, latent_dim, _frames = latents.shape
        if latent_dim != self.latent_dim:
            raise ValueError(
                "audio codec latent dimension must match the configured codec: "
                f"expected={self.latent_dim}, got={latent_dim}"
            )
        x = latents.transpose(1, 2)  # [B, T, D]

        # Project to token dim
        x = self.input_projection(x)  # [B, T, token_dim]

        quantized = torch.zeros_like(x)
        token_indices: list[torch.Tensor] = []
        commitment_loss = x.new_zeros(())
        codebook_loss = x.new_zeros(())

        residual = x
        for codebook in self.codebooks:
            embedding = cast(nn.Embedding, codebook)
            # Find nearest codebook entry
            distances = torch.cdist(
                residual, embedding.weight
            )  # [B, T, codebook_size]
            tokens_i = distances.argmin(dim=-1)  # [B, T]
            token_indices.append(tokens_i)

            quantized_i = embedding(tokens_i)  # [B, T, token_dim]

            # Compute losses
            commitment_loss += torch.mean(
                (quantized_i.detach() - residual) ** 2
            )
            codebook_loss += torch.mean((quantized_i - residual.detach()) ** 2)

            quantized = quantized + quantized_i
            # Residual assignments are discrete; keep codebook optimization in
            # its dedicated loss rather than leaking it through later searches.
            residual = residual - quantized_i.detach()

        # Preserve quantized values in the forward pass while routing
        # reconstruction gradients through the encoder projection.  Without
        # this straight-through path, argmin/codebook lookup disconnects the
        # reconstruction loss from the encoder.
        quantized_st = x + (quantized - x).detach()

        # Project back to latent_dim.
        quantized = self.output_projection(quantized_st)  # [B, T, D]
        quantized = quantized.transpose(1, 2)  # [B, D, T]

        tokens = torch.stack(token_indices, dim=-1)  # [B, T, n_codebooks]

        return quantized, tokens, commitment_loss, codebook_loss


class ScratchAudioCodecEncoder(nn.Module):
    """Encode audio waveforms into latent sequences."""

    def __init__(self, *, config: "AudioCodecSettings") -> None:
        super().__init__()
        self.latent_dim = int(config.latent_dim)
        self.stride_factors = _codec_stride_factors(config)
        self.downsample_factor = prod(self.stride_factors)
        self.hidden_dim = int(config.hidden_dim)

        # Input projection
        self.input_conv = nn.Conv1d(
            1, config.hidden_dim // 4, kernel_size=7, stride=1, padding=3
        )
        # Codec sequences can legitimately shrink to one latent frame per
        # sample. GroupNorm remains defined there, unlike BatchNorm in training
        # mode, and does not make a codec reconstruction depend on batch peers.
        self.input_norm = nn.GroupNorm(1, config.hidden_dim // 4)

        # Strided convolutions for downsampling
        self.downsample_blocks = nn.ModuleList()
        current_dim = config.hidden_dim // 4
        for i, stride in enumerate(self.stride_factors):
            next_dim = min(config.hidden_dim * (2**i), config.hidden_dim * 4)
            self.downsample_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        current_dim,
                        next_dim,
                        kernel_size=stride,
                        stride=stride,
                    ),
                    nn.GroupNorm(1, next_dim),
                    nn.GELU(),
                    nn.Conv1d(next_dim, next_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(1, next_dim),
                    nn.GELU(),
                )
            )
            current_dim = next_dim

        # Final projection to latent_dim
        self.output_conv = nn.Conv1d(
            current_dim, config.latent_dim, kernel_size=1
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform to latents.

        Args:
            waveform: [B, 1, T] audio waveform

        Returns:
            latents: [B, D, T] latent sequence
        """
        x = waveform
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = nn.functional.gelu(x)

        for block in self.downsample_blocks:
            x = block(x)

        latents = self.output_conv(x)
        return cast(torch.Tensor, latents)


class ScratchAudioCodecDecoder(nn.Module):
    """Decode quantized latents back to waveform."""

    def __init__(self, *, config: "AudioCodecSettings") -> None:
        super().__init__()
        self.latent_dim = int(config.latent_dim)
        encoder_strides = _codec_stride_factors(config)
        self.stride_factors = tuple(reversed(encoder_strides))
        self.upsample_factor = prod(self.stride_factors)
        self.hidden_dim = int(config.hidden_dim)

        # Input projection
        self.input_conv = nn.Conv1d(
            config.latent_dim, config.hidden_dim * 4, kernel_size=1
        )

        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        current_dim = config.hidden_dim * 4
        for i, stride in enumerate(self.stride_factors):
            next_dim = max(config.hidden_dim // (2 ** (3 - i)), 64)
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        current_dim,
                        next_dim,
                        kernel_size=stride,
                        stride=stride,
                    ),
                    nn.GroupNorm(1, next_dim),
                    nn.GELU(),
                    nn.Conv1d(next_dim, next_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(1, next_dim),
                    nn.GELU(),
                )
            )
            current_dim = next_dim

        # Final output
        self.output_conv = nn.Conv1d(current_dim, 1, kernel_size=7, padding=3)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to waveform.

        Args:
            latents: [B, D, T] quantized latents

        Returns:
            waveform: [B, 1, T] reconstructed waveform
        """
        x = latents
        x = self.input_conv(latents)

        for block in self.upsample_blocks:
            x = block(x)

        waveform = cast(torch.Tensor, self.output_conv(x))
        return torch.tanh(waveform)  # Output in [-1, 1]


class ScratchAudioCodec(nn.Module):
    """Complete audio codec with encoder, RVQ quantizer, and decoder."""

    def __init__(self, *, config: "AudioCodecSettings") -> None:
        super().__init__()
        self.config = config
        self.encoder = ScratchAudioCodecEncoder(config=config)
        self.quantizer = ResidualVectorQuantizer(config=config)
        self.decoder = ScratchAudioCodecDecoder(config=config)
        self.config = config

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform to continuous latents."""
        return cast(torch.Tensor, self.encoder(waveform))

    def quantize(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize latents using RVQ."""
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            self.quantizer(latents),
        )

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents to waveform."""
        return cast(torch.Tensor, self.decoder(latents))

    def forward(
        self, waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Full encode-quantize-decode cycle for reconstruction loss.

        Args:
            waveform: [B, 1, T] input waveform

        Returns:
            reconstructed: [B, 1, T] reconstructed waveform
            quant_info: dict with tokens, losses, etc.
        """
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("audio codec waveform must have shape [B, 1, T]")
        original_length = int(waveform.shape[-1])
        if original_length <= 0:
            raise ValueError(
                "audio codec waveform must contain at least one sample"
            )

        factor = self.encoder.downsample_factor
        padded_length = ((original_length + factor - 1) // factor) * factor
        padded_waveform = nn.functional.pad(
            waveform,
            (0, padded_length - original_length),
        )

        latents = self.encode(padded_waveform)
        quantized, tokens, commitment_loss, codebook_loss = self.quantize(
            latents
        )
        reconstructed = self.decode(quantized)[..., :original_length]

        return reconstructed, {
            "tokens": tokens,  # [B, T, n_codebooks]
            "quantized_latents": quantized,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
        }

    def reconstruct_loss(
        self, waveform: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss (L1 + multi-resolution STFT)."""
        # L1 loss
        l1_loss = nn.functional.l1_loss(reconstructed, waveform)

        # Multi-resolution STFT loss
        stft_loss = self._multi_resolution_stft_loss(waveform, reconstructed)

        return l1_loss + stft_loss

    def _multi_resolution_stft_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute multi-resolution STFT loss."""
        stft_loss = x.new_zeros(())
        for n_fft in [512, 1024, 2048]:
            x_stft = torch.stft(
                x.squeeze(1),
                n_fft=n_fft,
                hop_length=n_fft // 4,
                window=torch.hann_window(n_fft, device=x.device),
                return_complex=True,
            )
            y_stft = torch.stft(
                y.squeeze(1),
                n_fft=n_fft,
                hop_length=n_fft // 4,
                window=torch.hann_window(n_fft, device=y.device),
                return_complex=True,
            )

            x_mag = torch.abs(x_stft)
            y_mag = torch.abs(y_stft)

            # Spectral convergence
            sc = torch.norm(y_mag - x_mag, p="fro") / (
                torch.norm(x_mag, p="fro") + 1e-8
            )
            # Log magnitude loss
            log_loss = nn.functional.l1_loss(
                torch.log(y_mag + 1e-8), torch.log(x_mag + 1e-8)
            )

            stft_loss += sc + log_loss

        return stft_loss / 3.0
