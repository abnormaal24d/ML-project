"""Audio token representation for speech and streaming tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from config.environment.default_values import DEFAULT_AUDIO_SAMPLE_RATE_HZ


@dataclass(frozen=True, slots=True)
class AudioTokenBatch:
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    frame_ms: int
    sample_rate: int


class AudioTokenizer:
    """Frame waveform into continuous or simple discrete audio tokens."""

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE_HZ,
        frame_ms: int = 20,
        hop_ms: int = 20,
        codebook_size: int = 1024,
        n_codebooks: int = 1,
        mode: str = "continuous",
    ) -> None:
        """Initialize audio token framing.

        ``sample_rate`` defaults to the shared audio sample rate in hertz.
        Frame and hop defaults are measured in milliseconds.
        """

        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.hop_ms = int(hop_ms)
        self.codebook_size = int(codebook_size)
        if int(n_codebooks) != 1:
            raise ValueError(
                "the current audio tokenizer supports exactly one codebook"
            )
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"continuous", "discrete"}:
            raise ValueError("mode must be 'continuous' or 'discrete'")
        self.mode = normalized_mode

    def encode(self, waveform: torch.Tensor) -> AudioTokenBatch:
        """Return audio tokens from a [batch, channels, samples] waveform."""

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3:
            raise ValueError("waveform must be [batch, channels, samples]")
        mono = waveform.mean(dim=1)
        frame_size = max(1, self.sample_rate * self.frame_ms // 1000)
        hop_size = max(1, self.sample_rate * self.hop_ms // 1000)
        if mono.shape[-1] < frame_size:
            # ``Tensor.unfold`` rejects windows that exceed the source length.
            # Pad first so even a short (or empty) waveform has one real frame.
            mono = torch.nn.functional.pad(
                mono,
                (0, frame_size - mono.shape[-1]),
            )
        frames = mono.unfold(dimension=-1, size=frame_size, step=hop_size)
        energy = frames.square().mean(dim=-1).sqrt()
        if frame_size == 1:
            # There are no adjacent samples in a one-sample frame.  Taking the
            # mean of that empty dimension would produce NaN tokens.
            zero_crossings = torch.zeros_like(energy)
        else:
            zero_crossings = (
                frames[..., 1:]
                .sign()
                .ne(frames[..., :-1].sign())
                .to(dtype=frames.dtype)
                .mean(dim=-1)
            )
        tokens = torch.stack([energy, zero_crossings], dim=-1)
        if self.mode == "discrete":
            scaled = energy.clamp(0.0, 1.0) * max(1, self.codebook_size - 1)
            discrete = scaled.round().to(dtype=torch.long).unsqueeze(1)
            return AudioTokenBatch(
                tokens=discrete,
                attention_mask=torch.ones(
                    discrete.shape[0],
                    discrete.shape[-1],
                    dtype=torch.bool,
                    device=discrete.device,
                ),
                frame_ms=self.frame_ms,
                sample_rate=self.sample_rate,
            )
        return AudioTokenBatch(
            tokens=tokens,
            attention_mask=torch.ones(
                tokens.shape[:2],
                dtype=torch.bool,
                device=tokens.device,
            ),
            frame_ms=self.frame_ms,
            sample_rate=self.sample_rate,
        )


__all__ = ["AudioTokenBatch", "AudioTokenizer"]
