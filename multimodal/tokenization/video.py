"""Canonical video tokenizer schema and frame-grid implementation.

The frame codec is injected so no process-global state is required. This keeps
the tokenizer safe under parallel tests, multiple model instances, distinct
codecs within one process and distributed workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray


class VideoFrameCodec(Protocol):
    """Reads and writes raw RGB video frames."""

    def read_frames(
        self,
        *,
        video_path: Path,
        frame_count: int,
        height: int,
        width: int,
    ) -> list[NDArray[np.uint8]]: ...

    def write_frames(
        self,
        *,
        frames_rgb: list[NDArray[np.uint8]],
        output_path: Path,
        fps: int,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class VideoTokenizationResult:
    """Tokenized video frames and the metadata needed for decoding."""

    tokens: torch.Tensor  # [T, Hq, Wq]
    fps: int
    frames: int
    height: int
    width: int
    tokenizer_name: str
    vocab_size: int


class VideoTokenizer(Protocol):
    """Schema consumed by video materialization and inference."""

    name: str
    vocab_size: int

    def encode(self, video_path: Path) -> VideoTokenizationResult: ...

    def decode(
        self,
        tokens: torch.Tensor,
        *,
        output_path: Path,
        fps: int,
        height: int,
        width: int,
    ) -> Path: ...


class VideoFrameGridTokenizer:
    """Deterministic frame-grid tokenizer backed by real video frames."""

    name = "video_frame_grid_v1"

    def __init__(
        self,
        *,
        frame_codec: VideoFrameCodec,
        vocab_size: int = 4096,
        grid_height: int = 16,
        grid_width: int = 16,
        frame_count: int = 8,
        height: int = 128,
        width: int = 128,
        fps: int = 8,
    ) -> None:
        self._frame_codec = frame_codec
        self.vocab_size = int(vocab_size)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.frame_count = int(frame_count)
        self.height = int(height)
        self.width = int(width)
        self.fps = int(fps)
        self._bins = max(2, int(round(self.vocab_size ** (1.0 / 3.0))))

    def encode(self, video_path: Path) -> VideoTokenizationResult:
        frames = self._frame_codec.read_frames(
            video_path=video_path,
            frame_count=self.frame_count,
            height=self.height,
            width=self.width,
        )
        token_frames = [
            self._frame_to_tokens(frame_rgb=frame) for frame in frames
        ]
        tokens = torch.stack(token_frames, dim=0).to(dtype=torch.long)
        return VideoTokenizationResult(
            tokens=tokens,
            fps=self.fps,
            frames=tokens.shape[0],
            height=self.height,
            width=self.width,
            tokenizer_name=self.name,
            vocab_size=self.vocab_size,
        )

    def decode(
        self,
        tokens: torch.Tensor,
        *,
        output_path: Path,
        fps: int,
        height: int,
        width: int,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames = self._tokens_to_frames(
            tokens=tokens.detach().cpu().long(),
            height=height,
            width=width,
        )
        self._frame_codec.write_frames(
            frames_rgb=frames,
            output_path=output_path,
            fps=fps,
        )
        return output_path

    def _frame_to_tokens(
        self, *, frame_rgb: NDArray[np.uint8]
    ) -> torch.Tensor:
        patch_h = max(1, self.height // self.grid_height)
        patch_w = max(1, self.width // self.grid_width)
        tokens = torch.zeros(
            self.grid_height, self.grid_width, dtype=torch.long
        )
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                patch = frame_rgb[
                    row * patch_h : (row + 1) * patch_h,
                    col * patch_w : (col + 1) * patch_w,
                    :,
                ]
                mean_rgb = patch.mean(axis=(0, 1))
                tokens[row, col] = self._quantize_rgb(mean_rgb)
        return tokens

    def _tokens_to_frames(
        self,
        *,
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> list[NDArray[np.uint8]]:
        if tokens.dim() == 4:
            tokens = tokens[0]
        if tokens.dim() != 3:
            raise ValueError("video tokens must have shape [T, H, W]")

        grid_height = int(tokens.shape[1])
        grid_width = int(tokens.shape[2])
        patch_h = max(1, height // grid_height)
        patch_w = max(1, width // grid_width)
        frames: list[NDArray[np.uint8]] = []
        for frame_tokens in tokens:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for row in range(grid_height):
                for col in range(grid_width):
                    color = self._dequantize_token(int(frame_tokens[row, col]))
                    frame[
                        row * patch_h : (row + 1) * patch_h,
                        col * patch_w : (col + 1) * patch_w,
                        :,
                    ] = color
            frames.append(frame)
        return frames

    def _quantize_rgb(self, rgb: NDArray[np.float64]) -> int:
        bins = self._bins
        red, green, blue = (int(np.clip(channel, 0, 255)) for channel in rgb)
        red_bin = min(bins - 1, red * bins // 256)
        green_bin = min(bins - 1, green * bins // 256)
        blue_bin = min(bins - 1, blue * bins // 256)
        token_id = red_bin * bins * bins + green_bin * bins + blue_bin
        return min(self.vocab_size - 1, token_id)

    def _dequantize_token(self, token_id: int) -> NDArray[np.uint8]:
        bins = self._bins
        token_id = int(token_id) % self.vocab_size
        blue_bin = token_id % bins
        token_id //= bins
        green_bin = token_id % bins
        red_bin = token_id // bins
        red = int((red_bin + 0.5) * 256 / bins)
        green = int((green_bin + 0.5) * 256 / bins)
        blue = int((blue_bin + 0.5) * 256 / bins)
        return np.array([red, green, blue], dtype=np.uint8)


__all__ = [
    "VideoFrameCodec",
    "VideoFrameGridTokenizer",
    "VideoTokenizationResult",
    "VideoTokenizer",
]
