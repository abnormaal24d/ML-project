"""Dataset-owned media loaders decode training tensors in-package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mmcrawler_datasets.collation.tensor_ops import (
    mask_audio,
    sample_generator,
)
from mmcrawler_datasets.tensors import (
    AudioTensorLoader,
    ImageTensorLoader,
    VideoDecodeError,
    VideoTensorLoader,
)


def test_audio_loader_resamples_and_pads(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / "tone.wav"
    sample_rate = 22050
    duration_seconds = 0.25
    t = np.linspace(
        0,
        duration_seconds,
        int(sample_rate * duration_seconds),
        endpoint=False,
        dtype=np.float32,
    )
    stereo = np.stack(
        [np.sin(2 * np.pi * 440 * t), np.cos(2 * np.pi * 440 * t)], axis=1
    )
    soundfile.write(str(path), stereo, sample_rate)

    loader = AudioTensorLoader(target_sample_rate=16_000, num_samples=8_000)
    waveform = loader.load(path=path)
    assert waveform.shape == (1, 8_000)
    assert torch.isfinite(waveform).all()
    assert float(waveform.abs().max()) <= 1.0


def test_audio_loader_rejects_missing_file(tmp_path: Path) -> None:
    loader = AudioTensorLoader(target_sample_rate=16_000, num_samples=100)
    with pytest.raises(FileNotFoundError):
        loader.load(path=tmp_path / "missing.wav")


def test_video_loader_samples_frames(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(12):
        frame = np.full(
            (48, 64, 3), fill_value=(index * 20) % 255, dtype=np.uint8
        )
        writer.write(frame)
    writer.release()

    loader = VideoTensorLoader(frame_count=4, image_size=32)
    video = loader.load(path=path)
    assert video.shape == (4, 3, 32, 32)
    assert torch.isfinite(video).all()
    assert float(video.min()) >= 0.0
    assert float(video.max()) <= 1.0


def test_video_loader_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    loader = VideoTensorLoader(frame_count=2, image_size=16)
    with pytest.raises((VideoDecodeError, FileNotFoundError)):
        loader.load(path=path)


def test_image_loader_returns_chw(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "img.png"
    Image.new("RGB", (40, 20), color=(10, 20, 30)).save(path)
    loader = ImageTensorLoader(image_size=16)
    image = loader.load(path=path)
    assert image.shape == (3, 16, 16)


def test_mask_audio_is_deterministic_for_sample_seed() -> None:
    audio = torch.randn(1, 1600)
    first = mask_audio(
        audio,
        probability=0.5,
        generator=sample_generator(
            base_seed=7,
            epoch=1,
            sample_id="s1",
            operation="audio_mask",
        ),
    )
    second = mask_audio(
        audio,
        probability=0.5,
        generator=sample_generator(
            base_seed=7,
            epoch=1,
            sample_id="s1",
            operation="audio_mask",
        ),
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[2], second[2])
