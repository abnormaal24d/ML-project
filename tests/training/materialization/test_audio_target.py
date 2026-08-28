from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
import torch

from mmcrawler_datasets.materialization.audio_generation import (
    AudioGenerationTargetMaterializer,
)
from mmcrawler_datasets.training_samples.models import TrainingSample
from mmcrawler_datasets.training_samples.targets import TrainingTaskTarget
from multimodal.tokenization.audio import AudioTokenizer


def _write_pcm_wav(path: Path, *, sample_rate: int = 8_000) -> None:
    samples = b"\x00\x00" * (sample_rate // 10)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples)


def test_audio_generation_target_is_materialized_inside_staging_root(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "target.wav"
    _write_pcm_wav(audio_path)
    sample = TrainingSample(
        sample_id="sample-a:speech_to_audio",
        snapshot_id="snapshot-a",
        task_target=TrainingTaskTarget(
            task_type="speech_to_audio",
            target_audio_path="target.wav",
            output_modalities=("audio",),
        ),
    )
    materializer = AudioGenerationTargetMaterializer(
        tokenizer=AudioTokenizer(
            sample_rate=8_000,
            mode="discrete",
            n_codebooks=1,
        ),
        output_root=tmp_path / "staging" / "audio_tokens",
    )

    materialized = materializer.materialize(sample, project_root=tmp_path)

    relative_tokens = materialized.task_target.target_audio_tokens_path
    assert relative_tokens is not None
    tokens_path = tmp_path / relative_tokens
    metadata_path = tokens_path.with_suffix(".json")
    assert tokens_path.is_file()
    assert metadata_path.is_file()
    tokens = torch.load(tokens_path, map_location="cpu", weights_only=True)
    assert tokens.ndim == 3
    assert tokens.shape[:2] == (1, 1)
    assert tokens.dtype == torch.long
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["mode"] == "discrete"
    assert metadata["sample_rate"] == 8_000
    assert metadata["token_schema"] == "audio_tokens_frames_v1"
    assert sample.task_target.target_audio_tokens_path is None


def test_audio_generation_target_rejects_source_outside_project(
    tmp_path: Path,
) -> None:
    external = tmp_path.parent / "external-target.wav"
    _write_pcm_wav(external)
    sample = TrainingSample(
        sample_id="sample-outside:speech_to_audio",
        snapshot_id="snapshot-a",
        task_target=TrainingTaskTarget(
            task_type="speech_to_audio",
            target_audio_path="target.wav",
            output_modalities=("audio",),
        ),
    )
    # Replace the in-project lexical path with a symlink to an external file.
    try:
        (tmp_path / "target.wav").symlink_to(external)
    except OSError as exc:
        if exc.winerror == 1314:
            pytest.skip(f"file symlinks are unavailable: {exc}")
        raise
    materializer = AudioGenerationTargetMaterializer(
        tokenizer=AudioTokenizer(sample_rate=8_000),
        output_root=tmp_path / "staging" / "audio_tokens",
    )

    try:
        materializer.materialize(sample, project_root=tmp_path)
    except RuntimeError as error:
        assert "outside the project" in str(error)
    else:
        raise AssertionError("external audio target was not rejected")
