"""Materialize speech-to-audio targets as durable token tensors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as functional

from mmcrawler_datasets.training_samples.models import TrainingSample
from multimodal.tokenization.audio import AudioTokenizer
from preprocessing.media.adapters.audio_decode import (
    CompositeAudioDecodeBackend,
    SoundFileAudioDecodeBackend,
    WaveAudioDecodeBackend,
)


class AudioGenerationTargetMaterializer:
    """Convert a target audio artifact into model-ready audio tokens."""

    def __init__(
        self,
        *,
        tokenizer: AudioTokenizer,
        output_root: Path,
        decoder: CompositeAudioDecodeBackend | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._output_root = output_root
        self._decoder = decoder or CompositeAudioDecodeBackend(
            wave_backend=WaveAudioDecodeBackend(),
            soundfile_backend_factory=SoundFileAudioDecodeBackend,
        )

    def materialize(
        self,
        sample: TrainingSample,
        *,
        project_root: Path,
    ) -> TrainingSample:
        if sample.task_target.task_type != "speech_to_audio":
            return sample
        raw_target = sample.task_target.target_audio_path
        if not raw_target:
            raise RuntimeError(
                f"audio generation sample {sample.sample_id} has no target"
            )

        root = project_root.resolve(strict=True)
        try:
            audio_path = (root / Path(raw_target)).resolve(strict=True)
            audio_path.relative_to(root)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"target audio is outside the project for {sample.sample_id}"
            ) from exc
        if not audio_path.is_file():
            raise RuntimeError(
                f"target audio is not a file for {sample.sample_id}"
            )

        waveform, source_rate = self._load_waveform(audio_path)
        if source_rate != self._tokenizer.sample_rate:
            target_length = max(
                1,
                round(
                    waveform.shape[-1]
                    * self._tokenizer.sample_rate
                    / source_rate
                ),
            )
            waveform = functional.interpolate(
                waveform.unsqueeze(0),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
        token_batch = self._tokenizer.encode(waveform.unsqueeze(0))

        output_root = self._output_root.resolve(strict=False)
        try:
            output_root.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                "audio token output escapes project root"
            ) from exc
        # A 128-bit digest prefix avoids Windows path-length failures while
        # retaining collision resistance for deterministic materialization.
        sample_key = hashlib.sha256(
            str(sample.sample_id).encode("utf-8")
        ).hexdigest()[:32]
        sample_dir = output_root / sample_key
        sample_dir.mkdir(parents=True, exist_ok=True)
        tokens_path = sample_dir / "target_audio_tokens.pt"
        metadata_path = sample_dir / "target_audio_tokens.json"
        tokens_tmp = tokens_path.with_suffix(tokens_path.suffix + ".tmp")
        metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        try:
            torch.save(token_batch.tokens.cpu().contiguous(), tokens_tmp)
            metadata_tmp.write_text(
                json.dumps(
                    {
                        # Schema identifier, not credential material.
                        "token_schema": "audio_tokens_frames_v1",  # nosec B105
                        "mode": self._tokenizer.mode,
                        "sample_rate": token_batch.sample_rate,
                        "frame_ms": token_batch.frame_ms,
                        "shape": list(token_batch.tokens.shape),
                        "dtype": str(token_batch.tokens.dtype),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            tokens_tmp.replace(tokens_path)
            metadata_tmp.replace(metadata_path)
        except Exception:
            tokens_tmp.unlink(missing_ok=True)
            metadata_tmp.unlink(missing_ok=True)
            tokens_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise

        return replace(
            sample,
            task_target=replace(
                sample.task_target,
                target_audio_tokens_path=tokens_path.relative_to(
                    root
                ).as_posix(),
            ),
        )

    def _load_waveform(self, path: Path) -> tuple[torch.Tensor, int]:
        decoded = self._decoder.decode(path=path, chunk_frames=65536)
        payload = b"".join(decoded.frames_iterator)
        if not payload:
            raise RuntimeError(f"target audio is empty: {path}")
        dtype = {
            1: torch.uint8,
            2: torch.int16,
            4: torch.int32,
        }.get(decoded.sample_width)
        if dtype is None:
            raise RuntimeError(
                f"unsupported PCM sample width: {decoded.sample_width}"
            )
        values = torch.frombuffer(bytearray(payload), dtype=dtype)
        if decoded.channels <= 0 or values.numel() % decoded.channels:
            raise RuntimeError(f"invalid decoded audio shape: {path}")
        values = values.reshape(-1, decoded.channels).transpose(0, 1)
        if dtype is torch.uint8:
            waveform = (values.to(torch.float32) - 128.0) / 128.0
        elif dtype is torch.int16:
            waveform = values.to(torch.float32) / 32768.0
        else:
            waveform = values.to(torch.float32) / 2147483648.0
        return waveform.clamp(-1.0, 1.0), decoded.sample_rate


__all__ = ["AudioGenerationTargetMaterializer"]
