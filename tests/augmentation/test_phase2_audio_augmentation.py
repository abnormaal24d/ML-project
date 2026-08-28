from __future__ import annotations

import math
import struct
import wave
from functools import partial
from pathlib import Path

import pytest

from augmentation.annotations.annotation_safety import (
    non_transformable_annotations,
)
from augmentation.audio.audio_augmenter import AudioAugmenter
from augmentation.audio.audio_stream_transformer import (
    AudioTransformParameters,
    prepare_audio_transform,
)
from augmentation.audio.audio_validation import (
    validate_audio_input,
    validate_audio_output,
)
from augmentation.generated_artifact_cache import AugmentationCache
from config.augmentation.audio_settings import AudioAugmentationSettings
from mmcrawler_datasets.schema import (
    ModalityObject,
    MultimodalSample,
    ProsodyFeatures,
    SpeakerSegment,
)
from preprocessing.media.adapters.audio_decode import WaveAudioDecodeBackend
from preprocessing.media.ports import DecodedAudio


class _Logger:
    def debug(self, *args: object, **kwargs: object) -> None:
        pass

    def warning(self, *args: object, **kwargs: object) -> None:
        pass


def _write_stereo_fixture(path: Path) -> None:
    sample_rate = 8_000
    frames: list[tuple[int, int]] = []
    for index in range(sample_rate * 2):
        if 2_000 <= index < 14_000:
            frames.append((12_000, 4_000))
        else:
            frames.append((0, 0))
    payload = b"".join(
        struct.pack("<hh", left, right) for left, right in frames
    )
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(payload)


def _augmenter(
    settings: AudioAugmentationSettings, tmp_path: Path
) -> AudioAugmenter:
    return AudioAugmenter(
        settings=settings,
        decoder=WaveAudioDecodeBackend(),
        cache=AugmentationCache(
            enabled=False,
            cache_directory=tmp_path / "cache",
            logger=_Logger(),
        ),
        validate_input=partial(
            validate_audio_input,
            allowed_mime_types=frozenset({"audio/wav", "audio/x-wav"}),
            max_input_bytes=10_000_000,
        ),
        validate_output=validate_audio_output,
        max_duration_seconds=60.0,
        logger=_Logger(),
    )


def test_audio_augmentation_resamples_converts_trims_speeds_and_updates_timestamps(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    _write_stereo_fixture(source)
    settings = AudioAugmentationSettings(
        enabled=True,
        operations=(
            "trim",
            "channel_conversion",
            "sample_rate_normalization",
            "speed_perturbation",
        ),
        target_sample_rate=16_000,
        target_channels=1,
        trim_silence_threshold_dbfs=-50.0,
        trim_padding_seconds=0.0,
        speed_factor=1.25,
        output_directory="objects/audio/augmented",
    )
    sample = MultimodalSample(
        sample_id="audio-source",
        audio=ModalityObject(
            path=Path("source.wav"),
            mime_type="audio/wav",
            byte_size=source.stat().st_size,
            metadata={"audio_duration_seconds": 2.0},
        ),
        speaker_segments=(
            SpeakerSegment(
                start_seconds=0.10,
                end_seconds=1.90,
                speaker_id="speaker-a",
            ),
            SpeakerSegment(
                start_seconds=0.0,
                end_seconds=0.10,
                speaker_id="outside",
            ),
        ),
        task_target={
            "word_timestamps": [
                {"word": "hello", "start_ms": 1000.0, "end_ms": 1100.0}
            ]
        },
        metadata={
            "modality": "audio",
            "audio_duration_seconds": 2.0,
            "transcript_segments": [
                {
                    "text": "kept",
                    "source": "asr",
                    "start_seconds": 0.5,
                    "end_seconds": 1.5,
                    "confidence": 0.9,
                },
                {
                    "text": "removed",
                    "source": "asr",
                    "start_seconds": 0.0,
                    "end_seconds": 0.1,
                    "confidence": 0.9,
                },
            ],
            "speaker_segments": [
                {
                    "speaker_id": "speaker-a",
                    "start_seconds": 0.1,
                    "end_seconds": 1.9,
                }
            ],
        },
    )

    produced, rejected = _augmenter(settings, tmp_path).augment(
        sample=sample,
        dataset_root=tmp_path,
    )

    assert rejected == ()
    assert len(produced) == 1
    variant = produced[0][1]
    assert variant.audio is not None and variant.audio.path is not None
    with wave.open(str(variant.audio.path), "rb") as reader:
        assert reader.getframerate() == 16_000
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getnframes() == 19_200
        assert reader.getnframes() / reader.getframerate() == pytest.approx(
            1.2
        )
        first_sample = struct.unpack("<h", reader.readframes(1))[0]
        assert first_sample == pytest.approx(8_000, abs=10)

    transform = variant.metadata["augmentation_audio_transform"]
    assert isinstance(transform, dict)
    assert transform["trim_start_seconds"] == pytest.approx(0.25)
    assert transform["trim_end_seconds"] == pytest.approx(1.75)
    assert transform["speed_factor"] == pytest.approx(1.25)
    assert transform["output_sample_rate"] == 16_000
    assert transform["output_channels"] == 1
    assert transform["output_duration_seconds"] == pytest.approx(1.2)

    assert len(variant.speaker_segments) == 1
    assert variant.speaker_segments[0].speaker_id == "speaker-a"
    assert variant.speaker_segments[0].start_seconds == pytest.approx(0.0)
    assert variant.speaker_segments[0].end_seconds == pytest.approx(1.2)

    segments = variant.metadata["transcript_segments"]
    assert isinstance(segments, list) and len(segments) == 1
    assert segments[0]["text"] == "kept"
    assert segments[0]["start_seconds"] == pytest.approx(0.2)
    assert segments[0]["end_seconds"] == pytest.approx(1.0)

    metadata_speakers = variant.metadata["speaker_segments"]
    assert isinstance(metadata_speakers, list) and len(metadata_speakers) == 1
    assert metadata_speakers[0]["start_seconds"] == pytest.approx(0.0)
    assert metadata_speakers[0]["end_seconds"] == pytest.approx(1.2)

    words = variant.task_target["word_timestamps"]
    assert isinstance(words, list) and len(words) == 1
    assert words[0]["start_ms"] == pytest.approx(600.0)
    assert words[0]["end_ms"] == pytest.approx(680.0)
    assert variant.metadata["audio_duration_seconds"] == pytest.approx(1.2)
    assert variant.audio.metadata["sample_rate"] == 16_000
    assert variant.audio.metadata["channels"] == 1
    assert variant.audio.metadata["duration_seconds"] == pytest.approx(1.2)


def test_speed_perturbation_changes_pcm_frame_count_not_only_header() -> None:
    sample_rate = 8_000
    values = [
        int(10_000 * math.sin(2 * math.pi * 220 * i / sample_rate))
        for i in range(sample_rate)
    ]
    payload = struct.pack(f"<{len(values)}h", *values)
    decoded = DecodedAudio(
        channels=1,
        sample_width=2,
        sample_rate=sample_rate,
        duration_sec=1.0,
        frames_iterator=iter((payload,)),
    )
    prepared = prepare_audio_transform(
        decoded_audio=decoded,
        parameters=AudioTransformParameters(
            operations=("speed_perturbation",),
            target_sample_rate=sample_rate,
            target_channels=1,
            gain_db=0.0,
            noise_std_fraction=0.0,
            trim_silence_threshold_dbfs=-50.0,
            trim_padding_seconds=0.0,
            speed_factor=2.0,
            noise_seed=1,
        ),
    )
    assert prepared.receipt.output_sample_rate == sample_rate
    assert prepared.receipt.output_frame_count == 4_000
    assert len(prepared.pcm_bytes) == 8_000
    assert prepared.receipt.output_duration_seconds == pytest.approx(0.5)


def test_audio_output_validation_checks_rate_channels_duration_clipping_and_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clipped.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8_000)
        writer.writeframes(struct.pack("<800h", *([32767] * 800)))

    clipped = validate_audio_output(
        path=path,
        expected_sample_rate=8_000,
        expected_channels=1,
        expected_duration_seconds=0.1,
        output_max_bytes=1_000_000,
        max_clipping_fraction=0.0,
        duration_tolerance_seconds=0.001,
    )
    assert clipped.rejection_reason == "generated_audio_clipping_excessive"

    wrong_rate = validate_audio_output(
        path=path,
        expected_sample_rate=16_000,
        expected_channels=1,
        expected_duration_seconds=0.1,
        output_max_bytes=1_000_000,
        max_clipping_fraction=1.0,
        duration_tolerance_seconds=0.001,
    )
    assert (
        wrong_rate.rejection_reason == "generated_audio_sample_rate_mismatch"
    )

    too_large = validate_audio_output(
        path=path,
        expected_sample_rate=8_000,
        expected_channels=1,
        expected_duration_seconds=0.1,
        output_max_bytes=100,
        max_clipping_fraction=1.0,
        duration_tolerance_seconds=0.001,
    )
    assert too_large.rejection_reason == "generated_audio_size_invalid"


def test_audio_timed_annotations_are_allowed_but_unknown_timestamps_and_prosody_block() -> (
    None
):
    allowed = MultimodalSample(
        sample_id="allowed",
        speaker_segments=(SpeakerSegment(0.0, 1.0, "speaker"),),
        metadata={
            "transcript_segments": [
                {"text": "hello", "start_seconds": 0.0, "end_seconds": 1.0}
            ],
            "word_timestamps": [
                {"word": "hello", "start_seconds": 0.0, "end_seconds": 0.5}
            ],
        },
    )
    assert (
        non_transformable_annotations(sample=allowed, media_kind="audio") == ()
    )

    unknown = MultimodalSample(
        sample_id="unknown",
        metadata={"event_timestamps": [0.1, 0.2]},
    )
    assert "metadata.event_timestamps" in non_transformable_annotations(
        sample=unknown,
        media_kind="audio",
    )

    with_prosody = MultimodalSample(
        sample_id="prosody",
        prosody=ProsodyFeatures(energy=0.5),
    )
    assert "prosody" in non_transformable_annotations(
        sample=with_prosody,
        media_kind="audio",
    )


def test_malformed_transcript_timing_rejects_audio_variant(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    _write_stereo_fixture(source)
    settings = AudioAugmentationSettings(
        enabled=True,
        operations=("trim",),
        trim_padding_seconds=0.0,
    )
    sample = MultimodalSample(
        sample_id="bad-timing",
        audio=ModalityObject(
            path=source,
            mime_type="audio/wav",
            byte_size=source.stat().st_size,
        ),
        metadata={"transcript_segments": [{"text": "missing timing"}]},
    )

    produced, rejected = _augmenter(settings, tmp_path).augment(
        sample=sample,
        dataset_root=tmp_path,
    )
    assert produced == ()
    assert rejected[0].reason == "audio_transform_failed"
    assert "missing_time_bounds" in (rejected[0].message or "")


def test_channel_conversion_supports_mono_to_stereo() -> None:
    source_values = (1000, -1000, 2000, -2000)
    decoded = DecodedAudio(
        channels=1,
        sample_width=2,
        sample_rate=8_000,
        duration_sec=len(source_values) / 8_000,
        frames_iterator=iter((struct.pack("<4h", *source_values),)),
    )
    prepared = prepare_audio_transform(
        decoded_audio=decoded,
        parameters=AudioTransformParameters(
            operations=("channel_conversion",),
            target_sample_rate=8_000,
            target_channels=2,
            gain_db=0.0,
            noise_std_fraction=0.0,
            trim_silence_threshold_dbfs=-50.0,
            trim_padding_seconds=0.0,
            speed_factor=1.0,
            noise_seed=1,
        ),
    )
    assert prepared.receipt.output_channels == 2
    assert struct.unpack("<8h", prepared.pcm_bytes) == (
        1000,
        1000,
        -1000,
        -1000,
        2000,
        2000,
        -2000,
        -2000,
    )


def test_default_audio_transform_is_byte_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_stereo_fixture(source)
    settings = AudioAugmentationSettings(
        enabled=True,
        output_directory="objects/audio/augmented",
    )
    sample = MultimodalSample(
        sample_id="deterministic-audio",
        audio=ModalityObject(
            path=source,
            mime_type="audio/wav",
            byte_size=source.stat().st_size,
        ),
    )
    augmenter = _augmenter(settings, tmp_path)
    first, first_rejections = augmenter.augment(
        sample=sample,
        dataset_root=tmp_path,
    )
    first_bytes = first[0][1].audio.path.read_bytes()  # type: ignore[union-attr]
    second, second_rejections = augmenter.augment(
        sample=sample,
        dataset_root=tmp_path,
    )
    second_bytes = second[0][1].audio.path.read_bytes()  # type: ignore[union-attr]

    assert first_rejections == second_rejections == ()
    assert first[0][1].sample_id == second[0][1].sample_id
    assert first_bytes == second_bytes
