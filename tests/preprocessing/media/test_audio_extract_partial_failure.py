"""Partial WAV extraction must not be reported as successful."""

from __future__ import annotations

from pathlib import Path

from preprocessing.media.adapters.pyav_media import (
    PyAvAudioTrackExtractor,
)
from preprocessing.media.transcript_segment_normalizer import (
    summarize_timeline,
)


def test_extract_failure_deletes_partial_wav_and_returns_not_extracted(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "clip_audio_16000.wav"
    partial.write_bytes(b"RIFF....partial")

    class _BoomStream:
        type = "audio"
        channels = 1

    class _BoomContainer:
        streams = [_BoomStream()]

        def decode(self, stream):  # noqa: ANN001, ARG002
            raise RuntimeError("decode failed mid-stream")

        def close(self) -> None:
            return None

    class _FakeAv:
        class audio:
            class resampler:
                class AudioResampler:
                    def __init__(self, **kwargs):  # noqa: ANN003
                        del kwargs

                    def resample(self, frame):  # noqa: ANN001
                        return [frame]

        @staticmethod
        def open(source):  # noqa: ANN001, ARG004
            return _BoomContainer()

    extractor = PyAvAudioTrackExtractor(av_module=_FakeAv())
    result = extractor.extract_to_wav(
        video_path=str(tmp_path / "clip.mp4"),
        output_dir=str(tmp_path),
        target_sample_rate=16000,
    )

    assert result.extracted is False
    assert result.audio_path is None
    assert not partial.exists()


def test_extract_unavailable_backend_returns_not_extracted() -> None:
    extractor = PyAvAudioTrackExtractor(av_module=None)
    result = extractor.extract_to_wav(video_path="missing.mp4")
    assert result.extracted is False
    assert result.audio_path is None


def test_audio_timeline_summary_preserves_timing_quality_and_analyzer_evidence() -> (
    None
):
    summary = summarize_timeline(
        segments=(
            {
                "text": "hello world",
                "start_seconds": 1.0,
                "end_seconds": 3.0,
                "confidence": 0.8,
            },
            {
                "text": "second segment",
                "start_seconds": 4.0,
                "end_seconds": 5.0,
                "confidence": 1.0,
            },
        ),
        duration_seconds=10.0,
        payload={
            "word_timestamps": [{"word": "hello", "start": 1.0}],
            "speaker_segments": [{"speaker": "speaker-1"}],
            "language_segments": [{"language": "en"}],
            "speaker_count": 1,
            "snr_db": 24.5,
            "speech_ratio": 0.6,
        },
    )

    assert summary["segment_count"] == 2
    assert summary["timed_coverage_seconds"] == 3.0
    assert summary["timed_coverage_ratio"] == 0.3
    assert summary["mean_segment_confidence"] == 0.9
    assert summary["word_timestamps_available"] is True
    assert summary["diarization_available"] is True
    assert summary["language_segments_available"] is True
    assert summary["snr_db"] == 24.5
