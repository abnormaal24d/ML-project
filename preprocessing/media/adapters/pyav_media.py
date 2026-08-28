"""PyAV-backed container probing, audio extraction, and video normalization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from preprocessing.media.ports import (
    VideoAudioTrackResult,
    VideoNormalizationResult,
)

if TYPE_CHECKING:
    from preprocessing.media.adapters.opencv_video import OpenCvVideoClipWriter


def _load_av() -> Any | None:
    try:
        import av
    except ImportError:
        return None
    return av


class PyAvContainerProbe:
    """Probe container streams with PyAV."""

    def __init__(self, *, av_module: Any | None = None) -> None:
        self._av = av_module if av_module is not None else _load_av()

    def probe_video(self, *, path: str) -> dict[str, object]:
        if self._av is None:
            return {"probe_status": "failed", "error_type": "ImportError"}
        try:
            container = self._av.open(path)
        except (OSError, RuntimeError, ValueError) as exc:
            return {
                "probe_status": "failed",
                "error_type": type(exc).__name__,
            }
        try:
            metadata: dict[str, object] = {"probe_status": "passed"}
            if container.duration is not None:
                metadata["duration_seconds"] = float(
                    container.duration
                ) / float(
                    self._av.time_base.denominator
                    if hasattr(self._av, "time_base")
                    else 1_000_000
                )
                # PyAV duration is in AV_TIME_BASE (microseconds)
                metadata["duration_seconds"] = (
                    float(container.duration) / 1_000_000.0
                )
            for stream in container.streams:
                if stream.type == "video":
                    metadata["width"] = (
                        int(getattr(stream, "width", 0) or 0) or None
                    )
                    metadata["height"] = (
                        int(getattr(stream, "height", 0) or 0) or None
                    )
                    rate = getattr(stream, "average_rate", None) or getattr(
                        stream, "base_rate", None
                    )
                    if rate is not None:
                        try:
                            metadata["fps"] = float(rate)
                        except (TypeError, ValueError, ZeroDivisionError):
                            pass
                    if getattr(stream, "frames", None):
                        metadata["frame_count"] = int(stream.frames)
                    break
            return {
                key: value
                for key, value in metadata.items()
                if value is not None
            }
        finally:
            container.close()

    def audio_stream_status(self, path: Path) -> str:
        if self._av is None:
            return "unavailable"
        try:
            container = self._av.open(str(path))
        except (OSError, RuntimeError, ValueError):
            return "missing"
        try:
            for stream in container.streams:
                if stream.type == "audio":
                    return "present"
            return "missing"
        finally:
            container.close()


class PyAvAudioTrackExtractor:
    """Extract the first audio stream from a video into a WAV file."""

    def __init__(self, *, av_module: Any | None = None) -> None:
        self._av = av_module if av_module is not None else _load_av()

    def extract_to_wav(
        self,
        *,
        video_path: str | None = None,
        body: bytes | None = None,
        output_dir: str | None = None,
        target_sample_rate: int = 16000,
    ) -> VideoAudioTrackResult:
        del body  # body-based extraction is not required by current callers
        if self._av is None:
            return VideoAudioTrackResult(
                extracted=False,
                error_type="ImportError",
            )
        if not video_path:
            return VideoAudioTrackResult(
                extracted=False,
                error_type="missing_video_path",
            )
        out_dir = Path(output_dir) if output_dir else Path(video_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = (
            out_dir / f"{Path(video_path).stem}_audio_{target_sample_rate}.wav"
        )
        partial = audio_path
        try:
            container = self._av.open(video_path)
        except (OSError, RuntimeError, ValueError) as exc:
            return VideoAudioTrackResult(
                extracted=False,
                error_type=type(exc).__name__,
            )
        try:
            audio_stream = next(
                (
                    stream
                    for stream in container.streams
                    if stream.type == "audio"
                ),
                None,
            )
            if audio_stream is None:
                return VideoAudioTrackResult(
                    extracted=False,
                    error_type="missing_audio_stream",
                )
            resampler = self._av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono"
                if int(getattr(audio_stream, "channels", 1) or 1) <= 1
                else "stereo",
                rate=int(target_sample_rate),
            )
            import wave

            channels = (
                1 if str(getattr(resampler, "layout", "mono")) == "mono" else 2
            )
            with wave.open(str(partial), "wb") as writer:
                writer.setnchannels(channels)
                writer.setsampwidth(2)
                writer.setframerate(int(target_sample_rate))
                for frame in container.decode(audio_stream):
                    for resampled in resampler.resample(frame):
                        writer.writeframes(bytes(resampled.planes[0]))
            duration = None
            if partial.exists() and partial.stat().st_size > 44:
                duration = (partial.stat().st_size - 44) / (
                    2 * channels * float(target_sample_rate)
                )
            return VideoAudioTrackResult(
                extracted=True,
                audio_path=str(partial),
                sample_rate=int(target_sample_rate),
                channels=channels,
                duration_seconds=duration,
            )
        except (OSError, RuntimeError, ValueError, AttributeError) as exc:
            if partial.exists():
                partial.unlink(missing_ok=True)
            return VideoAudioTrackResult(
                extracted=False,
                error_type=type(exc).__name__,
            )
        finally:
            container.close()


class CompositeVideoNormalizerBackend:
    """Normalize video containers to a stable training representation."""

    def __init__(
        self,
        *,
        clip_writer: OpenCvVideoClipWriter | None = None,
    ) -> None:
        # Lazy import to avoid hard OpenCV import at module load for pure probe use.
        if clip_writer is None:
            from preprocessing.media.adapters.opencv_video import (
                OpenCvVideoClipWriter,
            )

            clip_writer = OpenCvVideoClipWriter()
        self._clip_writer = clip_writer

    def normalize(
        self,
        *,
        input_path: str,
        target_fps: float,
    ) -> VideoNormalizationResult:
        source = Path(input_path)
        if not source.is_file():
            return VideoNormalizationResult(
                normalized_path=None,
                status="failed",
                error_type="file_not_found",
            )
        if not self._clip_writer.is_available():
            return VideoNormalizationResult(
                normalized_path=None,
                status="failed",
                error_type="opencv_unavailable",
            )
        output = source.with_name(f"{source.stem}.normalized.mp4")
        try:
            self._clip_writer.write_resized_clip(
                source_path=source,
                output_path=output,
                max_duration_seconds=3600.0,
                output_fps=max(1, int(round(target_fps or 25.0))),
                output_size=(640, 360),
            )
            return VideoNormalizationResult(
                normalized_path=str(output),
                status="passed",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return VideoNormalizationResult(
                normalized_path=None,
                status="failed",
                error_type=type(exc).__name__,
            )
