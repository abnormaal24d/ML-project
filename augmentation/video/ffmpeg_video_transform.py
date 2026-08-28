"""FFmpeg-backed video augmentation adapter using internally built argv."""

from __future__ import annotations

import json
import math
import subprocess  # nosec: B404
from fractions import Fraction
from pathlib import Path

from augmentation.video.video_transform_backend import (
    AudioPolicy,
    ResizeMode,
    VideoProbe,
)
from config.augmentation.video_settings import VideoAugmentationSettings
from config.media_toolchain import MediaToolchainSettings
from preprocessing.media.adapters.versioned_executable import (
    ExecutableVerificationError,
    resolve_and_verify_executable,
)


class FfmpegVideoTransformBackend:
    """Execute timestamp-correct transforms and strict decode validation."""

    def __init__(
        self,
        *,
        toolchain: MediaToolchainSettings,
        settings: VideoAugmentationSettings,
    ) -> None:
        try:
            self._ffmpeg, self._ffmpeg_version = resolve_and_verify_executable(
                tool_name="ffmpeg",
                configured_executable=toolchain.ffmpeg_executable,
                expected_version=toolchain.ffmpeg_expected_version,
                timeout_seconds=settings.probe_timeout_seconds,
                required=settings.enabled,
            )
        except ExecutableVerificationError as error:
            raise RuntimeError(str(error)) from error
        try:
            self._ffprobe, self._ffprobe_version = (
                resolve_and_verify_executable(
                    tool_name="ffprobe",
                    configured_executable=toolchain.ffprobe_executable,
                    expected_version=toolchain.ffprobe_expected_version,
                    timeout_seconds=settings.probe_timeout_seconds,
                    required=settings.enabled,
                )
            )
        except ExecutableVerificationError as error:
            raise RuntimeError(str(error)) from error
        self._probe_timeout_seconds = settings.probe_timeout_seconds

    def is_available(self) -> bool:
        return (
            self._ffmpeg_version is not None
            and self._ffprobe_version is not None
        )

    def provenance(self) -> dict[str, object]:
        """Return the resolved, verified toolchain bound to generated media."""

        return {
            "ffmpeg_executable": self._ffmpeg,
            "ffmpeg_version": self._ffmpeg_version,
            "ffprobe_executable": self._ffprobe,
            "ffprobe_version": self._ffprobe_version,
            "probe_timeout_seconds": self._probe_timeout_seconds,
        }

    def probe(self, *, path: Path) -> VideoProbe:
        payload = self._probe_payload(
            path=path,
            timeout_seconds=self._probe_timeout_seconds,
        )
        streams = payload.get("streams")
        if not isinstance(streams, list):
            raise ValueError("video_probe_streams_missing")
        video_stream = next(
            (
                stream
                for stream in streams
                if isinstance(stream, dict)
                and stream.get("codec_type") == "video"
            ),
            None,
        )
        if not isinstance(video_stream, dict):
            raise ValueError("video_probe_video_stream_missing")
        audio_stream = next(
            (
                stream
                for stream in streams
                if isinstance(stream, dict)
                and stream.get("codec_type") == "audio"
            ),
            None,
        )
        width = _positive_int(video_stream.get("width"), field="width")
        height = _positive_int(video_stream.get("height"), field="height")
        fps = _positive_rate(
            video_stream.get("avg_frame_rate")
            or video_stream.get("r_frame_rate")
        )
        duration = _duration_seconds(
            payload=payload, video_stream=video_stream
        )
        frame_count = _optional_positive_int(video_stream.get("nb_frames"))
        if frame_count is None:
            frame_count = max(1, int(round(duration * fps)))
        audio_duration = (
            _optional_duration(audio_stream.get("duration"))
            if isinstance(audio_stream, dict)
            else None
        )
        return VideoProbe(
            width=width,
            height=height,
            fps=fps,
            duration_seconds=duration,
            frame_count=frame_count,
            has_audio=isinstance(audio_stream, dict),
            video_codec=_optional_string(video_stream.get("codec_name")),
            audio_codec=(
                _optional_string(audio_stream.get("codec_name"))
                if isinstance(audio_stream, dict)
                else None
            ),
            audio_sample_rate=(
                _optional_positive_int(audio_stream.get("sample_rate"))
                if isinstance(audio_stream, dict)
                else None
            ),
            audio_channels=(
                _optional_positive_int(audio_stream.get("channels"))
                if isinstance(audio_stream, dict)
                else None
            ),
            audio_duration_seconds=audio_duration,
        )

    def render_clip(
        self,
        *,
        source_path: Path,
        output_path: Path,
        crop_start_seconds: float,
        crop_duration_seconds: float,
        output_fps: float,
        output_width: int,
        output_height: int,
        resize_mode: ResizeMode,
        audio_policy: AudioPolicy,
        timeout_seconds: float,
    ) -> None:
        if crop_start_seconds < 0.0 or crop_duration_seconds <= 0.0:
            raise ValueError("invalid_video_crop_window")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(
            f".{output_path.stem}.tmp{output_path.suffix}"
        )
        tmp_path.unlink(missing_ok=True)
        video_filter = _video_filter(
            crop_start_seconds=crop_start_seconds,
            crop_duration_seconds=crop_duration_seconds,
            output_fps=output_fps,
            output_width=output_width,
            output_height=output_height,
            resize_mode=resize_mode,
        )
        command = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
        ]
        source_has_audio = self.probe(path=source_path).has_audio
        if audio_policy == "preserve" and source_has_audio:
            filter_complex = (
                f"[0:v:0]{video_filter}[v];"
                f"[0:a:0]atrim=start={crop_start_seconds:.9f}:"
                f"duration={crop_duration_seconds:.9f},"
                f"asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0,"
                f"apad=whole_dur={crop_duration_seconds:.9f},"
                f"atrim=duration={crop_duration_seconds:.9f}[a]"
            )
            command.extend(
                [
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[v]",
                    "-map",
                    "[a]",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "160k",
                ]
            )
        elif audio_policy in {"preserve", "remove"}:
            command.extend(["-vf", video_filter, "-an"])
        else:
            raise ValueError(f"unsupported_video_audio_policy:{audio_policy}")
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "cfr",
                "-movflags",
                "+faststart",
                str(tmp_path),
            ]
        )
        try:
            _run(command=command, timeout_seconds=timeout_seconds)
            if not tmp_path.is_file() or tmp_path.stat().st_size <= 0:
                raise RuntimeError("video_clip_write_failed")
            tmp_path.replace(output_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def extract_keyframe(
        self,
        *,
        source_path: Path,
        output_path: Path,
        timestamp_seconds: float,
        output_width: int,
        output_height: int,
        resize_mode: ResizeMode,
        timeout_seconds: float,
    ) -> None:
        if timestamp_seconds < 0.0:
            raise ValueError("invalid_keyframe_timestamp")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(
            f".{output_path.stem}.tmp{output_path.suffix}"
        )
        tmp_path.unlink(missing_ok=True)
        command = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-ss",
            f"{timestamp_seconds:.9f}",
            "-frames:v",
            "1",
            "-vf",
            _spatial_filter(
                output_width=output_width,
                output_height=output_height,
                resize_mode=resize_mode,
            ),
            "-q:v",
            "2",
            str(tmp_path),
        ]
        try:
            _run(command=command, timeout_seconds=timeout_seconds)
            if not tmp_path.is_file() or tmp_path.stat().st_size <= 0:
                raise RuntimeError("video_keyframe_write_failed")
            tmp_path.replace(output_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def decode_check(self, *, path: Path, timeout_seconds: float) -> None:
        _run(
            command=[
                self._ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ],
            timeout_seconds=timeout_seconds,
        )

    def validate_keyframe(
        self,
        *,
        path: Path,
        expected_width: int,
        expected_height: int,
        output_max_bytes: int,
        timeout_seconds: float,
    ) -> dict[str, object]:
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError("generated_keyframe_missing_or_empty")
        if path.stat().st_size > output_max_bytes:
            raise ValueError("generated_keyframe_size_invalid")
        self.decode_check(path=path, timeout_seconds=timeout_seconds)
        payload = self._probe_payload(
            path=path, timeout_seconds=timeout_seconds
        )
        streams = payload.get("streams")
        if not isinstance(streams, list):
            raise ValueError("generated_keyframe_probe_failed")
        stream = next(
            (
                item
                for item in streams
                if isinstance(item, dict) and item.get("codec_type") == "video"
            ),
            None,
        )
        if not isinstance(stream, dict):
            raise ValueError("generated_keyframe_stream_missing")
        width = _positive_int(stream.get("width"), field="keyframe_width")
        height = _positive_int(stream.get("height"), field="keyframe_height")
        if width != expected_width or height != expected_height:
            raise ValueError("generated_keyframe_dimensions_mismatch")
        return {
            "byte_size": path.stat().st_size,
            "width": width,
            "height": height,
            "format": _optional_string(stream.get("codec_name")) or "unknown",
            "decoded": True,
        }

    def _probe_payload(
        self,
        *,
        path: Path,
        timeout_seconds: float,
    ) -> dict[str, object]:
        if not path.is_file():
            raise FileNotFoundError(path)
        completed = _run(
            command=[
                self._ffprobe,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            timeout_seconds=timeout_seconds,
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("video_probe_invalid_json") from exc
        if not isinstance(payload, dict):
            raise ValueError("video_probe_payload_must_be_object")
        return payload


def _video_filter(
    *,
    crop_start_seconds: float,
    crop_duration_seconds: float,
    output_fps: float,
    output_width: int,
    output_height: int,
    resize_mode: ResizeMode,
) -> str:
    return (
        f"trim=start={crop_start_seconds:.9f}:duration={crop_duration_seconds:.9f},"
        "setpts=PTS-STARTPTS,"
        f"fps=fps={output_fps:.9f}:round=near,"
        f"{_spatial_filter(output_width=output_width, output_height=output_height, resize_mode=resize_mode)}"
    )


def _spatial_filter(
    *,
    output_width: int,
    output_height: int,
    resize_mode: ResizeMode,
) -> str:
    if resize_mode == "letterbox":
        return (
            f"scale={output_width}:{output_height}:force_original_aspect_ratio=decrease:"
            "flags=lanczos,"
            f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            "setsar=1"
        )
    if resize_mode == "center_crop":
        return (
            f"scale={output_width}:{output_height}:force_original_aspect_ratio=increase:"
            "flags=lanczos,"
            f"crop={output_width}:{output_height},setsar=1"
        )
    raise ValueError(f"unsupported_video_resize_mode:{resize_mode}")


def _run(
    *,
    command: list[str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # nosec: B603
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("video_backend_timeout") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()[-2000:]
        raise RuntimeError(f"video_backend_failed:{detail}") from exc


def _positive_rate(value: object) -> float:
    if not isinstance(value, str) or value in {"0/0", "N/A", ""}:
        raise ValueError("video_probe_fps_missing")
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("video_probe_fps_invalid") from exc
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("video_probe_fps_invalid")
    return rate


def _duration_seconds(
    *,
    payload: dict[str, object],
    video_stream: dict[str, object],
) -> float:
    candidates = [video_stream.get("duration")]
    format_payload = payload.get("format")
    if isinstance(format_payload, dict):
        candidates.append(format_payload.get("duration"))
    for value in candidates:
        try:
            duration = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0.0:
            return duration
    raise ValueError("video_probe_duration_missing")


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(
        value, (str, bytes, bytearray, int)
    ):
        raise ValueError(f"video_probe_{field}_invalid")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"video_probe_{field}_invalid") from exc
    if number <= 0:
        raise ValueError(f"video_probe_{field}_invalid")
    return number


def _optional_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(
        value, (str, bytes, bytearray, int)
    ):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _optional_duration(value: object) -> float | None:
    try:
        duration = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration > 0.0 else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
