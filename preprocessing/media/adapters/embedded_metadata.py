"""Pinned local adapter for embedded metadata and stream inspection."""

from __future__ import annotations

import json
import subprocess  # nosec: B404
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from config.media_toolchain import MediaToolchainSettings
from config.preprocessing.media_settings import MediaPrivacySettings
from preprocessing.media.adapters.versioned_executable import (
    ExecutableVerificationError,
    resolve_and_verify_executable,
)


class FfmpegEmbeddedMetadataAdapter:
    """Inspect and sanitize metadata using resolved, version-pinned tools."""

    def __init__(
        self,
        *,
        toolchain: MediaToolchainSettings,
        settings: MediaPrivacySettings,
        required: bool,
    ) -> None:
        try:
            self._ffmpeg, self._ffmpeg_version = resolve_and_verify_executable(
                tool_name="ffmpeg",
                configured_executable=toolchain.ffmpeg_executable,
                expected_version=toolchain.ffmpeg_expected_version,
                timeout_seconds=settings.probe_timeout_seconds,
                required=required,
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
                    required=required,
                )
            )
        except ExecutableVerificationError as error:
            raise RuntimeError(str(error)) from error
        self._probe_timeout_seconds = settings.probe_timeout_seconds
        self._metadata_strip_timeout_seconds = (
            settings.metadata_strip_timeout_seconds
        )
        self._video_audio_probe_timeout_seconds = (
            settings.video_audio_probe_timeout_seconds
        )

    def inspect(
        self,
        *,
        path: Path,
        modality: str,
    ) -> tuple[str, ...] | None:
        return _inspect_embedded_metadata(
            path=path,
            modality=modality,
            ffprobe_executable=self._ffprobe,
            timeout_seconds=self._probe_timeout_seconds,
        )

    def remove(
        self,
        *,
        source: Path,
        destination: Path,
        modality: str,
    ) -> bool:
        return _remove_embedded_metadata(
            source=source,
            destination=destination,
            modality=modality,
            ffmpeg_executable=self._ffmpeg,
            timeout_seconds=self._metadata_strip_timeout_seconds,
        )

    def has_audio_stream(self, path: Path) -> bool | None:
        return _has_audio_stream(
            path=path,
            ffprobe_executable=self._ffprobe,
            timeout_seconds=self._video_audio_probe_timeout_seconds,
        )

    def provenance(self) -> dict[str, object]:
        return {
            "ffmpeg_executable": self._ffmpeg,
            "ffmpeg_version": self._ffmpeg_version,
            "ffprobe_executable": self._ffprobe,
            "ffprobe_version": self._ffprobe_version,
            "probe_timeout_seconds": self._probe_timeout_seconds,
            "metadata_strip_timeout_seconds": (
                self._metadata_strip_timeout_seconds
            ),
            "video_audio_probe_timeout_seconds": (
                self._video_audio_probe_timeout_seconds
            ),
        }


def _inspect_embedded_metadata(
    *,
    path: Path,
    modality: str,
    ffprobe_executable: str,
    timeout_seconds: float,
) -> tuple[str, ...] | None:
    candidate = path.expanduser().resolve()
    if not candidate.is_file():
        return None
    if modality == "image":
        try:
            with Image.open(candidate) as image:
                image_values: list[str] = []
                for value in image.getexif().values():
                    text = _metadata_text(value)
                    if text:
                        image_values.append(text)
                technical_keys = {
                    "background",
                    "dpi",
                    "duration",
                    "exif",
                    "gamma",
                    "jfif",
                    "jfif_density",
                    "jfif_unit",
                    "jfif_version",
                    "loop",
                    "srgb",
                    "transparency",
                }
                for key, value in image.info.items():
                    if str(key).casefold() in technical_keys:
                        continue
                    text = _metadata_text(value)
                    if text:
                        image_values.append(text)
                return tuple(image_values)
        except (Image.DecompressionBombError, OSError, ValueError):
            return None
    if modality in {"audio", "video"}:
        try:
            # The executable and flags are fixed; the media argument is absolute.
            completed = subprocess.run(  # nosec: B607
                [
                    ffprobe_executable,
                    "-v",
                    "error",
                    "-show_entries",
                    "format_tags:stream_tags",
                    "-of",
                    "json",
                    str(candidate),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False,  # nosec: B603
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            return None
        media_values: list[str] = []
        technical_tags = {
            "compatible_brands",
            "encoder",
            "handler_name",
            "language",
            "major_brand",
            "minor_version",
            "vendor_id",
        }
        format_tags = payload.get("format", {}).get("tags", {})
        if isinstance(format_tags, Mapping):
            media_values.extend(
                text
                for key, raw in format_tags.items()
                if str(key).casefold() not in technical_tags
                and (text := _metadata_text(raw))
            )
        streams = payload.get("streams", ())
        if isinstance(streams, list):
            for stream in streams:
                tags = (
                    stream.get("tags", {})
                    if isinstance(stream, Mapping)
                    else {}
                )
                if isinstance(tags, Mapping):
                    media_values.extend(
                        text
                        for key, raw in tags.items()
                        if str(key).casefold() not in technical_tags
                        and (text := _metadata_text(raw))
                    )
        return tuple(media_values)
    return None


def _remove_embedded_metadata(
    *,
    source: Path,
    destination: Path,
    modality: str,
    ffmpeg_executable: str,
    timeout_seconds: float,
) -> bool:
    try:
        source_path = source.expanduser().resolve()
        destination_path = destination.expanduser().resolve()
        if source_path == destination_path:
            return False
        if modality == "image":
            with Image.open(source_path) as image:
                if getattr(image, "n_frames", 1) != 1:
                    return False
                clean = image.copy()
                clean.info.clear()
                clean.save(destination_path, format=image.format)
            return True
        if modality in {"audio", "video"}:
            stream_mapping = (
                ["-map", "0:a:0"]
                if modality == "audio"
                else ["-map", "0:V:0", "-map", "0:a:0?"]
            )
            # The executable and flags are fixed; both media arguments are absolute.
            completed = subprocess.run(  # nosec: B607
                [
                    ffmpeg_executable,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(source_path),
                    *stream_mapping,
                    "-map_metadata",
                    "-1",
                    "-map_chapters",
                    "-1",
                    "-c",
                    "copy",
                    str(destination_path),
                ],
                check=False,
                capture_output=True,
                timeout=timeout_seconds,
                shell=False,  # nosec: B603
            )
            return completed.returncode == 0 and destination_path.is_file()
    except (
        Image.DecompressionBombError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ):
        return False
    return False


def _has_audio_stream(
    *,
    path: Path,
    ffprobe_executable: str,
    timeout_seconds: float,
) -> bool | None:
    candidate = path.expanduser().resolve()
    if not candidate.is_file():
        return None
    try:
        completed = subprocess.run(  # nosec: B603
            [
                ffprobe_executable,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(candidate),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return "audio" in completed.stdout.lower()


def _metadata_text(value: object) -> str:
    if isinstance(value, bytes):
        for encoding in ("utf-8", "latin-1"):
            try:
                return value.decode(encoding).strip()[:4096]
            except UnicodeDecodeError:
                continue
        return ""
    return str(value).strip()[:4096]


__all__ = [
    "FfmpegEmbeddedMetadataAdapter",
]
