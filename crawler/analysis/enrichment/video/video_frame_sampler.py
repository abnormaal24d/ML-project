"""Video frame sampling and keyframe normalization."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from preprocessing.media.ports import (
    FrameProcessor,
    VideoCaptureSession,
    VideoReader,
)
from shared.runtime_primitives import IdGenerator

_LOGGER = logging.getLogger(__name__)

_SEQUENTIAL_STEP_SECONDS = 0.25
_FRAME_IMAGE_EXTENSION = ".jpg"


class VideoFrameSampler:
    """Sample representative frames from a video file."""

    def __init__(
        self,
        *,
        output_directory: Path | str = "runtime/tmp/video_keyframes",
        id_generator: IdGenerator,
        video_reader: VideoReader,
        frame_processor: FrameProcessor,
    ) -> None:
        self._output_directory = Path(output_directory)
        self._id_generator = id_generator
        self._video_reader = video_reader
        self._frame_processor = frame_processor

    def sample(
        self,
        *,
        path: Path,
        max_frames: int = 8,
    ) -> list[dict[str, Any]]:
        reader = self._video_reader
        if not reader.is_available():
            return []

        if max_frames <= 0:
            return []

        capture = reader.open(path)
        if capture is None:
            return []

        self._output_directory.mkdir(parents=True, exist_ok=True)

        try:
            fps = float(capture.get_fps() or 0.0)
            frame_count = int(capture.get_frame_count() or 0)

            if fps <= 0.0:
                fps = 25.0

            if frame_count <= 0:
                return self._sample_sequentially(
                    capture=capture,
                    fps=fps,
                    max_frames=max_frames,
                    source_path=path,
                )

            sampled = self._sample_by_frame_positions(
                capture=capture,
                fps=fps,
                frame_count=frame_count,
                max_frames=max_frames,
                source_path=path,
            )

            if sampled:
                return sampled

            sampled = self._sample_by_timestamps(
                capture=capture,
                fps=fps,
                frame_count=frame_count,
                max_frames=max_frames,
                source_path=path,
            )

            if sampled:
                return sampled

            return self._sample_sequentially(
                capture=capture,
                fps=fps,
                max_frames=max_frames,
                source_path=path,
            )
        finally:
            capture.close()

    def _sample_by_frame_positions(
        self,
        *,
        capture: VideoCaptureSession,
        fps: float,
        frame_count: int,
        max_frames: int,
        source_path: Path,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []

        for frame_index in self._representative_frame_indexes(
            frame_count=frame_count,
            max_frames=max_frames,
        ):
            capture.set_frame_index(frame_index)
            ok, frame = capture.read_frame()

            if not ok or frame is None:
                continue

            frame_path = self._write_sampled_frame(
                frame=frame,
                frame_index=frame_index,
                source_path=source_path,
            )

            if frame_path is None:
                continue

            output.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 4),
                    "frame": frame,
                    "frame_path": frame_path,
                }
            )

        return output

    def _sample_by_timestamps(
        self,
        *,
        capture: VideoCaptureSession,
        fps: float,
        frame_count: int,
        max_frames: int,
        source_path: Path,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        duration_seconds = frame_count / fps

        for timestamp_seconds in self._representative_timestamps(
            duration_seconds=duration_seconds,
            max_frames=max_frames,
        ):
            capture.set_timestamp_ms(timestamp_seconds * 1000.0)
            ok, frame = capture.read_frame()

            if not ok or frame is None:
                continue

            frame_index = int(round(timestamp_seconds * fps))

            frame_path = self._write_sampled_frame(
                frame=frame,
                frame_index=frame_index,
                source_path=source_path,
            )

            if frame_path is None:
                continue

            output.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(timestamp_seconds, 4),
                    "frame": frame,
                    "frame_path": frame_path,
                }
            )

        return output

    def _sample_sequentially(
        self,
        *,
        capture: VideoCaptureSession,
        fps: float,
        max_frames: int,
        source_path: Path,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        step_frames = max(
            1,
            int(round(_SEQUENTIAL_STEP_SECONDS * fps)),
        )
        frame_index = 0

        capture.set_frame_index(0)

        while len(output) < max_frames:
            ok, frame = capture.read_frame()

            if not ok or frame is None:
                break

            if frame_index % step_frames == 0:
                frame_path = self._write_sampled_frame(
                    frame=frame,
                    frame_index=frame_index,
                    source_path=source_path,
                )

                if frame_path is not None:
                    output.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_seconds": round(frame_index / fps, 4),
                            "frame": frame,
                            "frame_path": frame_path,
                        }
                    )

            frame_index += 1

        return output

    @staticmethod
    def _representative_frame_indexes(
        *,
        frame_count: int,
        max_frames: int,
    ) -> tuple[int, ...]:
        if max_frames <= 0:
            return ()

        if frame_count <= 1:
            return (0,)

        count = min(max_frames, frame_count)
        last_index = max(0, frame_count - 1)

        if count == 1:
            return (0,)

        indexes = {
            int(round((last_index * position) / (count - 1)))
            for position in range(count)
        }

        return tuple(sorted(indexes))

    @staticmethod
    def _representative_timestamps(
        *,
        duration_seconds: float,
        max_frames: int,
    ) -> tuple[float, ...]:
        if max_frames <= 0 or duration_seconds <= 0.0:
            return ()

        count = max(1, max_frames)

        if count == 1:
            return (0.0,)

        return tuple(
            max(0.0, (duration_seconds * position) / (count - 1))
            for position in range(count)
        )

    def _write_sampled_frame(
        self,
        *,
        frame: Any,
        frame_index: int,
        source_path: Path,
    ) -> str | None:
        try:
            import numpy as np
        except ImportError:
            return None

        processor = self._frame_processor
        if not processor.is_available():
            return None

        if frame is None:
            return None

        tmp_path: Path | None = None

        try:
            frame_array = np.asarray(frame)

            if frame_array.size == 0:
                return None

            if frame_array.dtype != np.uint8:
                frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)

            frame_array = np.ascontiguousarray(frame_array)

            frame_token = _stable_frame_token(
                path=source_path,
                frame_index=frame_index,
            )
            path = self._output_directory / (
                f"frame_{frame_token}{_FRAME_IMAGE_EXTENSION}"
            )

            self._output_directory.mkdir(parents=True, exist_ok=True)

            tmp_path = path.with_name(
                f"{path.stem}.{self._id_generator.generate()}.tmp{path.suffix}"
            )

            if not processor.write_image(tmp_path, frame_array):
                tmp_path.unlink(missing_ok=True)
                return None

            tmp_path.replace(path)
            return str(path)

        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    _LOGGER.warning(
                        "video_frame_temp_cleanup_failed",
                        extra={
                            "path": str(tmp_path),
                            "error_type": type(cleanup_error).__name__,
                        },
                    )
            return None


def _stable_frame_token(*, path: Path, frame_index: int) -> str:
    parts: tuple[str, ...]

    try:
        resolved = path.resolve()
        stat = resolved.stat()
        parts = (
            str(resolved),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(frame_index),
        )
    except OSError:
        parts = (str(path), str(frame_index))

    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
