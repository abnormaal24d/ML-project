"""OpenCV-backed video capture, frame processing, and clip writing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _load_cv2() -> Any | None:
    try:
        import cv2
    except ImportError:
        return None
    return cv2


class OpenCvVideoCaptureSession:
    """Thin session wrapper around ``cv2.VideoCapture``."""

    def __init__(self, *, capture: Any, cv2: Any) -> None:
        self._capture = capture
        self._cv2 = cv2

    def get_fps(self) -> float | None:
        value = float(self._capture.get(self._cv2.CAP_PROP_FPS) or 0.0)
        return value if value > 0.0 else None

    def get_frame_count(self) -> int | None:
        value = int(self._capture.get(self._cv2.CAP_PROP_FRAME_COUNT) or 0)
        return value if value > 0 else None

    def get_frame_size(self) -> tuple[int | None, int | None]:
        """Return positive frame dimensions when the backend reports them."""

        width = int(self._capture.get(self._cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return (
            width if width > 0 else None,
            height if height > 0 else None,
        )

    def set_frame_index(self, frame_index: int) -> None:
        self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, float(frame_index))

    def set_timestamp_ms(self, timestamp_ms: float) -> None:
        self._capture.set(self._cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))

    def read_frame(self) -> tuple[bool, Any]:
        ok, frame = self._capture.read()
        if not ok:
            return False, None
        return True, frame

    def close(self) -> None:
        self._capture.release()


class OpenCvVideoReader:
    """Open and probe video containers with OpenCV."""

    def __init__(self, *, cv2_module: Any | None = None) -> None:
        self._cv2 = cv2_module if cv2_module is not None else _load_cv2()

    def is_available(self) -> bool:
        return self._cv2 is not None

    def open(self, path: str | Path) -> OpenCvVideoCaptureSession | None:
        if self._cv2 is None:
            return None
        capture = self._cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            return None
        return OpenCvVideoCaptureSession(capture=capture, cv2=self._cv2)

    def probe(self, path: str | Path) -> dict[str, Any]:
        session = self.open(path)
        if session is None:
            return {}
        try:
            fps = session.get_fps()
            frame_count = session.get_frame_count()
            duration = None
            if fps and frame_count:
                duration = float(frame_count) / float(fps)
            width, height = session.get_frame_size()
            metadata: dict[str, Any] = {
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration,
            }
            if width is not None:
                metadata["width"] = width
            if height is not None:
                metadata["height"] = height
            return metadata
        finally:
            session.close()

    def sample_uniform(
        self,
        path: str | Path,
        *,
        n: int,
    ) -> list[dict[str, Any]]:
        session = self.open(path)
        if session is None or n <= 0:
            return []
        try:
            frame_count = int(session.get_frame_count() or 0)
            fps = float(session.get_fps() or 0.0) or 25.0
            if frame_count <= 0:
                frames: list[dict[str, Any]] = []
                for index in range(n):
                    ok, frame = session.read_frame()
                    if not ok or frame is None:
                        break
                    frames.append(
                        {
                            "frame": frame,
                            "timestamp_seconds": index / fps,
                            "frame_index": index,
                        }
                    )
                return frames
            indices = [
                min(
                    frame_count - 1,
                    int(round(i * (frame_count - 1) / max(1, n - 1))),
                )
                for i in range(n)
            ]
            sampled: list[dict[str, Any]] = []
            for index in indices:
                session.set_frame_index(index)
                ok, frame = session.read_frame()
                if not ok or frame is None:
                    continue
                sampled.append(
                    {
                        "frame": frame,
                        "timestamp_seconds": index / fps,
                        "frame_index": index,
                    }
                )
            return sampled
        finally:
            session.close()


class OpenCvFrameProcessor:
    """Frame-level transforms and I/O with OpenCV."""

    def __init__(self, *, cv2_module: Any | None = None) -> None:
        self._cv2 = cv2_module if cv2_module is not None else _load_cv2()

    def is_available(self) -> bool:
        return self._cv2 is not None

    def read_image(self, path: str | Path) -> Any | None:
        if self._cv2 is None:
            return None
        frame = self._cv2.imread(str(path), self._cv2.IMREAD_COLOR)
        return frame if frame is not None else None

    def write_image(self, path: str | Path, frame: Any) -> bool:
        if self._cv2 is None or frame is None:
            return False
        return bool(self._cv2.imwrite(str(path), frame))

    def resize(
        self,
        frame: Any,
        *,
        width: int,
        height: int,
    ) -> Any | None:
        if self._cv2 is None or frame is None:
            return None
        if width <= 0 or height <= 0:
            return None
        return self._cv2.resize(frame, (int(width), int(height)))

    def bgr_to_rgb(self, frame: Any) -> Any | None:
        if self._cv2 is None or frame is None:
            return None
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def bgr_to_gray(self, frame: Any) -> Any | None:
        if self._cv2 is None or frame is None:
            return None
        if getattr(frame, "ndim", 0) == 2:
            return frame
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)

    def laplacian_variance(self, frame: Any) -> float | None:
        if self._cv2 is None or frame is None:
            return None
        try:
            gray = frame
            if getattr(frame, "ndim", 0) == 3:
                gray = self.bgr_to_gray(frame)
            if gray is None:
                return None
            return float(self._cv2.Laplacian(gray, self._cv2.CV_64F).var())
        except (TypeError, ValueError, RuntimeError):
            return None

    def absdiff(self, left: Any, right: Any) -> Any | None:
        if self._cv2 is None or left is None or right is None:
            return None
        return self._cv2.absdiff(left, right)

    def encode_jpeg(self, frame: Any) -> bytes | None:
        if self._cv2 is None or frame is None:
            return None
        ok, encoded = self._cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return bytes(encoded.tobytes())


class OpenCvVideoClipWriter:
    """Write short resized clips with OpenCV."""

    def __init__(self, *, cv2_module: Any | None = None) -> None:
        self._cv2 = cv2_module if cv2_module is not None else _load_cv2()
        self._reader = OpenCvVideoReader(cv2_module=self._cv2)
        self._processor = OpenCvFrameProcessor(cv2_module=self._cv2)

    def is_available(self) -> bool:
        return self._cv2 is not None and self._reader.is_available()

    def write_resized_clip(
        self,
        *,
        source_path: Path,
        output_path: Path,
        max_duration_seconds: float,
        output_fps: float,
        output_size: tuple[int, int],
    ) -> None:
        if self._cv2 is None:
            raise RuntimeError("opencv_unavailable")
        session = self._reader.open(source_path)
        if session is None:
            raise RuntimeError(f"video_open_failed:{source_path}")
        tmp_path = output_path.with_suffix(".tmp.mp4")
        width, height = int(output_size[0]), int(output_size[1])
        fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
        writer = self._cv2.VideoWriter(
            str(tmp_path),
            fourcc,
            float(max(1, output_fps)),
            (width, height),
        )
        if not writer.isOpened():
            session.close()
            raise RuntimeError("video_writer_open_failed")
        try:
            max_frames = max(1, int(max_duration_seconds * max(1, output_fps)))
            written = 0
            while written < max_frames:
                ok, frame = session.read_frame()
                if not ok or frame is None:
                    break
                resized = self._processor.resize(
                    frame,
                    width=width,
                    height=height,
                )
                if resized is None:
                    continue
                writer.write(resized)
                written += 1
        finally:
            writer.release()
            session.close()
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError("video_clip_write_failed")
        tmp_path.replace(output_path)


class OpenCvVideoFrameCodec:
    """Higher-level RGB frame grid codec for multimodal tokenization."""

    def __init__(
        self,
        *,
        video_reader: OpenCvVideoReader | None = None,
        frame_processor: OpenCvFrameProcessor | None = None,
        cv2_module: Any | None = None,
    ) -> None:
        cv2 = cv2_module if cv2_module is not None else _load_cv2()
        self._reader = video_reader or OpenCvVideoReader(cv2_module=cv2)
        self._processor = frame_processor or OpenCvFrameProcessor(
            cv2_module=cv2
        )
        self._cv2 = cv2

    def read_frames(
        self,
        *,
        video_path: Path,
        frame_count: int,
        height: int,
        width: int,
    ) -> list[NDArray[np.uint8]]:
        if (
            not self._reader.is_available()
            or not self._processor.is_available()
        ):
            raise RuntimeError("opencv_required_for_video_tokenization")
        session = self._reader.open(video_path)
        if session is None:
            raise RuntimeError(f"video_decode_failed:{video_path}")
        try:
            total_frames = int(session.get_frame_count() or 0)
            if total_frames <= 0:
                total_frames = frame_count
            indices = _sample_frame_indices(
                total_frames=total_frames,
                frame_count=frame_count,
            )
            frames: list[NDArray[np.uint8]] = []
            for index in indices:
                session.set_frame_index(index)
                ok, frame_bgr = session.read_frame()
                if not ok or frame_bgr is None:
                    continue
                resized = self._processor.resize(
                    frame_bgr,
                    width=width,
                    height=height,
                )
                if resized is None:
                    continue
                frame_rgb = self._processor.bgr_to_rgb(resized)
                if frame_rgb is None:
                    continue
                frames.append(frame_rgb)
        finally:
            session.close()
        if not frames:
            raise RuntimeError(f"video_decode_failed:{video_path}")
        while len(frames) < frame_count:
            frames.append(frames[-1].copy())
        return frames[:frame_count]

    def write_frames(
        self,
        *,
        frames_rgb: list[NDArray[np.uint8]],
        output_path: Path,
        fps: int,
    ) -> None:
        if self._cv2 is None or not self._processor.is_available():
            raise RuntimeError("opencv_required_for_video_encoding")
        if not frames_rgb:
            raise RuntimeError("video_encode_failed:empty_frames")
        height, width = frames_rgb[0].shape[:2]
        tmp_path = output_path.with_suffix(".tmp.mp4")
        fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
        writer = self._cv2.VideoWriter(
            str(tmp_path),
            fourcc,
            float(max(1, fps)),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("video_writer_open_failed")
        try:
            for frame_rgb in frames_rgb:
                frame_bgr = self._cv2.cvtColor(
                    frame_rgb,
                    self._cv2.COLOR_RGB2BGR,
                )
                writer.write(frame_bgr)
        finally:
            writer.release()
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError("video_encode_failed:empty_output")
        tmp_path.replace(output_path)


def _sample_frame_indices(*, total_frames: int, frame_count: int) -> list[int]:
    if total_frames <= 1:
        return [0] * frame_count
    if frame_count <= 1:
        return [0]
    step = max(1, (total_frames - 1) // max(1, frame_count - 1))
    return [
        min(total_frames - 1, index * step) for index in range(frame_count)
    ]
