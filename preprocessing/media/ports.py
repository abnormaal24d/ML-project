"""Media schemas used by crawler, augmentation, and multimodal code.

Concrete codecs live in ``preprocessing.media.adapters``. Domain modules
depend only on these protocols and result types.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """Chunked PCM audio stream for augmentation transforms."""

    channels: int
    sample_width: int
    sample_rate: int
    duration_sec: float
    frames_iterator: Iterator[bytes]


@dataclass(frozen=True, slots=True)
class VideoAudioTrackResult:
    """Result of extracting an audio track from a video container."""

    extracted: bool
    audio_path: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    duration_seconds: float | None = None
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class ImageNormalizationResult:
    """Normalized image bytes and format metadata for training storage."""

    normalized_bytes: bytes | None
    format: str | None = None
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    was_oriented: bool = False
    was_converted: bool = False
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class VideoNormalizationResult:
    """Normalized video file produced for training/analysis."""

    normalized_path: str | None
    status: str = "passed"
    error_type: str | None = None


IMAGE_NORMALIZATION_SOFT_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    ImportError,
)


class ImageLoader(Protocol):
    def open_image(self, *, body: bytes) -> Any | None: ...


class ImageHashCalculator(Protocol):
    def compute(
        self,
        *,
        body: bytes | None = None,
        img: Any = None,
    ) -> str | None: ...


class ImageMetadataAssembler(Protocol):
    def assemble(
        self,
        *,
        img: Any,
        body: bytes,
        fmt: str | None = None,
    ) -> Any: ...


class FrameProcessor(Protocol):
    def is_available(self) -> bool: ...

    def read_image(self, path: str | Path) -> Any | None: ...

    def write_image(self, path: str | Path, frame: Any) -> bool: ...

    def resize(
        self,
        frame: Any,
        *,
        width: int,
        height: int,
    ) -> Any | None: ...

    def bgr_to_rgb(self, frame: Any) -> Any | None: ...

    def bgr_to_gray(self, frame: Any) -> Any | None: ...

    def laplacian_variance(self, frame: Any) -> float | None: ...

    def absdiff(self, left: Any, right: Any) -> Any | None: ...

    def encode_jpeg(self, frame: Any) -> bytes | None: ...


class VideoCaptureSession(Protocol):
    def get_fps(self) -> float | None: ...

    def get_frame_count(self) -> int | None: ...

    def set_frame_index(self, frame_index: int) -> None: ...

    def set_timestamp_ms(self, timestamp_ms: float) -> None: ...

    def read_frame(self) -> tuple[bool, Any]: ...

    def close(self) -> None: ...


class VideoReader(Protocol):
    def is_available(self) -> bool: ...

    def open(self, path: str | Path) -> VideoCaptureSession | None: ...

    def probe(self, path: str | Path) -> dict[str, Any]: ...

    def sample_uniform(
        self,
        path: str | Path,
        *,
        n: int,
    ) -> list[dict[str, Any]]: ...


class EmbeddedMetadataAdapter(Protocol):
    """Inspect and remove metadata through injected local media tooling."""

    def inspect(
        self,
        *,
        path: Path,
        modality: str,
    ) -> tuple[str, ...] | None: ...

    def remove(
        self,
        *,
        source: Path,
        destination: Path,
        modality: str,
    ) -> bool: ...

    def has_audio_stream(self, path: Path) -> bool | None: ...

    def provenance(self) -> dict[str, object]: ...


class AudioDecodeBackend(Protocol):
    def decode(
        self,
        *,
        path: Path,
        chunk_frames: int,
    ) -> DecodedAudio: ...


class AudioTrackExtractor(Protocol):
    def extract_to_wav(
        self,
        *,
        video_path: str | None = None,
        body: bytes | None = None,
        output_dir: str | None = None,
        target_sample_rate: int = 16000,
    ) -> VideoAudioTrackResult: ...


class VideoFrameCodec(Protocol):
    def read_frames(
        self,
        *,
        video_path: Path,
        frame_count: int,
        height: int,
        width: int,
    ) -> list[Any]: ...

    def write_frames(
        self,
        *,
        frames_rgb: list[Any],
        output_path: Path,
        fps: int,
    ) -> None: ...
