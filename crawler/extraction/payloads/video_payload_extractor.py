"""Objective video payload metadata from already-fetched bytes/files.

No OCR, transcription, scene graphs, keyframe sampling, or quality scoring.
"""

from __future__ import annotations

import hashlib
import struct
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VideoPayloadExtractionResult:
    """Deterministic properties of one video payload."""

    duration_seconds: float | None
    width: int | None
    height: int | None
    fps: float | None
    frame_count: int | None
    format: str | None
    byte_size: int
    sha256: str

    def as_metadata_dict(self) -> dict[str, Any]:
        """Return the canonical persisted video metadata mapping."""

        return {
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "format": self.format,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


class VideoPayloadExtractor:
    """Extract objective video payload metadata from raw bytes or a path."""

    def extract(self, *, body: bytes) -> VideoPayloadExtractionResult | None:
        """Return payload metadata, or ``None`` when unreadable.

        Empty bodies always return ``None``. When only partial stream fields are
        available, those fields may be ``None`` while ``byte_size`` and
        ``sha256`` remain set.
        """

        if not body:
            return None

        byte_size = len(body)
        sha256 = hashlib.sha256(body).hexdigest()
        format_name = _guess_format(body=body)

        # Prefer in-memory MP4 atom parsing (no temp file).
        mp4_meta = _read_mp4_atoms(body=body)
        if mp4_meta is not None:
            return VideoPayloadExtractionResult(
                duration_seconds=mp4_meta.get("duration_seconds"),
                width=mp4_meta.get("width"),
                height=mp4_meta.get("height"),
                fps=None,
                frame_count=None,
                format=format_name or "MP4",
                byte_size=byte_size,
                sha256=sha256,
            )

        # Optional OpenCV / installed video backend via temporary file.
        opencv_meta = _probe_with_optional_opencv(body=body)
        if opencv_meta is None:
            # Still return identity when container magic is known.
            if format_name is None:
                return None
            return VideoPayloadExtractionResult(
                duration_seconds=None,
                width=None,
                height=None,
                fps=None,
                frame_count=None,
                format=format_name,
                byte_size=byte_size,
                sha256=sha256,
            )

        return VideoPayloadExtractionResult(
            duration_seconds=opencv_meta.get("duration_seconds"),
            width=opencv_meta.get("width"),
            height=opencv_meta.get("height"),
            fps=opencv_meta.get("fps"),
            frame_count=opencv_meta.get("frame_count"),
            format=format_name or opencv_meta.get("format"),
            byte_size=byte_size,
            sha256=sha256,
        )

    def extract_from_path(
        self,
        *,
        path: Path,
    ) -> VideoPayloadExtractionResult | None:
        """Extract objective metadata from a local video path."""

        try:
            body = path.read_bytes()
        except OSError:
            return None
        return self.extract(body=body)


def _guess_format(*, body: bytes) -> str | None:
    if len(body) >= 12 and body[4:8] == b"ftyp":
        brand = body[8:12].decode("ascii", errors="ignore").strip().upper()
        if brand in {"QT  ", "QT"}:
            return "MOV"
        return "MP4"
    if body.startswith(b"\x1a\x45\xdf\xa3"):
        return "WEBM"
    if body.startswith(b"RIFF") and body[8:12] == b"AVI ":
        return "AVI"
    if body.startswith(b"OggS"):
        return "OGG"
    return None


def _probe_with_optional_opencv(*, body: bytes) -> dict[str, Any] | None:
    try:
        import cv2
    except ImportError:
        return None

    suffix = ".mp4"
    if body.startswith(b"\x1a\x45\xdf\xa3"):
        suffix = ".webm"
    elif body.startswith(b"RIFF") and body[8:12] == b"AVI ":
        suffix = ".avi"

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as handle:
            handle.write(body)
            temp_path = Path(handle.name)
    except OSError:
        return None

    try:
        capture = cv2.VideoCapture(str(temp_path))
        if not capture.isOpened():
            return None
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()
    except Exception:
        return None
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    duration = frame_count / fps if fps > 0.0 and frame_count > 0 else None
    if (
        duration is None
        and width <= 0
        and height <= 0
        and frame_count <= 0
        and fps <= 0
    ):
        return None

    return {
        "duration_seconds": duration,
        "width": width if width > 0 else None,
        "height": height if height > 0 else None,
        "fps": fps if fps > 0 else None,
        "frame_count": frame_count if frame_count > 0 else None,
        "format": None,
    }


def _read_mp4_atoms(*, body: bytes) -> dict[str, Any] | None:
    if len(body) < 12 or body[4:8] != b"ftyp":
        # Still try moov search for incomplete/tail probes that lack ftyp.
        if b"moov" not in body:
            return None

    moov = _extract_named_atom(data=body, atom_type=b"moov")
    if moov is None:
        return None

    metadata: dict[str, Any] = {}
    duration = _extract_mvhd_duration(data=moov)
    if duration is not None:
        metadata["duration_seconds"] = duration
    width, height = _extract_tkhd_dimensions(data=moov)
    if width is not None:
        metadata["width"] = width
    if height is not None:
        metadata["height"] = height
    if not metadata:
        return None
    return metadata


def _extract_named_atom(*, data: bytes, atom_type: bytes) -> bytes | None:
    if len(atom_type) != 4:
        return None
    offset = 0
    while True:
        index = data.find(atom_type, offset)
        if index < 4:
            return None
        atom_start = index - 4
        atom = _slice_atom_at(data=data, offset=atom_start)
        if atom is not None:
            return atom
        offset = index + 4


def _slice_atom_at(*, data: bytes, offset: int) -> bytes | None:
    if offset < 0 or offset + 8 > len(data):
        return None
    size = int.from_bytes(data[offset : offset + 4], "big")
    header_size = 8
    if size == 1:
        if offset + 16 > len(data):
            return None
        size = int.from_bytes(data[offset + 8 : offset + 16], "big")
        header_size = 16
    if size < header_size:
        return None
    atom_end = offset + size
    if atom_end > len(data):
        return None
    return data[offset:atom_end]


def _iter_child_atoms(*, data: bytes, start: int = 8) -> Iterator[bytes]:
    offset = start
    while offset + 8 <= len(data):
        atom = _slice_atom_at(data=data, offset=offset)
        if atom is None:
            break
        yield atom
        offset += len(atom)


def _extract_mvhd_duration(*, data: bytes) -> float | None:
    for atom in _iter_child_atoms(data=data):
        if atom[4:8] == b"mvhd":
            return _parse_mvhd_duration(atom=atom)
    return None


def _parse_mvhd_duration(*, atom: bytes) -> float | None:
    if len(atom) < 28:
        return None
    version = atom[8]
    body = atom[8:]
    if version == 1:
        if len(body) < 36:
            return None
        timescale = int.from_bytes(body[20:24], "big")
        duration = int.from_bytes(body[24:32], "big")
    else:
        timescale = int.from_bytes(body[12:16], "big")
        duration = int.from_bytes(body[16:20], "big")
    if timescale <= 0:
        return None
    return duration / float(timescale)


def _extract_tkhd_dimensions(
    *,
    data: bytes,
) -> tuple[int | None, int | None]:
    for atom in _iter_child_atoms(data=data):
        if atom[4:8] != b"trak":
            continue
        for child in _iter_child_atoms(data=atom):
            if child[4:8] != b"tkhd":
                continue
            width, height = _parse_tkhd_dimensions(atom=child)
            if width and height:
                return width, height
    return None, None


def _parse_tkhd_dimensions(*, atom: bytes) -> tuple[int | None, int | None]:
    if len(atom) < 92:
        return None, None
    version = atom[8]
    if version == 1:
        if len(atom) < 104:
            return None, None
        width_offset = 96
        height_offset = 100
    else:
        width_offset = 84
        height_offset = 88
    if height_offset + 4 > len(atom):
        return None, None
    # 16.16 fixed point.
    width = struct.unpack(">I", atom[width_offset : width_offset + 4])[0] >> 16
    height = (
        struct.unpack(">I", atom[height_offset : height_offset + 4])[0] >> 16
    )
    if width <= 0 or height <= 0:
        return None, None
    return int(width), int(height)
