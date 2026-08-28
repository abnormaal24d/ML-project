"""MP4 tail metadata parsing."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any


class Mp4TailMetadataReader:
    """Extract lightweight MP4 metadata from a downloaded tail range."""

    def read(self, *, path: Path) -> dict[str, Any]:
        """Parse MP4 moov metadata from a tail byte range."""

        try:
            data = path.read_bytes()
        except OSError:
            return {}

        moov = _extract_named_atom(data=data, atom_type=b"moov")
        if moov is None:
            return {}

        metadata: dict[str, Any] = {}
        duration = _extract_mvhd_duration(data=moov)
        if duration is not None:
            metadata["duration_seconds"] = duration

        width, height = _extract_tkhd_dimensions(data=moov)
        if width is not None:
            metadata["width"] = width
        if height is not None:
            metadata["height"] = height

        return metadata


def _extract_named_atom(
    *,
    data: bytes,
    atom_type: bytes,
) -> bytes | None:
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

    return duration / timescale


def _extract_tkhd_dimensions(
    *,
    data: bytes,
) -> tuple[int | None, int | None]:
    width: int | None = None
    height: int | None = None

    for atom in _iter_child_atoms(data=data):
        if atom[4:8] != b"trak":
            continue

        for child in _iter_child_atoms(data=atom):
            if child[4:8] != b"tkhd":
                continue

            track_width, track_height = _parse_tkhd_dimensions(atom=child)
            if track_width is None or track_height is None:
                continue
            if track_width <= 0 or track_height <= 0:
                continue
            if (
                width is None
                or height is None
                or track_width * track_height > width * height
            ):
                width = track_width
                height = track_height

    return width, height


def _parse_tkhd_dimensions(
    *,
    atom: bytes,
) -> tuple[int | None, int | None]:
    if len(atom) < 9:
        return None, None

    version = atom[8]
    body = atom[8:]
    width_offset = 88 if version == 1 else 76
    height_offset = 92 if version == 1 else 80

    if len(body) < height_offset + 4:
        return None, None

    width = _fixed_16_16_to_int(body[width_offset : width_offset + 4])
    height = _fixed_16_16_to_int(body[height_offset : height_offset + 4])
    return width, height


def _fixed_16_16_to_int(value: bytes) -> int | None:
    if len(value) != 4:
        return None
    raw = int.from_bytes(value, "big")
    return int(round(raw / 65_536))
