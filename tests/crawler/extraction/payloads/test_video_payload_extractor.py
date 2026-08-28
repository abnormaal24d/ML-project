"""Tests for objective VideoPayloadExtractor (no OCR / scoring)."""

from __future__ import annotations

import hashlib
import struct

from crawler.extraction.payloads.video_payload_extractor import (
    VideoPayloadExtractionResult,
    VideoPayloadExtractor,
)


def _atom(atom_type: bytes, payload: bytes) -> bytes:
    size = 8 + len(payload)
    return struct.pack(">I", size) + atom_type + payload


def _minimal_mp4(
    *,
    timescale: int = 1000,
    duration: int = 2500,
    width: int = 320,
    height: int = 240,
) -> bytes:
    """Build a tiny ftyp+moov MP4 with duration and track dimensions."""

    # mvhd version 0 body after version/flags
    mvhd_body = bytearray(100)
    mvhd_body[0] = 0  # version
    # creation/modification already zero
    struct.pack_into(">I", mvhd_body, 12, timescale)
    struct.pack_into(">I", mvhd_body, 16, duration)
    # rate 1.0
    struct.pack_into(">I", mvhd_body, 20, 0x00010000)
    # volume 1.0
    struct.pack_into(">H", mvhd_body, 24, 0x0100)

    # tkhd version 0: width/height at offsets 84/88 of atom (including header)
    # atom = size(4)+type(4)+version/flags(4)+... => fixed fields in body
    tkhd_body = bytearray(84)
    tkhd_body[0] = 0  # version
    # width/height are 16.16 fixed at body offsets 76/80 for version 0
    # body offset 0 is version; full atom offsets: 8+version_block
    # _parse_tkhd uses width_offset=84 on full atom (including 8-byte header)
    # so body index = 84-8 = 76
    struct.pack_into(">I", tkhd_body, 76, width << 16)
    struct.pack_into(">I", tkhd_body, 80, height << 16)

    mvhd = _atom(b"mvhd", bytes(mvhd_body))
    tkhd = _atom(b"tkhd", bytes(tkhd_body))
    trak = _atom(b"trak", tkhd)
    moov = _atom(b"moov", mvhd + trak)
    ftyp = _atom(b"ftyp", b"isom" + b"\x00\x00\x00\x00" + b"isom")
    return ftyp + moov


def test_extract_mp4_duration_and_dimensions() -> None:
    body = _minimal_mp4(duration=2500, timescale=1000, width=640, height=360)
    result = VideoPayloadExtractor().extract(body=body)
    assert isinstance(result, VideoPayloadExtractionResult)
    assert result.format == "MP4"
    assert result.duration_seconds == 2.5
    assert result.width == 640
    assert result.height == 360
    assert result.byte_size == len(body)
    assert result.sha256 == hashlib.sha256(body).hexdigest()


def test_extract_empty_body_returns_none() -> None:
    assert VideoPayloadExtractor().extract(body=b"") is None


def test_extract_garbage_returns_none() -> None:
    assert VideoPayloadExtractor().extract(body=b"not-a-video") is None


def test_result_has_no_enrichment_fields() -> None:
    body = _minimal_mp4()
    result = VideoPayloadExtractor().extract(body=body)
    assert result is not None
    field_names = set(result.__dataclass_fields__)
    forbidden = {
        "transcript_text",
        "ocr_text",
        "keyframes",
        "scene_graph",
        "quality_score",
        "blur_variance",
    }
    assert field_names.isdisjoint(forbidden)


def test_as_metadata_dict_keys() -> None:
    body = _minimal_mp4()
    result = VideoPayloadExtractor().extract(body=body)
    assert result is not None
    payload = result.as_metadata_dict()
    assert payload["duration_seconds"] == result.duration_seconds
    assert payload["width"] == result.width
    assert "sha256" in payload
