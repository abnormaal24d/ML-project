"""FetchedPayload cleanup also removes derived media scratch siblings."""

from __future__ import annotations

from pathlib import Path

from crawler.fetching.results.payload import FetchedPayload


def _payload(*, path: Path) -> FetchedPayload:
    return FetchedPayload(
        temp_path=path,
        byte_size=4,
        sha256_hex="a" * 64,
        sniff_bytes=b"data",
        chunk_count=1,
    )


def test_cleanup_removes_normalized_siblings(tmp_path: Path) -> None:
    payload_file = tmp_path / "fetch-abc.bin"
    payload_file.write_bytes(b"data")
    normalized_image = tmp_path / "fetch-abc.bin.normalized.jpg"
    normalized_image.write_bytes(b"image")
    normalized_video = tmp_path / "fetch-abc.bin.normalized.mp4"
    normalized_video.write_bytes(b"video")

    _payload(path=payload_file).cleanup()

    assert not payload_file.exists()
    assert not normalized_image.exists()
    assert not normalized_video.exists()


def test_cleanup_removes_extracted_audio_sibling(tmp_path: Path) -> None:
    payload_file = tmp_path / "clip.mp4"
    payload_file.write_bytes(b"data")
    audio = tmp_path / "clip_audio_16000.wav"
    audio.write_bytes(b"wav")

    _payload(path=payload_file).cleanup()

    assert not payload_file.exists()
    assert not audio.exists()


def test_cleanup_ignores_unrelated_siblings(tmp_path: Path) -> None:
    payload_file = tmp_path / "fetch-abc.bin"
    payload_file.write_bytes(b"data")
    unrelated = tmp_path / "other.normalized.jpg"
    unrelated.write_bytes(b"keep")
    other_audio = tmp_path / "other_audio_16000.wav"
    other_audio.write_bytes(b"keep")

    _payload(path=payload_file).cleanup()

    assert not payload_file.exists()
    assert unrelated.exists()
    assert other_audio.exists()


def test_cleanup_is_idempotent_for_missing_payload(tmp_path: Path) -> None:
    payload_file = tmp_path / "fetch-abc.bin"
    _payload(path=payload_file).cleanup()
    _payload(path=payload_file).cleanup()
