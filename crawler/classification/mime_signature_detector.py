"""Strong MIME signature detection over a bounded byte sample."""

from __future__ import annotations

from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.settings.classification import MimeSignatureDetectorSettings

try:
    import filetype  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    filetype = None


class MimeSignatureDetector:
    """Resolve strong binary or textual magic signatures."""

    def __init__(
        self, settings: MimeSignatureDetectorSettings, logger: ProjectLogger
    ) -> None:
        self._settings = settings
        self._logger = logger

    def detect(self, *, sample: bytes) -> str | None:
        if not self._settings.enabled or not sample:
            return None

        window_size = max(
            int(self._settings.sample_size_bytes),
            int(self._settings.maximum_signature_size),
        )
        window = sample[:window_size]
        if window.startswith(b"%PDF") and not window.startswith(b"%PDF-"):
            return None

        detected = self._detect_known_signature(sample=window)
        if detected is not None:
            return detected

        if self._settings.use_filetype and filetype is not None:
            guess = filetype.guess(window)
            if guess is not None:
                return str(guess.mime).casefold()

        self._logger.debug("mime_signature_detector_no_match")
        return None

    def _detect_known_signature(self, *, sample: bytes) -> str | None:
        stripped = sample.lstrip()
        lower_sample = sample[:1024].lower()
        if sample.startswith(b"%PDF-"):
            return "application/pdf"
        if sample.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if sample.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if sample.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if stripped.startswith(b"<svg") or (
            stripped.startswith(b"<?xml") and b"<svg" in lower_sample
        ):
            return "image/svg+xml"
        if sample.startswith(b"RIFF") and len(sample) >= 12:
            return self._detect_riff(sample)
        if sample.startswith(b"OggS"):
            return self._detect_ogg(sample)
        if sample.startswith(b"ID3") or sample.startswith(b"\xff\xfb"):
            return "audio/mpeg"
        if (
            len(sample) >= 2
            and sample[0] == 0xFF
            and (sample[1] & 0xF6) == 0xF0
        ):
            return "audio/aac"
        if sample.startswith(b"fLaC"):
            return "audio/flac"
        if sample.startswith((b"II*\x00", b"MM\x00*")):
            return "image/tiff"
        if sample.startswith(b"\x1f\x8b"):
            return "application/gzip"
        if len(sample) > 262 and sample[257:262] == b"ustar":
            return "application/x-tar"
        if sample.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
            return self._detect_zip_container(sample)
        if sample.startswith(b"\x1aE\xdf\xa3"):
            return self._detect_ebml(sample)
        return self._detect_ftyp(sample)

    @staticmethod
    def _detect_riff(sample: bytes) -> str | None:
        return {
            b"WAVE": "audio/wav",
            b"WEBP": "image/webp",
            b"AVI ": "video/x-msvideo",
        }.get(sample[8:12])

    @staticmethod
    def _detect_ogg(sample: bytes) -> str:
        lower = sample[:512].lower()
        if b"theora" in lower or b"fishead" in lower:
            return "video/ogg"
        if b"opushead" in lower or b"vorbis" in lower:
            return "audio/ogg"
        return "application/ogg"

    @staticmethod
    def _detect_zip_container(sample: bytes) -> str:
        lower = sample.lower()
        if (
            b"application/epub+zip" in lower[:4096]
            or b"meta-inf/container.xml" in lower
        ):
            return "application/epub+zip"
        if b"[content_types].xml" in lower:
            if b"word/" in lower:
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if b"xl/" in lower:
                return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if b"ppt/" in lower:
                return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        return "application/zip"

    @staticmethod
    def _detect_ebml(sample: bytes) -> str:
        lower = sample[:1024].lower()
        has_audio = b"a_opus" in lower or b"a_vorbis" in lower
        has_video = b"v_vp8" in lower or b"v_vp9" in lower
        if b"webm" in lower:
            if has_audio and not has_video:
                return "audio/webm"
            return "video/webm"
        if b"matroska" in lower:
            return "video/x-matroska"
        return "video/webm"

    @staticmethod
    def _detect_ftyp(sample: bytes) -> str | None:
        index = sample[:16].find(b"ftyp")
        if index < 0:
            return None
        brand = sample[index + 4 : index + 8].lower()
        compatible = sample[index + 8 : index + 32].lower()
        brands = {brand} | {
            compatible[offset : offset + 4]
            for offset in range(0, len(compatible), 4)
            if len(compatible[offset : offset + 4]) == 4
        }
        if brands & {b"avif", b"avis"}:
            return "image/avif"
        if brands & {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
            return "image/heic"
        if brands & {b"m4a ", b"m4b ", b"m4p ", b"mp4a"}:
            return "audio/mp4"
        if b"qt  " in brands:
            return "video/quicktime"
        return "video/mp4"
