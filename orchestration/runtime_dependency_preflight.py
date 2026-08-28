"""Runtime dependency preflight for enabled configuration features.

Configuration may enable features that require Python modules or system
executables. This module verifies the exact requirements the active
configuration selects and collects every missing requirement before the
crawl starts.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config.errors import RuntimeDependencyError
from config.settings.root import Settings
from preprocessing.media.adapters.versioned_executable import (
    ExecutableVerificationError,
    resolve_and_verify_executable,
    resolve_executable,
)

if TYPE_CHECKING:
    from config.preprocessing.media_settings import DiarizationSettings


@dataclass(frozen=True, slots=True)
class OptionalDependencyReport:
    """Status snapshot after required dependencies were verified present."""

    optional_dependency_status: dict[str, object]
    summary: str
    preprocessing_backends: dict[str, str] | None = None

    @classmethod
    def from_status(
        cls,
        optional_dependency_status: dict[str, object],
    ) -> OptionalDependencyReport:
        backends: list[str] = []
        ocr = optional_dependency_status.get("ocr_backend")
        asr = optional_dependency_status.get("asr_backend")
        if ocr:
            backends.append(f"ocr:{ocr}")
        if asr:
            backends.append(f"asr:{asr}")
        summary = "ok"
        if backends:
            summary = "ok," + ",".join(backends)
        return cls(
            optional_dependency_status=optional_dependency_status,
            summary=summary,
        )


def validate_optional_dependencies(
    *,
    settings: Settings,
) -> OptionalDependencyReport:
    """Validate required runtime dependencies for the active configuration.

    Raises ``RuntimeError`` with every missing requirement when any check
    fails. Callers wrap this as ``ApplicationContainerBuildError``.
    """

    missing: list[str] = []
    processors = settings.collection.processors
    ocr_backend = str(settings.preprocessing.ocr.backend).strip().lower()
    any_ocr = bool(
        processors.image.run_ocr
        or processors.video.run_ocr
        or processors.document.run_ocr
    )

    transcription_required = bool(
        settings.preprocessing.transcription.enabled
        and (
            processors.audio.run_transcription
            or processors.video.run_transcription
        )
    )
    if transcription_required:
        _require_module(
            "faster_whisper",
            "speech transcription",
            missing=missing,
        )

    diarization = settings.preprocessing.diarization
    if diarization.enabled and diarization.backend == "pyannote":
        _validate_pyannote_runtime(settings=diarization, missing=missing)

    if any_ocr and ocr_backend != "disabled":
        if ocr_backend == "tesseract":
            _require_module(
                "pytesseract",
                "tesseract OCR",
                missing=missing,
            )
            _require_executable(
                "tesseract",
                "tesseract OCR",
                missing=missing,
            )
        elif ocr_backend == "rapidocr":
            _require_module(
                "rapidocr_onnxruntime",
                "RapidOCR",
                missing=missing,
            )
        else:
            missing.append(
                f"unsupported OCR backend '{ocr_backend}' "
                "(expected disabled|tesseract|rapidocr)"
            )

    if processors.document.extract_text:
        _require_module("pypdf", "PDF text extraction", missing=missing)

    if processors.document.run_ocr and ocr_backend != "disabled":
        _require_module("pdf2image", "PDF OCR", missing=missing)
        _require_executable("pdfinfo", "PDF OCR", missing=missing)
        _require_executable("pdftoppm", "PDF OCR", missing=missing)

    if processors.image.extract_metadata or processors.image.run_ocr:
        _require_module("PIL", "image processing", missing=missing)

    audio_augmentation_required = bool(
        settings.augmentation.enabled and settings.augmentation.audio.enabled
    )
    audio_fingerprint_settings = settings.preprocessing.audio_validation
    audio_fingerprint_required = bool(
        processors.audio.enabled
        and audio_fingerprint_settings.enabled
        and audio_fingerprint_settings.require_audio_fingerprint
    )
    video_augmentation_required = bool(
        settings.augmentation.enabled and settings.augmentation.video.enabled
    )
    video_augmentation_settings = settings.augmentation.video

    resolved_ffmpeg = None
    resolved_ffprobe = None
    resolved_fpcalc = None

    if audio_fingerprint_required:
        resolved_fpcalc = _require_executable_version(
            tool_name="fpcalc",
            configured_executable=audio_fingerprint_settings.chromaprint_executable,
            expected_version=audio_fingerprint_settings.chromaprint_expected_version,
            timeout_seconds=audio_fingerprint_settings.chromaprint_timeout_seconds,
            feature_name="audio fingerprinting",
            missing=missing,
        )
    if video_augmentation_required:
        media_toolchain = settings.media_toolchain
        resolved_ffmpeg = _require_executable_version(
            tool_name="ffmpeg",
            configured_executable=media_toolchain.ffmpeg_executable,
            expected_version=media_toolchain.ffmpeg_expected_version,
            timeout_seconds=video_augmentation_settings.probe_timeout_seconds,
            feature_name="video augmentation",
            missing=missing,
        )
        resolved_ffprobe = _require_executable_version(
            tool_name="ffprobe",
            configured_executable=media_toolchain.ffprobe_executable,
            expected_version=media_toolchain.ffprobe_expected_version,
            timeout_seconds=video_augmentation_settings.probe_timeout_seconds,
            feature_name="video augmentation probing",
            missing=missing,
        )
        _require_module(
            "PIL",
            "video keyframe validation",
            missing=missing,
        )

    if audio_augmentation_required:
        _require_module(
            "numpy",
            "audio augmentation",
            missing=missing,
        )
        _require_module(
            "scipy",
            "audio resampling and speed perturbation",
            missing=missing,
        )
        _require_module(
            "soundfile",
            "non-WAV audio augmentation",
            missing=missing,
        )

    if missing:
        raise RuntimeDependencyError(
            "missing required runtime dependencies:\n- " + "\n- ".join(missing)
        )

    status: dict[str, object] = {
        "optional_dependencies_validated": True,
        "ocr_backend": ocr_backend if any_ocr else "not_required",
        "asr_backend": (
            "whisper" if transcription_required else "not_required"
        ),
        "pillow_available": True,
        "pypdf_available": processors.document.extract_text
        or importlib.util.find_spec("pypdf") is not None,
        "faster_whisper_available": transcription_required
        or importlib.util.find_spec("faster_whisper") is not None,
        "ffmpeg_available": resolved_ffmpeg is not None,
        "mutagen_available": importlib.util.find_spec("mutagen") is not None,
        "opencv_available": importlib.util.find_spec("cv2") is not None,
        "media_decoder_available": (
            resolved_ffmpeg is not None
            or importlib.util.find_spec("av") is not None
        ),
        "media_decoder_backend": (
            "system_ffmpeg"
            if resolved_ffmpeg is not None
            else (
                "pyav" if importlib.util.find_spec("av") is not None else None
            )
        ),
        "lxml_available": importlib.util.find_spec("lxml") is not None,
        "aiodns_available": importlib.util.find_spec("aiodns") is not None,
        "async_dns_available": importlib.util.find_spec("aiodns") is not None,
        "pypdf_crypto_available": (
            importlib.util.find_spec("pypdf") is not None
            and importlib.util.find_spec("cryptography") is not None
        ),
        "cryptography_available": (
            importlib.util.find_spec("cryptography") is not None
        ),
        "tesseract_required": any_ocr and ocr_backend == "tesseract",
        "audio_augmentation_required": audio_augmentation_required,
        "audio_fingerprint_required": audio_fingerprint_required,
        "chromaprint_available": resolved_fpcalc is not None,
        "video_augmentation_required": video_augmentation_required,
        "ffprobe_available": resolved_ffprobe is not None,
        "numpy_available": (
            audio_augmentation_required
            or importlib.util.find_spec("numpy") is not None
        ),
        "scipy_available": (
            audio_augmentation_required
            or importlib.util.find_spec("scipy") is not None
        ),
        "soundfile_available": (
            audio_augmentation_required
            or importlib.util.find_spec("soundfile") is not None
        ),
    }
    return OptionalDependencyReport.from_status(status)


def _require_module(
    module_name: str,
    feature_name: str,
    *,
    missing: list[str],
) -> None:
    if importlib.util.find_spec(module_name) is None:
        missing.append(
            f"{feature_name} requires Python module '{module_name}'"
        )


def _require_executable(
    executable_name: str,
    feature_name: str,
    *,
    missing: list[str],
) -> None:
    if resolve_executable(executable_name) is None:
        missing.append(
            f"{feature_name} requires executable '{executable_name}' on PATH"
        )


def _require_executable_version(
    *,
    tool_name: str,
    configured_executable: str,
    expected_version: str | None,
    timeout_seconds: float,
    feature_name: str,
    missing: list[str],
) -> str | None:
    """Require a configured executable and its exact release version.

    Returns the resolved absolute path on success, None on failure (error added to missing).
    """
    if expected_version is None:
        missing.append(
            f"{feature_name} requires a configured expected executable version"
        )
        return None

    try:
        executable_path, _ = resolve_and_verify_executable(
            tool_name=tool_name,
            configured_executable=configured_executable,
            expected_version=expected_version,
            timeout_seconds=timeout_seconds,
            required=True,
        )
    except ExecutableVerificationError as error:
        missing.append(f"{feature_name}: {error}")
        return None

    return executable_path


def _validate_pyannote_runtime(
    *,
    settings: DiarizationSettings,
    missing: list[str],
) -> None:
    """Static preflight for the configured pyannote stack.

    Does not call ``Pipeline.from_pretrained`` — remote model loading belongs
    solely to the diarization backend builder.
    """

    from importlib.metadata import version as dist_version

    from packaging.version import Version

    try:
        import pyannote.audio as pyannote_audio  # type: ignore[import-not-found]
        import torch
        import torchaudio  # type: ignore[import-not-found]

        _ = pyannote_audio.Pipeline
    except Exception as exc:  # noqa: BLE001 — surface any import failure
        missing.append(
            "configured pyannote diarization runtime cannot be imported: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    torch_release = torch.__version__.split("+", 1)[0]
    torchaudio_release = torchaudio.__version__.split("+", 1)[0]
    if torch_release != torchaudio_release:
        missing.append(
            "pyannote diarization requires matching torch and torchaudio "
            f"releases; found torch={torch_release}, "
            f"torchaudio={torchaudio_release}"
        )

    # Remote Hugging Face loading is the path that hits the Hub auth API.
    local_model_path = settings.local_model_path
    if local_model_path is not None:
        return

    try:
        pyannote_release = Version(dist_version("pyannote.audio"))
        hub_release = Version(dist_version("huggingface_hub"))
    except Exception as exc:  # noqa: BLE001
        missing.append(
            "cannot determine pyannote/huggingface_hub package versions: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    if pyannote_release.major < 4 and hub_release.major >= 1:
        missing.append(
            "configured pyannote remote loading is incompatible with the "
            f"installed Hub client: pyannote.audio {pyannote_release} "
            "still calls hf_hub_download(..., use_auth_token=...), "
            f"but huggingface_hub {hub_release} removed that parameter. "
            "Pin huggingface_hub to a 0.x release for pyannote 3.x, "
            "or migrate pyannote and the model path together."
        )


def image_processing_requires_pillow(*, settings: Settings) -> bool:
    processors = settings.collection.processors
    image_settings = processors.image
    return bool(
        image_settings.enabled
        and (image_settings.extract_metadata or image_settings.run_ocr)
    )


def audio_video_processing_enabled(
    *,
    settings: Settings,
    report: OptionalDependencyReport,
) -> bool:
    """Return whether audio/video processing can run with current deps."""

    del settings
    status = report.optional_dependency_status
    return bool(status.get("media_decoder_available"))


def audio_video_processing_requires_decoder(*, settings: Settings) -> bool:
    processors = settings.collection.processors
    return bool(
        processors.audio.run_transcription
        or processors.video.run_transcription
        or processors.video.extract_audio_track
    )


__all__ = [
    "OptionalDependencyReport",
    "validate_optional_dependencies",
    "image_processing_requires_pillow",
    "audio_video_processing_enabled",
    "audio_video_processing_requires_decoder",
]
