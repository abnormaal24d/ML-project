"""Speaker diarization service with honest backend reporting.

The service only emits diarization segments when speaker timing evidence is
available from upstream transcript segments or from an explicitly configured
backend. Without such evidence it returns an empty, unavailable result instead
of fabricating speakers or confidence scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from preprocessing.media.adapters.pyannote_adapter import (
    load_pyannote_backend,
)
from preprocessing.media.speech.speaker_diarization_result import (
    SpeakerDiarizationResult,
    _validate_segments,
)
from preprocessing.provenance import (
    ProducerProvenance,
    ProducerType,
    hash_bytes,
    hash_file,
    hash_parameters,
    hash_text,
)

if TYPE_CHECKING:
    from config.preprocessing.media_settings import DiarizationSettings

_NO_BACKEND_VERSION = "no-diarization-backend-v1"
_TRANSCRIPT_HINT_VERSION = "transcript-speaker-hints-v1"


class SpeakerDiarizer:
    """Run speaker diarization only when real speaker evidence is available."""

    def __init__(
        self,
        *,
        settings: DiarizationSettings,
        backend: Any | None = None,
    ) -> None:
        self.enabled = settings.enabled
        self.backend = settings.backend
        self.model_name = settings.model_name
        self.device = settings.device
        self._backend_impl = backend
        self._settings = settings

    def diarize(
        self,
        *,
        audio_bytes: bytes | None = None,
        audio_path: str | Path | None = None,
        sample_rate: int | None = None,
        transcript_segments: list[dict[str, Any]] | None = None,
        max_speakers: int | None = None,
    ) -> SpeakerDiarizationResult:
        """Return diarization from real speaker hints or a configured backend.

        Transcript segments are accepted only when they already carry speaker
        identifiers. Plain ASR timing segments are not converted into a guessed
        single-speaker result.
        """

        if not self.enabled or self.backend == "disabled":
            return SpeakerDiarizationResult(
                segments=(),
                speaker_count=0,
                overlapping_speech=False,
                model_name="unavailable",
                model_version=_NO_BACKEND_VERSION,
            )

        hint_segments = _segments_from_transcript_hints(transcript_segments)
        if hint_segments:
            return _result(
                raw_segments=hint_segments,
                model_name="transcript-speaker-hints",
                model_version=_TRANSCRIPT_HINT_VERSION,
                provenance=_hint_provenance(hint_segments),
            )

        if self._backend_impl is not None and (audio_bytes or audio_path):
            backend_segments = _run_backend(
                backend=self._backend_impl,
                audio_bytes=audio_bytes,
                audio_path=audio_path,
                sample_rate=sample_rate,
                max_speakers=max_speakers,
            )
            if backend_segments:
                provenance = _backend_provenance(
                    settings=self._settings,
                    backend=self._backend_impl,
                    audio_bytes=audio_bytes,
                    audio_path=audio_path,
                    raw_segments=backend_segments,
                )
                return _result(
                    raw_segments=backend_segments,
                    model_name=self.model_name,
                    model_version=_backend_version(self._backend_impl),
                    provenance=provenance,
                )

        return SpeakerDiarizationResult(
            segments=(),
            speaker_count=0,
            overlapping_speech=False,
            model_name="unavailable",
            model_version=_NO_BACKEND_VERSION,
        )


def get_diarization_service(
    settings: DiarizationSettings,
) -> SpeakerDiarizer:
    """Build the configured preprocessing-owned diarization service."""

    configured_backend = str(settings.backend)

    if not settings.enabled or configured_backend == "disabled":
        return SpeakerDiarizer(settings=settings)

    if configured_backend == "pyannote":
        backend = load_pyannote_backend(
            settings=settings,
            device=settings.device,
        )
        return SpeakerDiarizer(
            settings=settings,
            backend=backend,
        )

    if configured_backend == "transcript_hints":
        return SpeakerDiarizer(settings=settings)

    raise RuntimeError(
        f"unsupported diarization backend: {configured_backend!r}"
    )


def _segments_from_transcript_hints(
    transcript_segments: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not transcript_segments:
        return []

    raw_segments: list[dict[str, Any]] = []
    current_speaker: str | None = None
    for segment in transcript_segments:
        speaker = segment.get("speaker_id")
        if speaker:
            current_speaker = str(speaker)
        if current_speaker is None:
            continue

        start = _float(segment.get("start_seconds"))
        end = _float(segment.get("end_seconds"))
        if start is None:
            continue
        if end is None:
            end = start + 0.5
        if end <= start:
            continue

        confidence = segment.get("confidence")
        raw_segments.append(
            {
                "speaker_id": current_speaker,
                "start_seconds": start,
                "end_seconds": end,
                "confidence": _float(confidence),
                "overlapping_speech": bool(
                    segment.get("overlapping_speech", False)
                ),
            }
        )
    return raw_segments


def _run_backend(
    *,
    backend: Any,
    audio_bytes: bytes | None,
    audio_path: str | Path | None,
    sample_rate: int | None,
    max_speakers: int | None,
) -> list[dict[str, Any]]:
    try:
        if hasattr(backend, "diarize"):
            result = backend.diarize(
                audio_bytes=audio_bytes,
                audio_path=None if audio_path is None else str(audio_path),
                sample_rate=sample_rate,
                max_speakers=max_speakers,
            )
        else:
            target = str(audio_path) if audio_path is not None else audio_bytes
            kwargs: dict[str, Any] = {}
            if max_speakers is not None:
                kwargs["num_speakers"] = max_speakers
            result = backend(target, **kwargs)
    except (OSError, RuntimeError, TypeError, ValueError):
        return []

    return _coerce_backend_segments(result)


def _coerce_backend_segments(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    if isinstance(result, dict):
        raw = result.get("segments") or []
    elif isinstance(result, (list, tuple)):
        raw = result
    elif hasattr(result, "itertracks"):
        raw = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            raw.append(
                {
                    "speaker_id": str(speaker),
                    "start_seconds": float(turn.start),
                    "end_seconds": float(turn.end),
                    "confidence": None,
                    "overlapping_speech": False,
                }
            )
    else:
        raw = []

    segments: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker_id")
        start = _float(item.get("start_seconds"))
        end = _float(item.get("end_seconds"))
        if not speaker or start is None or end is None or end <= start:
            continue
        segments.append(
            {
                "speaker_id": str(speaker),
                "start_seconds": start,
                "end_seconds": end,
                "confidence": _float(item.get("confidence")),
                "overlapping_speech": bool(
                    item.get("overlapping_speech", False)
                ),
                "embedding": item.get("embedding"),
            }
        )
    return segments


def _result(
    *,
    raw_segments: list[dict[str, Any]],
    model_name: str,
    model_version: str,
    provenance: ProducerProvenance | None = None,
) -> SpeakerDiarizationResult:
    segments = _validate_segments(raw_segments)
    speakers = {segment.speaker_id for segment in segments}
    return SpeakerDiarizationResult(
        segments=tuple(segments),
        speaker_count=len(speakers),
        overlapping_speech=any(
            segment.overlapping_speech for segment in segments
        ),
        model_name=model_name,
        model_version=model_version,
        provenance=provenance,
    )


def _backend_provenance(
    *,
    settings: DiarizationSettings,
    backend: Any,
    audio_bytes: bytes | None,
    audio_path: str | Path | None,
    raw_segments: list[dict[str, Any]],
) -> ProducerProvenance:
    if audio_bytes is not None:
        source_hash = hash_bytes(audio_bytes)
    elif audio_path is not None and Path(audio_path).is_file():
        source_hash = hash_file(Path(audio_path))
    else:
        source_hash = hash_text(str(audio_path or "unknown-audio"))
    warnings = tuple(
        warning
        for value, warning in (
            (
                settings.model_revision,
                "unpinned_model_revision",
            ),
            (
                settings.model_artifact_hash,
                "unverified_model_artifact",
            ),
        )
        if value is None
    )
    return ProducerProvenance(
        producer_type=ProducerType.EXTERNAL_MODEL,
        producer_name="pyannote.audio",
        producer_version=_backend_version(backend),
        model_id=settings.model_name,
        model_revision=settings.model_revision,
        artifact_hash=settings.model_artifact_hash,
        parameters_hash=hash_parameters(
            {
                "backend": "pyannote",
                "device": settings.device,
            }
        ),
        source_hash=source_hash,
        output_hash=hash_parameters(
            {"segments": _segment_hash_payload(raw_segments)}
        ),
        confidence=_average_segment_confidence(raw_segments),
        warnings=warnings,
    )


def _hint_provenance(
    raw_segments: list[dict[str, Any]],
) -> ProducerProvenance:
    payload_hash = hash_parameters(
        {"segments": _segment_hash_payload(raw_segments)}
    )
    return ProducerProvenance(
        producer_type=ProducerType.SOURCE,
        producer_name="transcript-speaker-hints",
        producer_version=_TRANSCRIPT_HINT_VERSION,
        parameters_hash=hash_parameters({"strategy": "transcript_hints"}),
        source_hash=payload_hash,
        output_hash=payload_hash,
        confidence=_average_segment_confidence(raw_segments),
    )


def _average_segment_confidence(
    segments: list[dict[str, Any]],
) -> float | None:
    values = [
        float(segment["confidence"])
        for segment in segments
        if segment.get("confidence") is not None
    ]
    if not values:
        return None
    return max(0.0, min(1.0, sum(values) / len(values)))


def _segment_hash_payload(
    segments: list[dict[str, Any]],
) -> list[dict[str, object]]:
    return [
        {
            "speaker_id": str(segment.get("speaker_id", "")),
            "start_seconds": _float(segment.get("start_seconds")),
            "end_seconds": _float(segment.get("end_seconds")),
            "confidence": _float(segment.get("confidence")),
            "overlapping_speech": bool(
                segment.get("overlapping_speech", False)
            ),
        }
        for segment in segments
    ]


def _backend_version(backend: Any) -> str:
    version = getattr(backend, "version", None) or getattr(
        backend, "model_version", None
    )
    return str(version) if version else "external-diarization-backend-v1"


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
