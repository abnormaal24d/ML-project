"""Audio analysis enrichment: models, analyzer, and assembler.

Objective stream metadata comes from ``AudioPayloadExtractor``. Transcription,
diarization, event detection, and emotion analysis remain enrichment concerns.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.collection.processors import AudioProcessorSettings
from crawler.extraction.payloads.audio_payload_extractor import (
    AudioPayloadExtractor,
)
from crawler.fetching.results.result import FetchResult
from logger.project_logger import ProjectLogger
from preprocessing.media.audio.audio_emotion_analyzer import (
    AudioEmotionAnalyzer,
)
from preprocessing.media.audio.audio_event_analyzer import (
    AudioEventAnalyzer,
)

if TYPE_CHECKING:
    from crawler.analysis.enrichment.media_files.media_payload_path_resolver import (
        MediaPayloadPathResolver,
    )
    from preprocessing.media.audio.audio_emotion_analyzer import (
        AudioEmotionResult,
    )
    from preprocessing.media.speech.speaker_diarization_result import (
        SpeakerDiarizationResult,
    )
    from preprocessing.media.speech.speaker_diarizer import (
        SpeakerDiarizer,
    )
    from preprocessing.media.speech.speech_transcriber import (
        SpeechTranscriber,
    )
    from preprocessing.media.speech.transcription_result import (
        TranscriptionResult,
    )


@dataclass(frozen=True, slots=True)
class AudioAnalysisResult:
    """Raw analysis output from AudioAnalyzer per final spec."""

    payload_path: str | None
    metadata: dict[str, object]
    transcript_text: str | None = None
    transcript_confidence: float | None = None
    transcript_source: str | None = None
    transcript_language: str | None = None
    transcript_segments: tuple[dict[str, object], ...] = ()
    transcription_status: str = "not_run"
    transcription_provenance: dict[str, object] | None = None
    language: str | None = None
    event_segments: tuple[dict[str, object], ...] = ()
    speaker_segments: tuple[dict[str, object], ...] = ()
    analysis_status: str = "completed"
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bitrate: int | None = None
    metadata_status: str = "ok"
    speaker_count: int = 0
    overlapping_speech: bool = False
    speaker_diarization_status: str | None = None
    speaker_diarization_model_name: str | None = None
    speaker_diarization_model_version: str | None = None
    background_noise_label: str | None = None
    acoustic_scene_label: str | None = None
    sound_label: str | None = None
    sound_events: tuple[dict[str, Any], ...] = ()
    audio_event_confidence: float | None = None
    audio_event_analysis_status: str = "not_run"
    audio_event_analysis_reasons: tuple[str, ...] = ()
    audio_event_model_name: str | None = None
    prosody: dict[str, Any] | None = None
    emotion_label: str | None = None
    emotion_confidence: float | None = None
    arousal: float | None = None
    valence: float | None = None
    dominance: float | None = None
    audio_emotion_analysis_status: str = "not_run"
    audio_emotion_analysis_reasons: tuple[str, ...] = ()
    audio_emotion_model_name: str | None = None


class AudioAnalyzer:
    """Orchestrate bounded audio analysis.

    Coordinates objective payload extraction, transcription, speaker
    diarization, audio-event classification and emotion analysis.
    """

    def __init__(
        self,
        *,
        settings: AudioProcessorSettings,
        media_file_resolver: MediaPayloadPathResolver,
        payload_extractor: AudioPayloadExtractor,
        diarization_service: SpeakerDiarizer,
        transcription_executor: SpeechTranscriber,
        event_analyzer: AudioEventAnalyzer,
        emotion_analyzer: AudioEmotionAnalyzer,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._media_file_resolver = media_file_resolver
        self._payload_extractor = payload_extractor
        self._transcription_executor = transcription_executor
        self._logger = logger
        self._diarization_service = diarization_service
        self._event_analyzer = event_analyzer
        self._emotion_analyzer = emotion_analyzer

    async def analyze(
        self,
        *,
        result: FetchResult,
    ) -> AudioAnalysisResult:
        """Analyze a fetched audio payload."""

        return await self._analyze_inner(
            result=result,
            run_transcription=self._settings.run_transcription,
            transcription_language=self._settings.transcription_language,
            max_duration_seconds=self._settings.max_duration_seconds,
        )

    async def _analyze_inner(
        self,
        *,
        result: FetchResult,
        run_transcription: bool,
        transcription_language: str | None,
        max_duration_seconds: float,
    ) -> AudioAnalysisResult:
        path = await self._media_file_resolver.resolve_path(
            result=result,
            suffix=".audio",
        )

        try:
            metadata = await asyncio.to_thread(
                self._read_payload_metadata,
                path=path,
            )

            transcription = await self._run_transcription(
                path=path,
                run_transcription=run_transcription,
                metadata=metadata,
                max_duration_seconds=max_duration_seconds,
                language=transcription_language,
            )

            analysis_bytes = await self._read_audio_analysis_bytes(path=path)
            sample_rate = _optional_int(metadata.get("sample_rate"))

            speaker_diarization = await self._run_diarization(
                analysis_bytes=analysis_bytes,
                path=path,
                metadata=metadata,
                transcription=transcription,
            )

            audio_events = await self._run_event_detection(
                analysis_bytes=analysis_bytes,
                sample_rate=sample_rate,
            )

            audio_emotion = await self._run_emotion_analysis(
                analysis_bytes=analysis_bytes,
                sample_rate=sample_rate,
                transcription=transcription,
            )

            return self._build_analysis_result(
                path=path,
                metadata=metadata,
                transcription=transcription,
                speaker_diarization=speaker_diarization,
                audio_events=audio_events,
                audio_emotion=audio_emotion,
            )
        finally:
            self._media_file_resolver.cleanup_owned_path(path)

    @staticmethod
    def _build_analysis_result(
        *,
        path: Path,
        metadata: dict[str, object],
        transcription: TranscriptionResult | None,
        speaker_diarization: SpeakerDiarizationResult,
        audio_events: dict[str, Any],
        audio_emotion: AudioEmotionResult,
    ) -> AudioAnalysisResult:
        duration_seconds = _optional_float(metadata.get("duration_seconds"))
        sample_rate = _optional_int(metadata.get("sample_rate"))
        channels = _optional_int(metadata.get("channels"))
        bitrate = _optional_int(metadata.get("bitrate"))
        metadata_status = (
            _optional_text(metadata.get("status")) or "missing_or_unreadable"
        )

        speaker_segments = _speaker_segments_payload(
            speaker_diarization=speaker_diarization
        )
        event_payload = dict(audio_events or {})
        emotion_payload = _audio_emotion_payload(audio_emotion=audio_emotion)

        transcript_segments = (
            tuple(asdict(segment) for segment in transcription.segments)
            if transcription is not None
            else ()
        )

        return AudioAnalysisResult(
            payload_path=str(path),
            metadata={
                "duration_seconds": duration_seconds,
                "sample_rate": sample_rate,
            },
            transcript_text=(
                transcription.text if transcription is not None else None
            ),
            transcript_confidence=(
                transcription.confidence if transcription is not None else None
            ),
            transcript_source=(
                transcription.provenance.producer_name
                if transcription is not None
                else None
            ),
            transcript_language=(
                transcription.language if transcription is not None else None
            ),
            transcript_segments=transcript_segments,
            transcription_status=(
                transcription.transcription_status
                if transcription is not None
                else "not_run"
            ),
            transcription_provenance=(
                transcription.provenance.to_dict()
                if transcription is not None
                else None
            ),
            language=(
                transcription.language if transcription is not None else None
            ),
            event_segments=_dict_tuple(event_payload.get("sound_events")),
            speaker_segments=speaker_segments,
            analysis_status="completed",
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            bitrate=bitrate,
            metadata_status=metadata_status,
            speaker_count=speaker_diarization.speaker_count,
            overlapping_speech=speaker_diarization.overlapping_speech,
            speaker_diarization_status=_speaker_diarization_status(
                speaker_diarization=speaker_diarization,
                speaker_segments=speaker_segments,
            ),
            speaker_diarization_model_name=speaker_diarization.model_name,
            speaker_diarization_model_version=speaker_diarization.model_version,
            background_noise_label=_optional_text(
                event_payload.get("background_noise_label")
            ),
            acoustic_scene_label=_optional_text(
                event_payload.get("acoustic_scene_label")
            ),
            sound_label=_sound_label(audio_event_payload=event_payload),
            sound_events=_dict_tuple(event_payload.get("sound_events")),
            audio_event_confidence=_optional_float(
                event_payload.get("confidence")
            ),
            audio_event_analysis_status=str(
                event_payload.get("analysis_status") or "not_run"
            ),
            audio_event_analysis_reasons=_string_tuple(
                event_payload.get("analysis_reasons")
            ),
            audio_event_model_name=_optional_text(
                event_payload.get("model_name")
            ),
            prosody=emotion_payload["prosody"],
            emotion_label=emotion_payload["emotion_label"],
            emotion_confidence=emotion_payload["emotion_confidence"],
            arousal=emotion_payload["arousal"],
            valence=emotion_payload["valence"],
            dominance=emotion_payload["dominance"],
            audio_emotion_analysis_status=emotion_payload["analysis_status"],
            audio_emotion_analysis_reasons=emotion_payload["analysis_reasons"],
            audio_emotion_model_name=emotion_payload["model_name"],
        )

    def _read_payload_metadata(self, *, path: Path) -> dict[str, object]:
        """Read objective stream metadata via AudioPayloadExtractor."""

        try:
            body = path.read_bytes()
        except OSError:
            return {
                "duration_seconds": None,
                "sample_rate": None,
                "channels": None,
                "bitrate": None,
                "format": None,
                "byte_size": None,
                "sha256": None,
                "status": "missing_or_unreadable",
            }

        extracted = self._payload_extractor.extract(body=body)
        if extracted is None:
            return {
                "duration_seconds": None,
                "sample_rate": None,
                "channels": None,
                "bitrate": None,
                "format": None,
                "byte_size": len(body),
                "sha256": None,
                "status": "missing_or_unreadable",
            }

        return {
            "duration_seconds": extracted.duration_seconds,
            "sample_rate": extracted.sample_rate,
            "channels": extracted.channels,
            "bitrate": extracted.bitrate,
            "format": extracted.format,
            "byte_size": extracted.byte_size,
            "sha256": extracted.sha256,
            "status": "extracted",
        }

    async def _run_transcription(
        self,
        *,
        path: Path,
        run_transcription: bool,
        metadata: dict[str, object],
        max_duration_seconds: float,
        language: str | None,
    ) -> TranscriptionResult | None:
        if not run_transcription:
            return None

        if not self._is_small_enough(path):
            self._logger.debug(
                "audio_transcription_skipped",
                reason="payload_exceeds_transcription_limit",
            )
            return None

        self._logger.debug(
            "audio_transcription_start",
            path=str(path),
        )

        transcription = (
            await self._transcription_executor.transcribe_if_allowed(
                media_path=path,
                run_transcription=True,
                duration_seconds=_optional_float(
                    metadata.get("duration_seconds")
                ),
                max_duration_seconds=max_duration_seconds,
                language=language,
            )
        )

        self._logger.debug(
            "audio_transcription_complete",
            has_transcription=transcription is not None,
        )

        return transcription

    async def _run_diarization(
        self,
        *,
        analysis_bytes: bytes,
        path: Path,
        metadata: dict[str, object],
        transcription: TranscriptionResult | None,
    ) -> SpeakerDiarizationResult:
        self._logger.debug(
            "audio_diarization_start",
            path=str(path),
        )

        result = await asyncio.to_thread(
            self._diarization_service.diarize,
            audio_bytes=analysis_bytes,
            audio_path=path,
            sample_rate=_optional_int(metadata.get("sample_rate")),
            transcript_segments=(
                None
                if transcription is None
                else [asdict(segment) for segment in transcription.segments]
            ),
        )

        self._logger.debug(
            "audio_diarization_complete",
            speaker_count=result.speaker_count,
        )

        return result

    async def _run_event_detection(
        self,
        *,
        analysis_bytes: bytes,
        sample_rate: int | None,
    ) -> dict[str, Any]:
        self._logger.debug("audio_event_detection_start")

        result = await asyncio.to_thread(
            self._event_analyzer.analyze,
            audio_bytes=analysis_bytes,
            sample_rate=sample_rate,
        )

        self._logger.debug("audio_event_detection_complete")

        return result

    async def _run_emotion_analysis(
        self,
        *,
        analysis_bytes: bytes,
        sample_rate: int | None,
        transcription: TranscriptionResult | None,
    ) -> AudioEmotionResult:
        self._logger.debug("audio_emotion_analysis_start")

        result = await asyncio.to_thread(
            self._emotion_analyzer.analyze,
            audio_bytes=analysis_bytes,
            sample_rate=sample_rate,
            transcript=(None if transcription is None else transcription.text),
        )

        self._logger.debug("audio_emotion_analysis_complete")

        return result

    def _is_small_enough(self, path: Path) -> bool:
        try:
            return bool(
                path.stat().st_size <= self._settings.max_transcription_bytes
            )
        except OSError:
            return False

    async def _read_audio_analysis_bytes(
        self,
        *,
        path: Path,
    ) -> bytes:
        if not self._is_small_enough(path):
            return b""

        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError:
            return b""


def _speaker_segments_payload(
    *,
    speaker_diarization: SpeakerDiarizationResult | None,
) -> tuple[dict[str, Any], ...]:
    if speaker_diarization is None:
        return ()

    return tuple(
        {
            "speaker_id": segment.speaker_id,
            "start_seconds": segment.start_seconds,
            "end_seconds": segment.end_seconds,
            "confidence": segment.confidence,
            "overlapping_speech": segment.overlapping_speech,
        }
        for segment in speaker_diarization.segments
    )


def _speaker_diarization_status(
    *,
    speaker_diarization: SpeakerDiarizationResult | None,
    speaker_segments: tuple[dict[str, Any], ...],
) -> str:
    if speaker_diarization is None:
        return "not_run"

    if speaker_segments:
        return "passed"

    return "unavailable"


def _audio_emotion_payload(
    *,
    audio_emotion: AudioEmotionResult | None,
) -> dict[str, Any]:
    if audio_emotion is None:
        return {
            "emotion_label": None,
            "emotion_confidence": None,
            "arousal": None,
            "valence": None,
            "dominance": None,
            "prosody": None,
            "model_name": None,
            "analysis_status": "not_run",
            "analysis_reasons": (),
        }

    prosody = (
        None
        if audio_emotion.prosody is None
        else _drop_none_values(asdict(audio_emotion.prosody))
    )

    return {
        "emotion_label": audio_emotion.emotion_label,
        "emotion_confidence": audio_emotion.emotion_confidence,
        "arousal": audio_emotion.arousal,
        "valence": audio_emotion.valence,
        "dominance": audio_emotion.dominance,
        "prosody": prosody,
        "model_name": audio_emotion.model_name,
        "analysis_status": audio_emotion.analysis_status,
        "analysis_reasons": audio_emotion.analysis_reasons,
    }


def _sound_label(
    *,
    audio_event_payload: dict[str, Any],
) -> str | None:
    for event in _dict_tuple(audio_event_payload.get("sound_events")):
        label = _optional_text(event.get("label"))

        if label and label != "silence":
            return label

    return _optional_text(audio_event_payload.get("background_noise_label"))


def _dict_tuple(
    value: object,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()

    return tuple(dict(item) for item in value if isinstance(item, dict))


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()

    return tuple(str(item) for item in value if str(item).strip())


def _drop_none_values(
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _optional_text(value: object) -> str | None:
    if value is None:
        return None

    text = " ".join(str(value).split())
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (float, int)):
        return float(value)

    try:
        return float(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not value.is_integer():
            return None
        return int(value)

    try:
        return int(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return None
