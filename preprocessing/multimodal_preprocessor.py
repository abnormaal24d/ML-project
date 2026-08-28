"""Concrete multimodal preprocessing orchestrator.

Keeps four fixed pipelines as explicit fields. Processors remain synchronous;
only this orchestrator is async and runs modalities in parallel via to_thread.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from logger.project_logger import ProjectLogger
from preprocessing.media.audio.audio_preprocessor import AudioPreprocessor
from preprocessing.media.image.image_preprocessor import ImagePreprocessor
from preprocessing.media.video.video_preprocessor import VideoPreprocessor
from preprocessing.preprocessing_input import PreprocessingInput
from preprocessing.preprocessing_result import (
    PreprocessingQuarantineRecord,
    PreprocessingResult,
)
from preprocessing.text.text_preprocessor import TextPreprocessor


class MultimodalPreprocessor:
    """Orchestrate the fixed text, image, audio, and video pipelines."""

    def __init__(
        self,
        *,
        text_preprocessor: TextPreprocessor,
        image_preprocessor: ImagePreprocessor,
        audio_preprocessor: AudioPreprocessor,
        video_preprocessor: VideoPreprocessor,
        logger: ProjectLogger,
    ) -> None:
        self._text_preprocessor = text_preprocessor
        self._image_preprocessor = image_preprocessor
        self._audio_preprocessor = audio_preprocessor
        self._video_preprocessor = video_preprocessor
        self._logger = logger
        self._logger.debug("multimodal_preprocessor_initialized")

    async def process(
        self,
        *,
        inputs: Iterable[PreprocessingInput],
    ) -> PreprocessingResult:
        """Group by modality and run the four pipelines in parallel."""

        groups, skipped, quarantine = self._group_inputs(inputs)

        (
            text_result,
            image_result,
            audio_result,
            video_result,
        ) = await asyncio.gather(
            asyncio.to_thread(
                self._text_preprocessor.process,
                inputs=tuple(groups["text"] + groups["document"]),
            ),
            asyncio.to_thread(
                self._image_preprocessor.process,
                inputs=tuple(groups["image"]),
            ),
            asyncio.to_thread(
                self._audio_preprocessor.process,
                inputs=tuple(groups["audio"]),
            ),
            asyncio.to_thread(
                self._video_preprocessor.process,
                inputs=tuple(groups["video"]),
            ),
        )

        skipped.update(text_result.skipped_sources)

        media_quarantine = (
            image_result.quarantine_records
            + audio_result.quarantine_records
            + video_result.quarantine_records
        )
        for record in media_quarantine:
            skipped[record.source_id] = record.reason

        return PreprocessingResult(
            documents=text_result.documents,
            images=image_result.items,
            audio=audio_result.items,
            video=video_result.items,
            skipped_sources=skipped,
            quarantine_records=(
                quarantine + text_result.quarantine_records + media_quarantine
            ),
            diagnostics={
                **text_result.diagnostics,
                "input_count_by_modality": {
                    name: len(group) for name, group in groups.items()
                },
                "output_count_by_type": {
                    "document": len(text_result.documents),
                    "image": len(image_result.items),
                    "audio": len(audio_result.items),
                    "video": len(video_result.items),
                },
            },
        )

    @staticmethod
    def _group_inputs(
        inputs: Iterable[PreprocessingInput],
    ) -> tuple[
        dict[str, list[PreprocessingInput]],
        dict[str, str],
        tuple[PreprocessingQuarantineRecord, ...],
    ]:
        groups: dict[str, list[PreprocessingInput]] = {
            "text": [],
            "document": [],
            "image": [],
            "audio": [],
            "video": [],
        }
        skipped: dict[str, str] = {}
        quarantine: list[PreprocessingQuarantineRecord] = []

        for item in inputs:
            group = groups.get(item.modality)
            if group is not None:
                group.append(item)
                continue

            reason = f"unsupported_modality:{item.modality}"
            skipped[item.source_id] = reason
            quarantine.append(
                PreprocessingQuarantineRecord.from_input(
                    item=item,
                    reason=reason,
                )
            )

        return groups, skipped, tuple(quarantine)
