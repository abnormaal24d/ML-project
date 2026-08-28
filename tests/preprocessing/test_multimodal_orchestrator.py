"""MultimodalPreprocessor orchestrates concrete pipelines without registries."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from preprocessing.media.base_media_preprocessor import (
    MediaPreprocessingResult,
)
from preprocessing.multimodal_preprocessor import (
    MultimodalPreprocessor,
)
from preprocessing.preprocessing_input import (
    LanguageEvidence,
    PreprocessingInput,
)
from preprocessing.preprocessing_result import (
    PreprocessingQuarantineRecord,
    PreprocessingResult,
)
from tests.support.logging import TEST_LOGGER


def _input(*, source_id: str, modality: str) -> PreprocessingInput:
    return PreprocessingInput(
        source_id=source_id,
        source_url=f"https://example.test/{source_id}",
        normalized_url=f"https://example.test/{source_id}",
        domain="example.test",
        path=f"/{source_id}",
        language_evidence=LanguageEvidence(language="en"),
        title=source_id,
        modality=modality,
    )


@pytest.mark.asyncio
async def test_process_runs_modalities_and_merges_results() -> None:
    text = MagicMock()
    text.process.return_value = PreprocessingResult(
        documents=(),
        skipped_sources={"t1": "duplicate"},
        quarantine_records=(),
        diagnostics={"exact_duplicates_dropped": 1},
    )
    image = MagicMock()
    image.process.return_value = MediaPreprocessingResult(
        items=(SimpleNamespace(media_id="img-1"),),
        quarantine_records=(),
    )
    audio = MagicMock()
    audio.process.return_value = MediaPreprocessingResult(
        items=(),
        quarantine_records=(),
    )
    video = MagicMock()
    quarantine = PreprocessingQuarantineRecord(
        source_id="v1",
        reason="safety_reject",
        modality="video",
    )
    video.process.return_value = MediaPreprocessingResult(
        items=(),
        quarantine_records=(quarantine,),
    )

    orchestrator = MultimodalPreprocessor(
        text_preprocessor=text,
        image_preprocessor=image,
        audio_preprocessor=audio,
        video_preprocessor=video,
        logger=TEST_LOGGER,
    )
    result = await orchestrator.process(
        inputs=(
            _input(source_id="t1", modality="text"),
            _input(source_id="d1", modality="document"),
            _input(source_id="i1", modality="image"),
            _input(source_id="a1", modality="audio"),
            _input(source_id="v1", modality="video"),
            _input(source_id="x1", modality="unknown"),
        )
    )

    assert result.skipped_sources["t1"] == "duplicate"
    assert result.skipped_sources["v1"] == "safety_reject"
    assert result.skipped_sources["x1"] == "unsupported_modality:unknown"
    assert len(result.images) == 1
    assert result.diagnostics["input_count_by_modality"] == {
        "text": 1,
        "document": 1,
        "image": 1,
        "audio": 1,
        "video": 1,
    }
    assert result.diagnostics["output_count_by_type"] == {
        "document": 0,
        "image": 1,
        "audio": 0,
        "video": 0,
    }
    assert result.diagnostics["exact_duplicates_dropped"] == 1

    text.process.assert_called_once()
    text_inputs = text.process.call_args.kwargs["inputs"]
    assert {item.source_id for item in text_inputs} == {"t1", "d1"}
    image.process.assert_called_once()
    audio.process.assert_called_once()
    video.process.assert_called_once()
