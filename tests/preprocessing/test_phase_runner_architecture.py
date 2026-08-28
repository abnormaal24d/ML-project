"""Architectural tests for MultimodalPreprocessor boundaries."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from preprocessing.media.base_media_preprocessor import (
    MediaPreprocessingResult,
)
from preprocessing.multimodal_preprocessor import MultimodalPreprocessor
from preprocessing.preprocessing_input import (
    LanguageEvidence,
    PreprocessingInput,
)
from preprocessing.preprocessing_result import PreprocessingResult
from tests.support.logging import TEST_LOGGER


def _input(*, source_id: str, modality: str = "text") -> PreprocessingInput:
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


def _empty_text_result() -> PreprocessingResult:
    return PreprocessingResult(documents=(), diagnostics={})


def _empty_media_result() -> MediaPreprocessingResult[object]:
    return MediaPreprocessingResult(items=(), quarantine_records=())


def _build_runner(
    *,
    text: MagicMock | None = None,
    image: MagicMock | None = None,
    audio: MagicMock | None = None,
    video: MagicMock | None = None,
) -> tuple[MultimodalPreprocessor, MagicMock, MagicMock, MagicMock, MagicMock]:
    text_pp = text or MagicMock()
    image_pp = image or MagicMock()
    audio_pp = audio or MagicMock()
    video_pp = video or MagicMock()
    if text is None:
        text_pp.process.return_value = _empty_text_result()
    if image is None:
        image_pp.process.return_value = _empty_media_result()
    if audio is None:
        audio_pp.process.return_value = _empty_media_result()
    if video is None:
        video_pp.process.return_value = _empty_media_result()
    workflow = MultimodalPreprocessor(
        text_preprocessor=text_pp,
        image_preprocessor=image_pp,
        audio_preprocessor=audio_pp,
        video_preprocessor=video_pp,
        logger=TEST_LOGGER,
    )
    return (
        workflow,
        text_pp,
        image_pp,
        audio_pp,
        video_pp,
    )


@pytest.mark.asyncio
async def test_phase_runner_routes_documents_to_text_pipeline_and_media_separately() -> (
    None
):
    """Runtime boundary: phase runner drives orchestrator; outputs stay typed."""

    text_pp = MagicMock()
    text_pp.process.return_value = PreprocessingResult(
        documents=(SimpleNamespace(document_id="doc-1"),),
        diagnostics={},
    )
    image_pp = MagicMock()
    image_pp.process.return_value = MediaPreprocessingResult(
        items=(SimpleNamespace(media_id="img-1"),),
        quarantine_records=(),
    )
    audio_pp = MagicMock()
    audio_pp.process.return_value = MediaPreprocessingResult(
        items=(SimpleNamespace(media_id="aud-1"),),
        quarantine_records=(),
    )
    video_pp = MagicMock()
    video_pp.process.return_value = MediaPreprocessingResult(
        items=(SimpleNamespace(media_id="vid-1"),),
        quarantine_records=(),
    )
    runner, _, _, _, _ = _build_runner(
        text=text_pp,
        image=image_pp,
        audio=audio_pp,
        video=video_pp,
    )

    result = await runner.process(
        inputs=(
            _input(source_id="t1", modality="text"),
            _input(source_id="i1", modality="image"),
            _input(source_id="a1", modality="audio"),
            _input(source_id="v1", modality="video"),
        )
    )

    assert len(result.documents) == 1
    assert len(result.images) == 1
    assert len(result.audio) == 1
    assert len(result.video) == 1
    text_pp.process.assert_called_once()
    image_pp.process.assert_called_once()
    audio_pp.process.assert_called_once()
    video_pp.process.assert_called_once()


@pytest.mark.asyncio
async def test_pipelines_start_concurrently_via_barrier() -> None:
    """Concurrency without timing: all four pipelines reach a shared barrier."""

    barrier = threading.Barrier(parties=4, timeout=5.0)
    started: list[str] = []
    lock = threading.Lock()

    def _text_process(*, inputs):  # noqa: ANN001, ARG001
        with lock:
            started.append("text")
        barrier.wait()
        return _empty_text_result()

    def _media_process(name: str):
        def _run(*, inputs):  # noqa: ANN001, ARG001
            with lock:
                started.append(name)
            barrier.wait()
            return _empty_media_result()

        return _run

    text_pp = MagicMock()
    text_pp.process.side_effect = _text_process
    image_pp = MagicMock()
    image_pp.process.side_effect = _media_process("image")
    audio_pp = MagicMock()
    audio_pp.process.side_effect = _media_process("audio")
    video_pp = MagicMock()
    video_pp.process.side_effect = _media_process("video")
    runner, _, _, _, _ = _build_runner(
        text=text_pp,
        image=image_pp,
        audio=audio_pp,
        video=video_pp,
    )

    await runner.process(
        inputs=(
            _input(source_id="t1", modality="text"),
            _input(source_id="i1", modality="image"),
            _input(source_id="a1", modality="audio"),
            _input(source_id="v1", modality="video"),
        )
    )
    assert set(started) == {"text", "image", "audio", "video"}


@pytest.mark.asyncio
async def test_systemic_pipeline_exception_propagates() -> None:
    """Infrastructure failures must not become partial silent results."""

    text_pp = MagicMock()
    text_pp.process.return_value = _empty_text_result()
    image_pp = MagicMock()
    image_pp.process.side_effect = RuntimeError("image backend unavailable")
    runner, _, _, _, _ = _build_runner(text=text_pp, image=image_pp)

    with pytest.raises(RuntimeError, match="image backend unavailable"):
        await runner.process(
            inputs=(
                _input(source_id="t1", modality="text"),
                _input(source_id="i1", modality="image"),
            )
        )


@pytest.mark.asyncio
async def test_orchestrator_reentrancy_does_not_leak_results() -> None:
    """Concurrent runner invocations must keep result sets isolated."""

    call_count = 0
    lock = threading.Lock()

    def _text_process(*, inputs):  # noqa: ANN001
        nonlocal call_count
        with lock:
            call_count += 1
            index = call_count
        # Distinct documents per invocation.
        return PreprocessingResult(
            documents=(SimpleNamespace(document_id=f"doc-{index}"),),
            diagnostics={"call": index},
        )

    text_pp = MagicMock()
    text_pp.process.side_effect = _text_process
    runner, _, _, _, _ = _build_runner(text=text_pp)

    first, second = await asyncio.gather(
        runner.process(inputs=(_input(source_id="a"),)),
        runner.process(inputs=(_input(source_id="b"),)),
    )

    ids = {first.documents[0].document_id, second.documents[0].document_id}
    assert ids == {"doc-1", "doc-2"}
    assert first.diagnostics["call"] != second.diagnostics["call"]
