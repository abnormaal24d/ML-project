"""Retry-fallback wiring contract for the crawler processor graph."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from config.collection.processors import BaseProcessorSettings
from config.load import load_settings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.fetching.errors.exceptions import RetryableFetchError
from crawler.fetching.results.result import FetchResult
from crawler.processing.processors.persisting_processor import (
    PersistingProcessor,
)
from crawler.processing.processors.processor_failure_handler import (
    ProcessorFailureHandler,
)
from orchestration.composition.runtime import handlers as handlers_module
from orchestration.composition.runtime.handlers import build_task_processor
from orchestration.composition.runtime.handler_composition import (
    page as page_module,
    feed as feed_module,
    image as image_module,
    audio as audio_module,
    video as video_module,
    document as document_module,
    registry as registry_module,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class _NoopLogger:
    def debug(self, *_args: object, **_kwargs: object) -> None:
        pass

    def info(self, *_args: object, **_kwargs: object) -> None:
        pass

    def warning(self, *_args: object, **_kwargs: object) -> None:
        pass

    def error(self, *_args: object, **_kwargs: object) -> None:
        pass


class _NoopCoverageGate:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def outcome_for(self, *, task: CrawlTask) -> None:
        del task
        return None


class _RetryingFetchService:
    async def fetch(self, *, task: CrawlTask) -> FetchResult:
        del task
        raise _retryable_error()


class _RetryableStageProcessor(
    PersistingProcessor[BaseProcessorSettings, str]
):
    """Concrete processor that can fail in each routed processor stage."""

    def __init__(
        self,
        *,
        failure_handler: ProcessorFailureHandler,
        retry_stage: str,
    ) -> None:
        super().__init__(
            settings=BaseProcessorSettings(persist_raw=False),
            dataset_writer=SimpleNamespace(),  # type: ignore[arg-type]
            logger=_NoopLogger(),  # type: ignore[arg-type]
            failure_handler=failure_handler,
        )
        self._retry_stage = retry_stage

    async def prepare_analysis(
        self,
        *,
        result: FetchResult,
    ) -> str:
        del result
        if self._retry_stage == "analysis":
            raise _retryable_error()
        return "analysis"

    async def validate_result(
        self,
        *,
        result: FetchResult,
        analysis: str | None,
    ) -> tuple[bool, str | None, dict[str, object]]:
        del result, analysis
        if self._retry_stage == "validation":
            raise _retryable_error()
        return True, None, {}

    async def build_enrichment(
        self,
        *,
        result: FetchResult,
        analysis: str | None,
    ) -> dict[str, object]:
        del result, analysis
        if self._retry_stage == "persistence":
            raise _retryable_error()
        return {}


def _retryable_error() -> RetryableFetchError:
    return RetryableFetchError(
        "temporary transport failure",
        retry_class="transport",
        retry_error_kind="connection_reset",
    )


def _task() -> CrawlTask:
    return CrawlTask(
        url="https://example.test/input",
        source_name="test",
        task_id="task-1",
        kind=MediaKind.PAGE,
    )


def _result() -> FetchResult:
    return FetchResult(
        url="https://example.test/input",
        final_url="https://example.test/final",
        status_code=200,
        headers={},
        fetched_at="2026-08-26T00:00:00Z",
        content_type="text/html",
        mime_type="text/html",
        encoding="utf-8",
        language="en",
        kind=MediaKind.PAGE,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retry_stage", ("analysis", "validation", "persistence")
)
async def test_non_default_retry_fallback_is_shared_by_fetch_and_processors(
    monkeypatch: pytest.MonkeyPatch,
    retry_stage: str,
) -> None:
    retry_wait_seconds = 17.25
    settings = load_settings(
        "dev",
        project_root=PROJECT_ROOT / "tmp-processor-failure-handler-wiring",
        config_root=PROJECT_ROOT,
        environment="dev",
    )
    scheduling = settings.collection.scheduling.model_copy(
        update={"default_retry_wait_seconds": retry_wait_seconds},
    )
    settings = settings.model_copy(
        update={
            "collection": settings.collection.model_copy(
                update={"scheduling": scheduling},
            )
        }
    )
    constructed_processors: list[_RetryableStageProcessor] = []

    def build_processor(**kwargs: object) -> _RetryableStageProcessor:
        failure_handler = kwargs["failure_handler"]
        assert isinstance(failure_handler, ProcessorFailureHandler)
        processor = _RetryableStageProcessor(
            failure_handler=failure_handler,
            retry_stage=retry_stage,
        )
        constructed_processors.append(processor)
        return processor

    for module, attr in (
        (page_module, "build_page_handler"),
        (feed_module, "build_feed_handler"),
        (image_module, "build_image_handler"),
        (audio_module, "build_audio_handler"),
        (video_module, "build_video_handler"),
        (document_module, "build_document_handler"),
    ):
        monkeypatch.setattr(module, attr, build_processor)

    # Also patch the registry module since it imports the functions at load time
    for attr in (
        "build_page_handler",
        "build_feed_handler",
        "build_image_handler",
        "build_audio_handler",
        "build_video_handler",
        "build_document_handler",
    ):
        monkeypatch.setattr(registry_module, attr, build_processor)

    monkeypatch.setattr(
        handlers_module,
        "CoverageFetchGate",
        _NoopCoverageGate,
    )

    task_processor = build_task_processor(
        settings=settings,
        page_settings=settings.collection.processors.page,
        logger_factory=SimpleNamespace(
            get_logger_for=lambda *_args, **_kwargs: _NoopLogger()
        ),
        id_generator=SimpleNamespace(),
        fetcher=_RetryingFetchService(),  # type: ignore[arg-type]
        coverage_tracker=SimpleNamespace(),
        scheduler=SimpleNamespace(),
        dataset_writer=SimpleNamespace(),
        url_filter=SimpleNamespace(),
        url_normalizer=SimpleNamespace(),
        host_normalizer=SimpleNamespace(),
        page_content_extractor=SimpleNamespace(),
        discovery_task_builder=SimpleNamespace(),
        network_access_guard=SimpleNamespace(),
        redirector=SimpleNamespace(),
    )

    shared_handler = task_processor._failure_handler
    assert all(
        processor._failure_handler is shared_handler
        for processor in constructed_processors
    )

    fetch_outcome = await task_processor.process(_task())
    processor_outcome = await constructed_processors[0].process_fetched(
        task=_task(),
        result=_result(),
    )

    assert fetch_outcome.retry_after_seconds == retry_wait_seconds
    assert fetch_outcome.stage == "fetch"
    assert processor_outcome.retry_after_seconds == retry_wait_seconds
    assert processor_outcome.stage == retry_stage
