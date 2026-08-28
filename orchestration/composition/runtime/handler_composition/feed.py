"""Feed handler composition."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, cast

from config.collection.processors import FeedProcessorSettings
from crawler.discovery.discovery_task_builder import DiscoveryTaskBuilder
from crawler.extraction.modalities.feed_content_extractor import (
    FeedContentExtractor,
)
from crawler.processing.handlers.feed_handler import FeedHandler
from crawler.scheduling.url_scheduler import UrlScheduler
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
from logger.factory import ProjectLoggerFactory

if TYPE_CHECKING:
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.url_filter.url_admission_filter import (
        UrlAdmissionFilter,
    )
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
    )
    from shared.runtime_primitives import IdGenerator


def _load_feed_parser() -> ModuleType:
    try:
        import feedparser  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError(
            "feed handling requires the 'feedparser' dependency to be "
            "installed"
        ) from exc
    return cast(ModuleType, feedparser)


def build_feed_handler(
    *,
    feed_settings: FeedProcessorSettings,
    url_filter: UrlAdmissionFilter,
    url_normalizer: UrlNormalizer,
    scheduler: UrlScheduler,
    writer: DatasetWriter,
    id_generator: IdGenerator,
    host_normalizer: HostNormalizer,
    logs: ProjectLoggerFactory,
    failure_handler: ProcessorFailureHandler,
) -> FeedHandler:
    """Build the feed handler with explicit discovery dependencies."""

    return FeedHandler(
        settings=feed_settings,
        url_filter=url_filter,
        url_normalizer=url_normalizer,
        scheduler=scheduler,
        dataset_writer=writer,
        logger=logs.get_logger_for(FeedHandler),
        failure_handler=failure_handler,
        feed_content_extractor=FeedContentExtractor(
            parser=_load_feed_parser(),
            max_entries=feed_settings.max_feed_entries,
            logger=logs.get_logger_for(FeedContentExtractor),
        ),
        id_generator=id_generator,
        host_normalizer=host_normalizer,
    )


__all__ = ["build_feed_handler"]
