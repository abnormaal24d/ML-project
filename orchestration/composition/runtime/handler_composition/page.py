"""Page handler composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.collection.processors import PageProcessorSettings
from config.coverage.settings import CoverageSettings
from crawler.coverage.discovery_budget import PageDiscoveryCapResolver
from crawler.discovery.discovery_task_builder import DiscoveryTaskBuilder
from crawler.extraction.modalities.page_content_extractor import (
    PageContentExtractor,
)
from crawler.processing.handlers.page_handler import PageHandler
from crawler.scheduling.url_scheduler import UrlScheduler
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
from logger.factory import ProjectLoggerFactory

if TYPE_CHECKING:
    from crawler.coverage.state import CoverageState
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.url_filter.url_admission_filter import (
        UrlAdmissionFilter,
    )
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
    )
    from shared.runtime_primitives import IdGenerator


def build_page_handler(
    *,
    page_settings: PageProcessorSettings,
    coverage_settings: CoverageSettings,
    page_content_extractor: PageContentExtractor,
    discovery_task_builder: DiscoveryTaskBuilder,
    url_filter: UrlAdmissionFilter,
    url_normalizer: UrlNormalizer,
    scheduler: UrlScheduler,
    writer: DatasetWriter,
    coverage: CoverageState,
    host_normalizer: HostNormalizer,
    id_generator: IdGenerator,
    rejected_reporter: object | None,
    logs: ProjectLoggerFactory,
    failure_handler: ProcessorFailureHandler,
) -> PageHandler:
    """Build the page handler with every external dependency visible."""

    return PageHandler(
        settings=page_settings,
        page_content_extractor=page_content_extractor,
        discovery_task_builder=discovery_task_builder,
        url_filter=url_filter,
        url_normalizer=url_normalizer,
        scheduler=scheduler,
        dataset_writer=writer,
        logger=logs.get_logger_for(PageHandler),
        failure_handler=failure_handler,
        cap_resolver=PageDiscoveryCapResolver(
            coverage_settings=coverage_settings
        ),
        coverage_tracker=coverage,
        focus_asset_boost=coverage_settings.focus.asset_ranking_boost,
        host_normalizer=host_normalizer,
        id_generator=id_generator,
        rejected_discovery_reporter=rejected_reporter,
    )


__all__ = ["build_page_handler"]
