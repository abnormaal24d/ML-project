"""Crawler state persistence subgraph composition.

Builds the storage primitives a crawl needs before its runtime exists:
the checkpoint store and the dead-letter writer. The runtime reader/writer
pair is constructed later in the execution subgraph, where scheduler and
worker pool are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from config.settings.crawler import CrawlStateStoreSettings
from logger.factory import ProjectLoggerFactory
from shared.runtime_primitives import Clock

if TYPE_CHECKING:
    from crawler.runtime.control.crawler_control_directory import (
        CrawlerControlDirectory,
    )
    from crawler.runtime.state.crawl_checkpoint_store import (
        CrawlerCheckpointStore,
    )
    from crawler.runtime.state.crawl_dead_letter_writer import (
        CrawlerDeadLetterWriter,
    )


@dataclass(frozen=True, slots=True)
class CrawlerStatePersistence:
    """Storage primitives built before the crawl runtime exists."""

    checkpoint_store: "CrawlerCheckpointStore | None"
    dead_letter_writer: "CrawlerDeadLetterWriter | None"
    dead_letter_path: Path | None


def build_crawler_state(
    *,
    state_settings: CrawlStateStoreSettings,
    control_directory: "CrawlerControlDirectory",
    clock: Clock,
    logger_factory: ProjectLoggerFactory,
    crawl_session_id: str,
) -> CrawlerStatePersistence:
    """Build crawler state persistence primitives."""
    from crawler.runtime.state.crawl_checkpoint_store import (
        CrawlerCheckpointStore,
    )
    from crawler.runtime.state.crawl_dead_letter_writer import (
        CrawlerDeadLetterWriter,
    )
    from crawler.runtime.state.runtime_checkpoint_payload_builder import (
        RuntimeCheckpointPayloadBuilder,
    )
    from crawler.runtime.state.state_path_resolver import (
        CrawlStatePathResolver,
    )

    if not state_settings.enabled:
        return CrawlerStatePersistence(
            checkpoint_store=None,
            dead_letter_writer=None,
            dead_letter_path=None,
        )

    # Resolve state directory using domain resolver
    state_path_resolver = CrawlStatePathResolver(
        settings=state_settings,
        control_directory=control_directory,
    )
    state_directory = state_path_resolver.resolve(
        crawl_session_id=crawl_session_id,
    )

    checkpoint_store = CrawlerCheckpointStore(
        settings=state_settings,
        state_directory=state_directory,
        checkpoint_path=state_directory / state_settings.checkpoint_filename,
        logger=logger_factory.get_logger_for(CrawlerCheckpointStore),
        payload_builder=RuntimeCheckpointPayloadBuilder(clock=clock),
    )

    dead_letter_writer = None
    dead_letter_path = None
    if state_settings.dead_letter_enabled:
        dead_letter_path = (
            state_directory / state_settings.dead_letter_filename
        )
        dead_letter_writer = CrawlerDeadLetterWriter(
            settings=state_settings,
            dead_letter_path=dead_letter_path,
            logger=logger_factory.get_logger_for(CrawlerDeadLetterWriter),
            clock=clock,
        )

    return CrawlerStatePersistence(
        checkpoint_store=checkpoint_store,
        dead_letter_writer=dead_letter_writer,
        dead_letter_path=dead_letter_path,
    )
