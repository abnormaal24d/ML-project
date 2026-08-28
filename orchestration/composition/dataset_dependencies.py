"""Build dataset-assembly dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mmcrawler_datasets.assembly.text_chunk_splitter import TextChunkSplitter

if TYPE_CHECKING:
    from collections.abc import Callable

    from config.settings.root import Settings
    from logger.project_logger import ProjectLogger


def build_text_chunk_splitter(
    *,
    settings: Settings,
    assign_split: Callable[[str], str | None],
    logger: ProjectLogger,
) -> TextChunkSplitter:
    """Build the dataset-level text chunk splitter."""

    return TextChunkSplitter(
        settings=settings.datasets.curation.document_chunker,
        assign_split=assign_split,
        logger=logger,
    )


__all__ = ["build_text_chunk_splitter"]
