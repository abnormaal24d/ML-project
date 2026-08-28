"""Crawler state directory resolution and validation.

Pure domain logic for resolving and validating state directories.
No I/O beyond path operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings.crawler import CrawlStateStoreSettings
    from crawler.runtime.control.crawler_control_directory import (
        CrawlerControlDirectory,
    )


class CrawlStatePathResolver:
    """Resolve and validate crawler state directory paths.

    Enforces containment within the control directory for security.
    """

    def __init__(
        self,
        *,
        settings: "CrawlStateStoreSettings",
        control_directory: "CrawlerControlDirectory",
    ) -> None:
        self._settings = settings
        self._control_directory = control_directory

    def resolve(self, crawl_session_id: str | None = None) -> Path:
        """Resolve the state directory for checkpoint and dead-letter stores.

        Args:
            crawl_session_id: Optional session ID for run-scoped state.

        Returns:
            Resolved state directory path.

        Raises:
            ValueError: If path escapes control directory or session ID is invalid.
        """
        control_root = self._control_directory.path().resolve()
        configured = Path(self._settings.state_subdirectory)
        state_directory = (control_root / configured).resolve()

        if not state_directory.is_relative_to(control_root):
            raise ValueError(
                "crawler state_subdirectory must remain within control_directory"
            )

        if self._settings.run_scoped_state and crawl_session_id:
            state_directory = self._apply_session_scope(
                state_directory=state_directory,
                control_root=control_root,
                crawl_session_id=crawl_session_id,
            )

        return state_directory

    def _apply_session_scope(
        self,
        *,
        state_directory: Path,
        control_root: Path,
        crawl_session_id: str,
    ) -> Path:
        """Apply run-scoped state directory with validation."""
        session_component = Path(crawl_session_id)

        if (
            crawl_session_id in {".", ".."}
            or session_component.name != crawl_session_id
        ):
            raise ValueError(
                "crawl_session_id must be a single path component"
            )

        state_directory = (state_directory / session_component).resolve()

        if not state_directory.is_relative_to(control_root):
            raise ValueError(
                "crawler run-scoped state must remain within control_directory"
            )

        return state_directory
