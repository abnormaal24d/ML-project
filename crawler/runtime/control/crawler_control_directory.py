"""Crawler runtime control directory resolution."""

from __future__ import annotations

from pathlib import Path

from config.settings.crawler import CrawlerSettings


class CrawlerControlDirectory:
    """Resolve the crawler runtime control directory."""

    def __init__(
        self,
        *,
        settings: CrawlerSettings,
        project_root: Path | None,
    ) -> None:
        self._settings = settings
        self._project_root = project_root

    def path(self) -> Path:
        configured = Path(self._settings.control_directory)

        if configured.is_absolute():
            raise ValueError("crawler control_directory must be relative")

        if self._project_root is not None:
            return self._within_project_root(
                project_root=self._project_root,
                configured=configured,
            )

        current = Path.cwd().resolve()
        for candidate in (current, *current.parents):
            if (
                (candidate / "pyproject.toml").exists()
                and (candidate / "config").exists()
                and (candidate / "crawler").exists()
                and (candidate / "orchestration").exists()
            ):
                return self._within_project_root(
                    project_root=candidate,
                    configured=configured,
                )

        return self._within_project_root(
            project_root=current,
            configured=configured,
        )

    @staticmethod
    def _within_project_root(*, project_root: Path, configured: Path) -> Path:
        root = Path(project_root).resolve()
        resolved = (root / configured).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                "crawler control_directory must remain within project_root"
            )
        return resolved

    def ensure_exists(self) -> None:
        self.path().mkdir(parents=True, exist_ok=True)

    def pause_flag_path(self) -> Path:
        """Return the crawler pause-flag path."""

        return self.path() / self._settings.pause_flag_filename

    def stop_flag_path(self) -> Path:
        """Return the crawler stop-flag path."""

        return self.path() / self._settings.stop_flag_filename

    def should_pause(self) -> bool:
        """Return whether the crawler pause flag exists."""

        return self.pause_flag_path().exists()

    def should_stop(self) -> bool:
        """Return whether the crawler stop flag exists."""

        return self.stop_flag_path().exists()

    def consume_stop(self) -> bool:
        """Atomically consume one pending operator stop request."""

        try:
            self.stop_flag_path().unlink()
        except FileNotFoundError:
            return False
        return True

    def request_pause(self) -> None:
        """Create the operator pause flag."""

        self.ensure_exists()
        self.pause_flag_path().touch(exist_ok=True)

    def clear_pause(self) -> None:
        """Remove the operator pause flag when present."""

        self.pause_flag_path().unlink(missing_ok=True)

    def clear_stop(self) -> None:
        """Remove an operator stop request that has not been consumed."""

        self.stop_flag_path().unlink(missing_ok=True)

    def request_stop(self) -> None:
        """Create the operator stop flag."""

        self.ensure_exists()
        self.stop_flag_path().touch(exist_ok=True)

    def status(
        self,
        *,
        crawl_state_status: str | None = None,
        workflow_id: str | None = None,
        generation_id: str | None = None,
        attempt_id: str | None = None,
        crawl_state_path: Path,
    ) -> dict[str, object]:
        """Return the current operator flags and persisted workflow state."""

        return {
            "control_directory": str(self.path()),
            "paused": self.should_pause(),
            "stop_requested": self.should_stop(),
            "crawl_state_path": str(crawl_state_path),
            "crawl_state_exists": crawl_state_path.is_file(),
            "crawl_status": crawl_state_status or "missing",
            "workflow_id": workflow_id,
            "generation_id": generation_id,
            "attempt_id": attempt_id,
        }
