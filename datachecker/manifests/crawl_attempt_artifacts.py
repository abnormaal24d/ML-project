"""Resolved artifacts produced by one crawl attempt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CrawlAttemptArtifacts:
    """Filesystem artifacts produced by a crawl attempt."""

    raw_run_directory: Path | None
    run_summary_path: Path | None

    def has_complete_files_on_disk(self) -> bool:
        """Point 18: explicit name for I/O check."""
        return (
            self.raw_run_directory is not None
            and self.raw_run_directory.is_dir()
            and self.run_summary_path is not None
            and self.run_summary_path.is_file()
        )

    def missing_completion_parts(self) -> list[str]:
        """Point 19: return explicit missing parts for diagnosis."""
        missing: list[str] = []
        if (
            self.raw_run_directory is None
            or not self.raw_run_directory.is_dir()
        ):
            missing.append("raw_run_directory")
        if (
            self.run_summary_path is None
            or not self.run_summary_path.is_file()
        ):
            missing.append("run_summary_path")
        return missing
