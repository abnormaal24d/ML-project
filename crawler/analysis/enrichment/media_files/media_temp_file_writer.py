"""Write temporary media files from bytes for analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

from logger.project_logger import ProjectLogger


class MediaTempFileWriter:
    """Write temporary media files from in-memory bytes.

    Logs the exact temporary file location for later cleanup.
    """

    def __init__(self, *, logger: ProjectLogger) -> None:
        self._logger = logger

    def write_bytes(self, *, body: bytes, suffix: str) -> Path:
        with tempfile.NamedTemporaryFile(
            delete=False,
            prefix="crawler_media_",
            suffix=suffix,
        ) as handle:
            handle.write(body)
            path = Path(handle.name)
            self._logger.debug(
                "temporary_media_file_writer_written",
                extra={
                    "temp_path": str(path),
                    "prefix": path.name[:8] if path.name else None,
                    "size_bytes": len(body),
                },
            )
            return path
