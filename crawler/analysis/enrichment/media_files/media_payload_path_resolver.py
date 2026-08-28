"""Resolve local media file paths for fetched payloads (for enrichment)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.fetching.results.result import FetchResult

from crawler.analysis.enrichment.media_files.media_temp_file_writer import (
    MediaTempFileWriter,
)


class MediaPayloadPathResolver:
    """Resolve a local analysis path for fetched media payloads."""

    def __init__(
        self,
        *,
        writer: MediaTempFileWriter,
        logger: ProjectLogger,
    ) -> None:
        self._writer = writer
        self._logger = logger
        self._owned_paths: set[Path] = set()
        self._lock = RLock()

    async def resolve_path(self, *, result: FetchResult, suffix: str) -> Path:
        """Return an on-disk file path for the fetch payload."""

        payload = result.payload
        if payload is not None:
            temp_path = payload.temp_path
            if not isinstance(temp_path, Path):
                raise TypeError(
                    "FetchResult payload temp_path must be a pathlib.Path"
                )
            self._logger.debug(
                "media_file_resolver_temp_path_exists",
                url=result.final_url,
                temp_path=str(temp_path),
                prefix=temp_path.name[:8] if temp_path.name else None,
            )
            return temp_path

        body = result.read_body_optional()
        if body:
            self._logger.debug(
                "media_file_resolver_download_start",
                url=result.final_url,
                suffix=suffix,
                body_bytes=len(body),
            )
            path = await asyncio.to_thread(
                self._writer.write_bytes,
                body=body,
                suffix=suffix,
            )
            resolved_path = path.resolve()
            with self._lock:
                self._owned_paths.add(resolved_path)
            self._logger.debug(
                "media_file_resolver_download_complete",
                url=result.final_url,
                temp_path=str(path),
                prefix=path.name[:8] if path.name else None,
            )
            return path

        raise ValueError(
            f"FetchResult payload is missing for media URL: {result.final_url}"
        )

    def owns_path(self, path: Path | None) -> bool:
        """Return whether this resolver created the temporary media file."""

        if path is None:
            return False

        with self._lock:
            return path.resolve() in self._owned_paths

    def cleanup_owned_path(self, path: Path | None) -> None:
        """Delete resolver-owned temporary media files.

        Persisted fetch payloads are owned by the fetch layer and are never
        deleted here.
        """

        if path is None:
            return

        resolved_path = path.resolve()
        with self._lock:
            if resolved_path not in self._owned_paths:
                return
            self._owned_paths.remove(resolved_path)

        try:
            path.unlink(missing_ok=True)
            self._logger.debug(
                "media_file_resolver_cleanup",
                path=str(path),
                prefix=path.name[:8] if path.name else None,
            )
        except OSError as exc:
            self._logger.warning(
                "media_file_resolver_cleanup_failed",
                path=str(path),
                error_type=type(exc).__name__,
            )
