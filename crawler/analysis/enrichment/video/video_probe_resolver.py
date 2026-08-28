"""Video metadata fallback resolution for partial media probes."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from crawler.analysis.enrichment.video.video_metadata_detection import (
    is_oversized_video_metadata_probe,
    is_partial_video_probe,
)
from logger.project_logger import ProjectLogger

_VIDEO_PROBE_SOFT_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    ValueError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from crawler.analysis.enrichment.video.video_probe_download import (
        VideoFullProbeDownloader,
        VideoTailProbeDownloader,
    )
    from crawler.extraction.payloads.video_payload_extractor import (
        VideoPayloadExtractor,
    )
    from crawler.fetching.results.result import FetchResult
    from preprocessing.media.video.mp4_tail_metadata import (
        Mp4TailMetadataReader,
    )


@dataclass(frozen=True, slots=True)
class VideoProbeResult:
    """Resolved metadata, analysis path, and probe status."""

    metadata: dict[str, Any]
    metadata_status: str
    analysis_path: Path
    cleanup_path: Path | None = None
    video_probe_metadata: dict[str, Any] | None = None
    video_probe_status: str | None = None
    video_probe_error_type: str | None = None


# VideoProbeLogEvent removed per instructions; logging is now direct inline with logger.info
# The include_* flags are passed directly as booleans to the log call when relevant.


class VideoProbeResolver:
    """Read video metadata and resolve fallback after partial probes.

    Responsibilities and collaborators:

    - payload_extractor: always used first to get objective container/stream
      metadata from the (possibly partial) local file.
    - tail_probe_downloader + tail_metadata_reader: used for "tail" (end-of-file)
      probe when the initial head was truncated/partial. The downloader fetches
      a small suffix; the Mp4TailMetadataReader parses moov/mvhd etc. from it.
    - full_probe_downloader: used as last-resort full metadata fetch when tail
      fallback also fails (for non-MP4 or when tail bytes insufficient).
    """

    def __init__(
        self,
        *,
        payload_extractor: VideoPayloadExtractor,
        tail_probe_downloader: VideoTailProbeDownloader | None,
        tail_metadata_reader: Mp4TailMetadataReader | None,
        full_probe_downloader: VideoFullProbeDownloader | None,
        logger: ProjectLogger,
        container_probe: Any | None = None,
    ) -> None:
        self._payload_extractor = payload_extractor
        self._tail_probe_downloader = tail_probe_downloader
        self._tail_metadata_reader = tail_metadata_reader
        self._full_probe_downloader = full_probe_downloader
        self._logger = logger
        self._container_probe = container_probe

    async def resolve_probe(
        self,
        *,
        result: FetchResult,
        path: Path,
    ) -> VideoProbeResult:
        """Return metadata, status, analysis path, and probe fields."""

        if result.payload is None:
            self._logger.warning(
                "video_probe_input_no_payload",
                url=getattr(result, "final_url", None),
            )
            return VideoProbeResult(
                metadata={},
                metadata_status="missing",
                analysis_path=path,
            )
        if not path.exists():
            self._logger.warning(
                "video_probe_input_path_missing",
                path=str(path),
                url=getattr(result, "final_url", None),
            )

        self._logger.debug(
            "video_probe_characterize_start",
            path=str(path),
            url=getattr(result, "final_url", None),
        )
        # Bounded characterize to prevent hangs (per audit P0 timeouts)
        metadata = await asyncio.wait_for(
            asyncio.to_thread(
                self._extract_payload_metadata,
                path=path,
            ),
            timeout=60.0,
        )
        probe_metadata: dict[str, Any] | None = None
        probe_status: str | None = None
        probe_error_type: str | None = None
        try:
            probe = self._container_probe
            if probe is None:
                raise RuntimeError("container_probe_not_configured")
            probe_meta = dict(
                await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: probe.probe_video(path=str(path)) or {}
                    ),
                    timeout=30.0,  # per audit P0 timeouts on IO
                )
            )
            self._logger.debug(
                "video_probe_meta_received",
                url=getattr(result, "final_url", None),
                path=str(path),
                has_real=_has_real_probe_metadata(probe_meta=probe_meta),
                probe_keys=sorted(probe_meta.keys())[:6],
            )
            if _has_real_probe_metadata(probe_meta=probe_meta):
                probe_metadata = dict(probe_meta)
                probe_status = "passed"
                if isinstance(metadata, dict):
                    merged = dict(probe_meta)
                    merged.update(
                        {
                            key: value
                            for key, value in metadata.items()
                            if key not in merged
                        }
                    )
                    metadata = merged
                    metadata["probe_enriched"] = True
            elif probe_meta:
                probe_metadata = dict(probe_meta)
                probe_status = str(probe_meta.get("probe_status") or "failed")
                probe_error_type = type(probe_meta).__name__
        except _VIDEO_PROBE_SOFT_ERRORS as exc:
            probe_status = "failed"
            probe_error_type = type(exc).__name__
            self._logger.warning(
                "video_probe_enrichment_failed",
                url=getattr(result, "url", None),
                content_type=getattr(result, "content_type", None),
                error_type=type(exc).__name__,
            )

        if metadata:
            status = (
                "partial_extracted"
                if is_partial_video_probe(result=result)
                else "full_extracted"
            )
            return VideoProbeResult(
                metadata=metadata,
                metadata_status=status,
                analysis_path=path,
                video_probe_metadata=probe_metadata,
                video_probe_status=probe_status,
                video_probe_error_type=probe_error_type,
            )

        if not is_partial_video_probe(result=result):
            return VideoProbeResult(
                metadata={},
                metadata_status="missing",
                analysis_path=path,
                video_probe_metadata=probe_metadata,
                video_probe_status=probe_status,
                video_probe_error_type=probe_error_type,
            )

        return await self._resolve_partial_probe_failure(
            result=result,
            path=path,
            oversized_metadata_probe=is_oversized_video_metadata_probe(
                result=result,
            ),
            probe_metadata=probe_metadata,
            probe_status=probe_status,
            probe_error_type=probe_error_type,
        )

    def _extract_payload_metadata(self, *, path: Path) -> dict[str, Any]:
        extraction = self._payload_extractor.extract_from_path(path=path)
        return extraction.as_metadata_dict() if extraction is not None else {}

    async def _resolve_partial_probe_failure(
        self,
        *,
        result: FetchResult,
        path: Path,
        oversized_metadata_probe: bool,
        probe_metadata: dict[str, Any] | None,
        probe_status: str | None,
        probe_error_type: str | None,
    ) -> VideoProbeResult:
        """Return metadata, status, analysis path, and cleanup path."""

        if oversized_metadata_probe:
            return VideoProbeResult(
                metadata={},
                metadata_status="partial_probe_failed_fallback_head_only",
                analysis_path=path,
                video_probe_metadata=probe_metadata,
                video_probe_status=probe_status,
                video_probe_error_type=probe_error_type,
            )

        self._logger.info(
            "video_metadata_probe_failed",
            reason="moov_atom_missing",
            probe="partial",
            url=result.final_url,
            stderr_suppressed=True,
        )

        tail_metadata = await self._read_tail_probe_metadata(result=result)
        if tail_metadata:
            self._logger.info(
                "video_metadata_probe_recovered",
                reason="moov_atom_missing",
                probe="tail_fallback",
                url=result.final_url,
            )
            return VideoProbeResult(
                metadata=tail_metadata,
                metadata_status="tail_fallback_extracted",
                analysis_path=path,
                video_probe_metadata=probe_metadata,
                video_probe_status=probe_status,
                video_probe_error_type=probe_error_type,
            )

        fallback_path = await self._download_full_probe_file(result=result)
        if fallback_path is None:
            return VideoProbeResult(
                metadata={},
                metadata_status="partial_failed",
                analysis_path=path,
                video_probe_metadata=probe_metadata,
                video_probe_status=probe_status,
                video_probe_error_type=probe_error_type,
            )

        fallback_metadata = await asyncio.wait_for(
            asyncio.to_thread(
                self._extract_payload_metadata,
                path=fallback_path,
            ),
            timeout=60.0,
        )
        if fallback_metadata:
            self._logger.info(
                "video_metadata_probe_recovered",
                reason="moov_atom_missing",
                probe="full_fallback",
                url=result.final_url,
                fallback_bytes=(
                    fallback_path.stat().st_size if fallback_path else None
                ),
            )
            return VideoProbeResult(
                metadata=fallback_metadata,
                metadata_status="full_fallback_extracted",
                analysis_path=fallback_path,
                cleanup_path=fallback_path,
                video_probe_metadata=probe_metadata,
                video_probe_status=probe_status,
                video_probe_error_type=probe_error_type,
            )

        self._logger.info(
            "video_metadata_probe_fallback_failed",
            reason="full_probe_unreadable",
            probe="full_fallback",
            url=result.final_url,
            fallback_bytes=(
                fallback_path.stat().st_size if fallback_path else None
            ),
        )
        fallback_path.unlink(missing_ok=True)
        return VideoProbeResult(
            metadata={},
            metadata_status="full_fallback_failed",
            analysis_path=path,
            video_probe_metadata=probe_metadata,
            video_probe_status=probe_status,
            video_probe_error_type=probe_error_type,
        )

    async def _read_tail_probe_metadata(
        self,
        *,
        result: FetchResult,
    ) -> dict[str, Any]:
        if (
            self._tail_probe_downloader is None
            or self._tail_metadata_reader is None
        ):
            return {}

        payload = result.payload
        tail_path = await asyncio.wait_for(
            asyncio.to_thread(
                self._tail_probe_downloader.download,
                url=result.final_url,
                source_content_length=(
                    None if payload is None else payload.source_content_length
                ),
            ),
            timeout=30.0,  # per audit P0 timeouts on IO
        )
        if tail_path is None:
            self._logger.info(
                "video_metadata_probe_fallback_failed",
                reason="download_failed",
                probe="tail_fallback",
                url=result.final_url,
            )
            return {}

        try:
            metadata = await asyncio.to_thread(
                self._tail_metadata_reader.read,
                path=tail_path,
            )
            if not metadata:
                self._logger.info(
                    "video_metadata_probe_fallback_failed",
                    reason="moov_atom_unreadable",
                    probe="tail_fallback",
                    url=result.final_url,
                )
            return metadata
        finally:
            with suppress(OSError):
                tail_path.unlink(missing_ok=True)

    async def _download_full_probe_file(
        self,
        *,
        result: FetchResult,
    ) -> Path | None:
        if self._full_probe_downloader is None:
            return None

        payload = result.payload
        max_bytes = self._full_probe_downloader.max_bytes
        source_content_length = (
            None if payload is None else payload.source_content_length
        )
        if (
            source_content_length is not None
            and source_content_length > max_bytes
        ):
            self._logger.info(
                "video_metadata_probe_fallback_skipped",
                reason="source_content_length_exceeded",
                probe="full_fallback",
                url=result.final_url,
                max_bytes=max_bytes,
            )
            return None

        fallback_path = await asyncio.wait_for(
            asyncio.to_thread(
                self._full_probe_downloader.download,
                url=result.final_url,
            ),
            timeout=60.0,  # per audit P0 timeouts on IO
        )
        if fallback_path is None:
            self._logger.info(
                "video_metadata_probe_fallback_failed",
                reason="download_failed",
                probe="full_fallback",
                url=result.final_url,
            )
        return fallback_path


def _has_real_probe_metadata(*, probe_meta: Mapping[str, object]) -> bool:
    if probe_meta.get("probe_status") in {
        "failed",
        "no_video_stream",
        "error",
    }:
        return False
    return any(
        probe_meta.get(key) is not None
        for key in (
            "duration_seconds",
            "width",
            "height",
            "fps",
            "video_codec",
            "container_format",
        )
    )
