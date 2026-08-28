"""Response body read-plan construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind
from crawler.fetching.media.strategy import (
    HeadPreflightAction,
    HeadPreflightResult,
)
from crawler.fetching.network.body.partial_store import (
    PartialPayloadSecurityError,
    PartialPayloadStorage,
)
from crawler.fetching.request.body_plan import BodyReadPlan
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from config.collection.fetching import FetcherSettings
    from crawler.fetching.request.context import (
        FetchRequestContext,
    )


@dataclass(frozen=True, slots=True)
class PartialResumeState:
    """Validated resume coordinates owned by body-plan resolution."""

    partial_path: Path
    resume_offset: int
    content_length: int | None
    owner_token: str
    etag: str | None
    last_modified: str | None

    @property
    def if_range(self) -> str:
        validator = self.etag or self.last_modified
        if not validator:
            raise ValueError("resume state has no representation validator")
        return validator


class BodyReadPlanResolver:
    """
    Build immutable body-read plans from profiles, HEAD results, and
    acceptance.
    """

    def __init__(
        self,
        *,
        settings: FetcherSettings,
        logger: ProjectLogger,
        partial_payload_storage: PartialPayloadStorage | None = None,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._partial_payload_storage = partial_payload_storage

    def build(
        self,
        *,
        context: FetchRequestContext,
        request_headers: Mapping[str, str],
        head_preflight_result: HeadPreflightResult | None = None,
    ) -> BodyReadPlan:
        """Build headers, byte budget, partial flag, and body-read mode."""

        headers = dict(request_headers)
        max_bytes = self._max_bytes_for_plan(
            context=context,
            head_preflight_result=head_preflight_result,
        )

        if self._is_streaming_preflight(head_preflight_result):
            return BodyReadPlan(
                headers=headers,
                max_bytes=max_bytes,
                allow_partial=(
                    context.acceptance.allow_partial_when_oversized
                ),
                mode=HeadPreflightAction.FETCH_STREAMING.value,
            )

        if self._is_forced_probe_preflight(head_preflight_result):
            preflight_action = (
                head_preflight_result.action
                if head_preflight_result is not None
                else HeadPreflightAction.FETCH_PARTIAL
            )
            probe_bytes = self._forced_media_probe_bytes(
                context=context,
            )
            if probe_bytes is not None:
                return self._build_range_read_plan(
                    headers=headers,
                    max_bytes=max_bytes,
                    probe_bytes=probe_bytes,
                    context=context,
                    mode=preflight_action.value,
                )

        if self._should_probe_unknown_length_media(
            context=context,
            head_preflight_result=head_preflight_result,
        ):
            probe_bytes = self._forced_media_probe_bytes(
                context=context,
            )
            if probe_bytes is not None:
                return self._build_range_read_plan(
                    headers=headers,
                    max_bytes=max_bytes,
                    probe_bytes=probe_bytes,
                    context=context,
                    mode="metadata_probe",
                )

        resume_plan = self._build_resume_read_plan(
            headers=headers,
            max_bytes=max_bytes,
            context=context,
        )
        if resume_plan is not None:
            return resume_plan

        return BodyReadPlan(
            headers=headers,
            max_bytes=max_bytes,
            allow_partial=False,
            mode="full",
        )

    def _build_resume_read_plan(
        self,
        *,
        headers: dict[str, str],
        max_bytes: int,
        context: FetchRequestContext,
    ) -> BodyReadPlan | None:
        if self._partial_payload_storage is None:
            return None
        resume_state = self._find_resume_state(url=context.url)
        if resume_state is None or resume_state.resume_offset >= max_bytes:
            return None

        resumed_headers = dict(headers)
        resumed_headers["Range"] = self._build_resume_range_header(
            resume_state=resume_state,
            max_bytes=max_bytes,
        )
        resumed_headers["Accept-Encoding"] = "identity"
        resumed_headers["If-Range"] = resume_state.if_range

        self._logger.debug(
            "media_partial_resume_requested",
            url=context.url,
            host=context.host,
            requested_kind=context.requested_kind,
            range_header=resumed_headers["Range"],
            resume_offset=resume_state.resume_offset,
            observed_bytes=resume_state.resume_offset,
            partial_path=str(resume_state.partial_path),
            full_body_max_bytes=max_bytes,
        )

        return BodyReadPlan(
            headers=resumed_headers,
            max_bytes=max_bytes,
            allow_partial=True,
            mode="resume_partial",
            resume_partial_path=resume_state.partial_path,
            resume_offset=resume_state.resume_offset,
            resume_owner_token=resume_state.owner_token,
            resume_etag=resume_state.etag,
            resume_last_modified=resume_state.last_modified,
        )

    def _find_resume_state(self, *, url: str) -> PartialResumeState | None:
        storage = self._partial_payload_storage
        if storage is None:
            return None

        expected_fingerprint = storage.url_fingerprint(url)
        candidates: list[PartialResumeState] = []
        for metadata_path in storage.iter_metadata_paths():
            metadata = storage.read_metadata(metadata_path=metadata_path)
            if metadata.get("url_sha256") != expected_fingerprint:
                continue
            partial_path = metadata_path.with_name(
                metadata_path.name.removesuffix(".json")
            )
            owner_token = str(metadata.get("owner_token") or "")
            try:
                _, validated_metadata, actual_size = storage.validate_resume(
                    path=partial_path,
                    owner_token=owner_token,
                )
            except (OSError, PartialPayloadSecurityError):
                continue

            content_length = self._optional_nonnegative_int(
                validated_metadata.get("content_length")
            )
            if content_length is not None and content_length <= actual_size:
                continue
            candidates.append(
                PartialResumeState(
                    partial_path=partial_path,
                    resume_offset=actual_size,
                    content_length=content_length,
                    owner_token=owner_token,
                    etag=self._optional_string(validated_metadata.get("etag")),
                    last_modified=self._optional_string(
                        validated_metadata.get("last_modified")
                    ),
                )
            )
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.resume_offset)

    @staticmethod
    def _build_resume_range_header(
        *,
        resume_state: PartialResumeState,
        max_bytes: int,
    ) -> str:
        start = int(resume_state.resume_offset)
        budget = int(max_bytes)
        if start < 0 or budget <= start:
            raise ValueError("resume offset exhausts the response byte budget")
        end = budget - 1
        if resume_state.content_length is not None:
            end = min(end, resume_state.content_length - 1)
        if end < start:
            raise ValueError("resume range is empty")
        return f"bytes={start}-{end}"

    @staticmethod
    def _optional_nonnegative_int(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    @staticmethod
    def _optional_string(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned or None

    def _build_range_read_plan(
        self,
        *,
        headers: dict[str, str],
        max_bytes: int,
        probe_bytes: int,
        context: FetchRequestContext,
        mode: str,
    ) -> BodyReadPlan:
        bounded_probe_bytes = max(1, min(int(probe_bytes), int(max_bytes)))
        headers["Range"] = f"bytes=0-{bounded_probe_bytes - 1}"
        headers["Accept-Encoding"] = "identity"

        self._logger.debug(
            "media_metadata_probe_requested",
            url=context.url,
            host=context.host,
            requested_kind=context.requested_kind,
            range_header=headers["Range"],
            probe_bytes=bounded_probe_bytes,
            full_body_max_bytes=max_bytes,
            body_read_mode=mode,
        )

        return BodyReadPlan(
            headers=headers,
            max_bytes=bounded_probe_bytes,
            allow_partial=True,
            mode=mode,
            probe_bytes=bounded_probe_bytes,
        )

    @staticmethod
    def _is_streaming_preflight(
        result: HeadPreflightResult | None,
    ) -> bool:
        return (
            result is not None
            and result.action == HeadPreflightAction.FETCH_STREAMING
        )

    @staticmethod
    def _is_forced_probe_preflight(
        result: HeadPreflightResult | None,
    ) -> bool:
        return result is not None and result.action in {
            HeadPreflightAction.METADATA_ONLY,
            HeadPreflightAction.FETCH_PARTIAL,
        }

    @staticmethod
    def _max_bytes_for_plan(
        *,
        context: FetchRequestContext,
        head_preflight_result: HeadPreflightResult | None,
    ) -> int:
        content_type = (
            None
            if head_preflight_result is None
            else getattr(head_preflight_result, "content_type", None)
        )
        return int(context.acceptance.max_bytes_for_content_type(content_type))

    @staticmethod
    def _should_probe_unknown_length_media(
        *,
        context: FetchRequestContext,
        head_preflight_result: HeadPreflightResult | None,
    ) -> bool:
        requested_kind = context.requested_kind
        if requested_kind not in {MediaKind.AUDIO, MediaKind.VIDEO}:
            return False
        if head_preflight_result is None:
            return True
        return getattr(head_preflight_result, "content_length", None) is None

    def _forced_media_probe_bytes(
        self,
        *,
        context: FetchRequestContext,
    ) -> int | None:
        requested_kind = context.requested_kind
        if requested_kind is MediaKind.AUDIO:
            return int(self._settings.audio_metadata_probe_bytes)
        if requested_kind is MediaKind.VIDEO:
            return int(self._settings.video_metadata_probe_bytes)
        return None
