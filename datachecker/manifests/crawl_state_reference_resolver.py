"""Resolve crawl-state manifest references."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from datachecker.manifests.crawl_attempt_artifacts import CrawlAttemptArtifacts

if TYPE_CHECKING:
    from config.path_resolution.workflow_artifact_paths import (
        ArtifactPathRegistry,
    )
    from datachecker.manifests.crawl_manifest import CrawlManifest
    from datachecker.manifests.crawl_state_manifest import CrawlStateManifest
    from logger.project_logger import ProjectLogger


class CrawlAttemptTimestampError(ValueError):
    """Raised when attempt timestamps violate the persisted UTC schema."""

    def __init__(
        self,
        *,
        artifact_name: str,
        field_name: str,
        raw_value: object,
        reason: str,
        artifact_path: Path | None = None,
    ) -> None:
        self.artifact_name = artifact_name
        self.field_name = field_name
        self.raw_value = raw_value
        self.reason = reason
        self.artifact_path = artifact_path
        location = (
            ""
            if artifact_path is None
            else f", artifact_path={str(artifact_path)!r}"
        )
        super().__init__(
            "invalid crawl-attempt timestamp: "
            f"artifact={artifact_name!r}, field={field_name!r}, "
            f"value={raw_value!r}, reason={reason!r}{location}"
        )


class CrawlStateReferenceResolver:
    """Resolve existing crawl-state paths and raw attempt artifacts."""

    def __init__(
        self,
        *,
        artifact_path_registry: ArtifactPathRegistry,
        logger: ProjectLogger,
        project_root: Path,
    ) -> None:
        self._artifact_path_registry = artifact_path_registry
        self._logger = logger
        self._project_root = Path(project_root).resolve(strict=True)
        if not self._project_root.is_dir():
            raise NotADirectoryError(
                f"project_root is not a directory: {self._project_root}"
            )

    def read_crawl_state(self) -> CrawlStateManifest | None:
        import json

        from datachecker.manifests.crawl_state_manifest import (
            CrawlStateManifest,
        )

        path = self._artifact_path_registry.crawl_state_manifest_path()
        if not path.is_file():
            return None

        try:
            payload = json.loads(path.read_text("utf-8"))
            manifest = CrawlStateManifest.from_payload(payload)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

        return self._normalize_crawl_state_references(manifest=manifest)

    def existing_crawl_manifest_path(self) -> Path | None:
        path = self._artifact_path_registry.crawl_manifest_path()
        return path if path.exists() else None

    def existing_crawl_manifest(self) -> CrawlManifest | None:
        import json

        from datachecker.manifests.crawl_manifest import CrawlManifest

        path = self._artifact_path_registry.crawl_manifest_path()
        if not path.is_file():
            return None

        try:
            payload = json.loads(path.read_text("utf-8"))
            return CrawlManifest.from_payload(payload)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

    def existing_crawl_manifest_completed_at(self) -> str | None:
        manifest = self.existing_crawl_manifest()
        return None if manifest is None else manifest.completed_at

    def latest_raw_attempt_since(
        self,
        *,
        started_at: datetime | None,
        include_running: bool = False,
        current_attempt_id: str | None = None,
    ) -> CrawlAttemptArtifacts:
        """Resolve the active attempt only when all timestamps meet the cutoff."""
        _ = include_running  # Issue 7 adds status filtering separately.

        state = self.read_crawl_state()
        if (
            state is None
            or current_attempt_id is None
            or state.attempt_id != current_attempt_id
            or not state.raw_run_id
            or not state.crawl_session_id
            or state.raw_run_directory is None
            or state.run_summary_path is None
            or not state.raw_run_directory.is_dir()
            or not state.run_summary_path.is_file()
        ):
            return CrawlAttemptArtifacts(None, None)

        if started_at is None:
            return CrawlAttemptArtifacts(
                state.raw_run_directory,
                state.run_summary_path,
            )

        cutoff = self._normalize_requested_boundary(started_at)
        state_manifest_path = (
            self._artifact_path_registry.crawl_state_manifest_path()
        )
        state_started_at = self._required_utc_timestamp(
            value=state.started_at,
            artifact_name="crawl_state",
            field_name="started_at",
            artifact_path=state_manifest_path,
        )
        state_completed_at = self._optional_utc_timestamp(
            value=state.completed_at,
            artifact_name="crawl_state",
            field_name="completed_at",
            artifact_path=state_manifest_path,
        )

        import json

        summary_payload = {}
        if state.run_summary_path.is_file():
            try:
                summary_payload = json.loads(
                    state.run_summary_path.read_text("utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass

        summary_started_at = self._required_utc_timestamp(
            value=summary_payload.get("started_at"),
            artifact_name="raw_run_summary",
            field_name="started_at",
            artifact_path=state.run_summary_path,
        )
        summary_completed_at = self._optional_utc_timestamp(
            value=summary_payload.get("completed_at"),
            artifact_name="raw_run_summary",
            field_name="completed_at",
            artifact_path=state.run_summary_path,
        )

        self._validate_timestamp_order(
            started_at=state_started_at,
            completed_at=state_completed_at,
            artifact_name="crawl_state",
            artifact_path=state_manifest_path,
        )
        self._validate_timestamp_order(
            started_at=summary_started_at,
            completed_at=summary_completed_at,
            artifact_name="raw_run_summary",
            artifact_path=state.run_summary_path,
        )
        if summary_started_at < state_started_at:
            raise CrawlAttemptTimestampError(
                artifact_name="raw_run_summary",
                field_name="started_at",
                raw_value=summary_payload.get("started_at"),
                reason=("raw run started before its owning crawl attempt"),
                artifact_path=state.run_summary_path,
            )

        observed_timestamps = {
            "crawl_state.started_at": state_started_at,
            "raw_run_summary.started_at": summary_started_at,
        }
        if state_completed_at is not None:
            observed_timestamps["crawl_state.completed_at"] = (
                state_completed_at
            )
        if summary_completed_at is not None:
            observed_timestamps["raw_run_summary.completed_at"] = (
                summary_completed_at
            )

        stale_fields = tuple(
            field_name
            for field_name, timestamp in observed_timestamps.items()
            if timestamp < cutoff
        )
        if stale_fields:
            self._logger.warning(
                "workflow_raw_attempt_before_requested_boundary_ignored",
                attempt_id=state.attempt_id,
                raw_run_id=state.raw_run_id,
                crawl_session_id=state.crawl_session_id,
                requested_started_at=cutoff.isoformat(),
                crawl_state_started_at=state_started_at.isoformat(),
                crawl_state_completed_at=(
                    None
                    if state_completed_at is None
                    else state_completed_at.isoformat()
                ),
                raw_run_started_at=summary_started_at.isoformat(),
                raw_run_completed_at=(
                    None
                    if summary_completed_at is None
                    else summary_completed_at.isoformat()
                ),
                stale_fields=stale_fields,
            )
            return CrawlAttemptArtifacts(None, None)

        return CrawlAttemptArtifacts(
            state.raw_run_directory,
            state.run_summary_path,
        )

    @staticmethod
    def _normalize_requested_boundary(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise CrawlAttemptTimestampError(
                artifact_name="request",
                field_name="started_at",
                raw_value=value,
                reason="timezone-aware datetime required",
            )
        return value.astimezone(timezone.utc)

    @staticmethod
    def _required_utc_timestamp(
        *,
        value: object,
        artifact_name: str,
        field_name: str,
        artifact_path: Path | None,
    ) -> datetime:
        if not isinstance(value, str) or not value.strip():
            raise CrawlAttemptTimestampError(
                artifact_name=artifact_name,
                field_name=field_name,
                raw_value=value,
                reason="non-empty ISO-8601 timestamp required",
                artifact_path=artifact_path,
            )

        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CrawlAttemptTimestampError(
                artifact_name=artifact_name,
                field_name=field_name,
                raw_value=value,
                reason="invalid ISO-8601 timestamp",
                artifact_path=artifact_path,
            ) from exc

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise CrawlAttemptTimestampError(
                artifact_name=artifact_name,
                field_name=field_name,
                raw_value=value,
                reason="persisted timestamp must include a timezone",
                artifact_path=artifact_path,
            )

        return parsed.astimezone(timezone.utc)

    @classmethod
    def _optional_utc_timestamp(
        cls,
        *,
        value: object,
        artifact_name: str,
        field_name: str,
        artifact_path: Path | None,
    ) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return cls._required_utc_timestamp(
            value=value,
            artifact_name=artifact_name,
            field_name=field_name,
            artifact_path=artifact_path,
        )

    @staticmethod
    def _validate_timestamp_order(
        *,
        started_at: datetime,
        completed_at: datetime | None,
        artifact_name: str,
        artifact_path: Path | None,
    ) -> None:
        if completed_at is None or completed_at >= started_at:
            return
        raise CrawlAttemptTimestampError(
            artifact_name=artifact_name,
            field_name="completed_at",
            raw_value=completed_at.isoformat(),
            reason="completed_at precedes started_at",
            artifact_path=artifact_path,
        )

    def _normalize_crawl_state_references(
        self,
        *,
        manifest: CrawlStateManifest,
    ) -> CrawlStateManifest:
        resolved = {
            "raw_run_directory": self._existing_runtime_path(
                manifest.raw_run_directory,
                field_name="raw_run_directory",
            ),
            "run_summary_path": self._existing_runtime_path(
                manifest.run_summary_path,
                field_name="run_summary_path",
            ),
            "previous_raw_run_directory": self._existing_runtime_path(
                manifest.previous_raw_run_directory,
                field_name="previous_raw_run_directory",
            ),
            "last_successful_manifest_path": self._existing_runtime_path(
                manifest.last_successful_manifest_path,
                field_name="last_successful_manifest_path",
            ),
            "last_successful_raw_run_directory": (
                self._existing_runtime_path(
                    manifest.last_successful_raw_run_directory,
                    field_name="last_successful_raw_run_directory",
                )
            ),
        }

        raw_link_exists = (
            resolved["raw_run_directory"] is not None
            and resolved["run_summary_path"] is not None
        )
        raw_run_id = manifest.raw_run_id if raw_link_exists else None
        crawl_session_id = (
            manifest.crawl_session_id if raw_link_exists else None
        )
        if not raw_link_exists:
            resolved["raw_run_directory"] = None
            resolved["run_summary_path"] = None

        last_successful_link_exists = (
            resolved["last_successful_manifest_path"] is not None
            and resolved["last_successful_raw_run_directory"] is not None
        )
        if not last_successful_link_exists:
            resolved["last_successful_manifest_path"] = None
            resolved["last_successful_raw_run_directory"] = None

        if (
            all(
                value == getattr(manifest, key)
                for key, value in resolved.items()
            )
            and raw_run_id == manifest.raw_run_id
            and crawl_session_id == manifest.crawl_session_id
        ):
            return manifest
        return replace(
            manifest,
            raw_run_id=raw_run_id,
            crawl_session_id=crawl_session_id,
            raw_run_directory=resolved["raw_run_directory"],
            run_summary_path=resolved["run_summary_path"],
            previous_raw_run_directory=resolved["previous_raw_run_directory"],
            last_successful_completed_at=(
                manifest.last_successful_completed_at
                if last_successful_link_exists
                else None
            ),
            last_successful_manifest_path=(
                resolved["last_successful_manifest_path"]
            ),
            last_successful_raw_run_directory=(
                resolved["last_successful_raw_run_directory"]
            ),
        )

    def _existing_runtime_path(
        self,
        value: Path | None,
        *,
        field_name: str,
    ) -> Path | None:
        if value is None:
            return None
        candidate = (
            value if value.is_absolute() else self._project_root / value
        )
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self._project_root)
        except ValueError as exc:
            raise ValueError(
                f"crawl-state {field_name} escapes project_root: {value}"
            ) from exc
        if resolved.exists():
            return resolved
        self._logger.warning(
            "workflow_crawl_state_stale_reference_ignored",
            field_name=field_name,
            manifest_path=str(
                self._artifact_path_registry.crawl_state_manifest_path()
            ),
            artifact_path=str(resolved),
        )
        return None
