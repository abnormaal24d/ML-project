"""Manifest model for crawl lifecycle status between workflow runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datachecker.manifests.artifact_manifest import ArtifactManifest
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus


@dataclass(frozen=True, slots=True)
class CrawlStateManifest(ArtifactManifest):
    """Persist crawl lifecycle state for clearer recovery and logger."""

    status: WorkflowLifecycleStatus
    attempt_id: str | None
    started_at: str | None
    updated_at: str | None
    completed_at: str | None
    raw_run_directory: Path | None
    run_summary_path: Path | None
    previous_status: WorkflowLifecycleStatus | None
    previous_raw_run_directory: Path | None
    last_successful_completed_at: str | None
    last_successful_manifest_path: Path | None
    error_type: str | None
    error_message: str | None
    source_registry_hash: str | None = None
    crawl_settings_hash: str | None = None
    last_error_type: str | None = None
    last_error_message: str | None = None
    last_transition_at: str | None = None
    last_successful_raw_run_directory: Path | None = None
    raw_run_id: str | None = None
    crawl_session_id: str | None = None

    def __post_init__(self) -> None:
        ArtifactManifest.__post_init__(self)
        if self.status.terminal and not self.completed_at:
            raise ValueError("terminal crawl state requires completed_at")
        if (
            self.status is WorkflowLifecycleStatus.RUNNING
            and self.completed_at
        ):
            raise ValueError("running crawl state must not have completed_at")
        linked_fields = (
            self.raw_run_id,
            self.raw_run_directory,
            self.run_summary_path,
            self.crawl_session_id,
        )
        if any(value is not None for value in linked_fields) and not all(
            value is not None for value in linked_fields
        ):
            raise ValueError(
                "raw-run state link must contain every identity field"
            )

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
    ) -> CrawlStateManifest:
        """Build a manifest instance from JSON payload data."""

        status = WorkflowLifecycleStatus.parse(payload.get("status"))

        previous_status_raw = cls.as_opt_str(payload.get("previous_status"))
        previous_status = (
            WorkflowLifecycleStatus.parse(
                previous_status_raw,
                field="previous_status",
            )
            if previous_status_raw is not None
            else None
        )

        return cls(
            **cls.identity_from_payload(payload),
            status=status,
            attempt_id=cls.as_opt_str(payload.get("attempt_id")),
            started_at=cls.as_opt_str(payload.get("started_at")),
            updated_at=cls.as_opt_str(payload.get("updated_at")),
            completed_at=cls.as_opt_str(payload.get("completed_at")),
            raw_run_directory=cls.as_opt_path(
                payload.get("raw_run_directory")
            ),
            run_summary_path=cls.as_opt_path(payload.get("run_summary_path")),
            previous_status=previous_status,
            previous_raw_run_directory=cls.as_opt_path(
                payload.get("previous_raw_run_directory")
            ),
            last_successful_completed_at=cls.as_opt_str(
                payload.get("last_successful_completed_at")
            ),
            last_successful_manifest_path=cls.as_opt_path(
                payload.get("last_successful_manifest_path")
            ),
            error_type=cls.as_opt_str(payload.get("error_type")),
            error_message=cls.as_opt_str(payload.get("error_message")),
            source_registry_hash=cls.as_opt_str(
                payload.get("source_registry_hash")
            ),
            crawl_settings_hash=cls.as_opt_str(
                payload.get("crawl_settings_hash")
            ),
            last_error_type=cls.as_opt_str(payload.get("last_error_type")),
            last_error_message=cls.as_opt_str(
                payload.get("last_error_message")
            ),
            last_transition_at=cls.as_opt_str(
                payload.get("last_transition_at")
            ),
            last_successful_raw_run_directory=cls.as_opt_path(
                payload.get("last_successful_raw_run_directory")
            ),
            raw_run_id=cls.as_opt_str(payload.get("raw_run_id")),
            crawl_session_id=cls.as_opt_str(payload.get("crawl_session_id")),
        )

    @property
    def is_terminal(self) -> bool:
        """True for terminal states where no further transitions are expected."""
        return bool(self.status.terminal)

    @property
    def is_recoverable(self) -> bool:
        """True for states that may be recoverable via repair or promotion."""
        return bool(self.status.recoverable)
