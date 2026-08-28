"""Writer for crawl-state lifecycle manifests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from datachecker.manifests.artifact_manifest import format_manifest_path
from datachecker.manifests.crawl_attempt_artifacts import CrawlAttemptArtifacts
from datachecker.manifests.crawl_state_manifest import CrawlStateManifest
from datachecker.manifests.manifest_file_writer import (
    GenerateId,
    ManifestWriterBase,
    Now,
)
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus

if TYPE_CHECKING:
    from config.path_resolution.workflow_artifact_paths import (
        ArtifactPathRegistry,
    )
    from datachecker.manifests.artifact_manifest import RunArtifactIdentity
    from datachecker.manifests.crawl_manifest import CrawlManifest
    from datachecker.manifests.crawl_state_reference_resolver import (
        CrawlStateReferenceResolver,
    )
    from datachecker.manifests.manifest_file_writer import ManifestFileWriter
    from logger.project_logger import ProjectLogger


class CrawlStateManifestWriter(ManifestWriterBase):
    """Persist crawl state transitions and resolve active attempt artifacts."""

    def __init__(
        self,
        *,
        artifact_path_registry: ArtifactPathRegistry,
        reference_resolver: CrawlStateReferenceResolver,
        logger: ProjectLogger,
        project_root: Path | None,
        file_writer: ManifestFileWriter,
        artifact_identity: RunArtifactIdentity,
        now: Now,
        generate_id: GenerateId,
    ) -> None:
        super().__init__(
            artifact_path_registry=artifact_path_registry,
            logger=logger,
            project_root=project_root,
            file_writer=file_writer,
            artifact_identity=artifact_identity,
            now=now,
        )
        self._reference_resolver = reference_resolver
        self._generate_id = generate_id

    def _new_crawl_attempt_id(self) -> str:
        """Return a compact crawl-attempt identifier."""

        identifier = self._generate_id()
        return f"crawl_attempt_{identifier[:24]}"

    def crawl_state_manifest_path(self) -> Path:
        return Path(self._artifact_path_registry.crawl_state_manifest_path())

    def read_current_state(self) -> CrawlStateManifest | None:
        """Return the current crawl state manifest (or None)."""
        return self._reference_resolver.read_crawl_state()

    def existing_crawl_manifest(self) -> CrawlManifest | None:
        """Return the canonical crawl manifest when one is persisted."""

        return self._reference_resolver.existing_crawl_manifest()

    def write_crawl_state_started(
        self,
        *,
        source_registry_hash: str,
        crawl_settings_hash: str,
    ) -> CrawlStateManifest:
        now = self._utc_now_iso()
        previous = self._reference_resolver.read_crawl_state()
        if (
            previous is not None
            and previous.generation_id == self._artifact_identity.generation_id
            and previous.workflow_id == self._artifact_identity.workflow_id
            and previous.status
            in {
                WorkflowLifecycleStatus.RUNNING,
                WorkflowLifecycleStatus.RECOVERING,
            }
        ):
            raise RuntimeError(
                "active crawl attempt must be reconciled before a new "
                "attempt starts"
            )
        existing_manifest = self._reference_resolver.existing_crawl_manifest()
        previous_status = (
            previous.status
            if previous is not None
            else (
                WorkflowLifecycleStatus.COMPLETED
                if existing_manifest is not None
                else None
            )
        )
        return self._transition(
            status=WorkflowLifecycleStatus.RUNNING,
            now=now,
            previous=previous,
            previous_status=previous_status,
            source_registry_hash=source_registry_hash,
            crawl_settings_hash=crawl_settings_hash,
            attempt_id=self._new_crawl_attempt_id(),
            started_at=now,
            completed_at=None,
            raw_run_directory=None,
            run_summary_path=None,
            clear_raw_run_link=True,
            previous_raw_run_directory=(
                previous.raw_run_directory
                if previous is not None
                else (
                    existing_manifest.raw_run_directory
                    if existing_manifest is not None
                    else None
                )
            ),
            last_successful_completed_at=(
                previous.last_successful_completed_at
                if previous is not None
                else (
                    existing_manifest.completed_at
                    if existing_manifest is not None
                    else None
                )
            ),
            last_successful_manifest_path=(
                previous.last_successful_manifest_path
                if previous is not None
                else self._reference_resolver.existing_crawl_manifest_path()
            ),
            error_type=None,
            error_message=None,
        )

    def write_crawl_finalization_started(
        self,
        *,
        raw_run_directory: Path | None,
        run_summary_path: Path | None,
    ) -> CrawlStateManifest:
        now = self._utc_now_iso()
        previous = self._require_crawl_attempt()
        return self._transition(
            status=WorkflowLifecycleStatus.RUNNING,
            now=now,
            previous=previous,
            previous_status=previous.status,
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
            error_type=None,
            error_message=None,
        )

    def link_raw_run_to_attempt(
        self,
        *,
        attempt_id: str | None,
        raw_run_id: str,
        raw_run_directory: Path,
        run_summary_path: Path | None,
        generation_id: str,
        crawl_session_id: str,
        workflow_id: str,
    ) -> CrawlStateManifest | None:
        """P0.1: Immediately link DatasetWriter-created run dir to the current attempt.

        Called as soon as run directory is created, before/during crawl.
        Does not finalize; just binds attempt <-> raw run identity.
        """
        if (
            not attempt_id
            or not raw_run_id.strip()
            or raw_run_directory is None
            or run_summary_path is None
            or not generation_id.strip()
            or not crawl_session_id.strip()
            or not workflow_id.strip()
        ):
            raise ValueError("raw-run link requires complete run identity")
        now = self._utc_now_iso()
        previous = self._require_crawl_attempt()
        if previous.attempt_id != attempt_id:
            raise RuntimeError(
                "raw-run link attempt does not match crawl state"
            )
        if previous.generation_id != generation_id:
            raise RuntimeError(
                "raw-run link generation does not match crawl state"
            )
        if previous.workflow_id != workflow_id:
            raise RuntimeError(
                "raw-run link workflow does not match crawl state"
            )
        return self._transition(
            status=previous.status,
            now=now,
            previous=previous,
            previous_status=previous.status,
            attempt_id=attempt_id,
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
            raw_run_id=raw_run_id,
            crawl_session_id=crawl_session_id,
            error_type=None,
            error_message=None,
        )

    def write_crawl_state_succeeded(
        self,
        *,
        crawl_manifest: CrawlManifest,
    ) -> CrawlStateManifest:
        now = self._utc_now_iso()
        previous = self._reference_resolver.read_crawl_state()
        if previous is None:
            raise RuntimeError("crawl state must exist before completion")
        self._require_same_generation(previous)
        self._require_same_generation(crawl_manifest)
        return self._transition(
            status=WorkflowLifecycleStatus.COMPLETED,
            now=now,
            previous=previous,
            previous_status=previous.status,
            started_at=crawl_manifest.started_at,
            completed_at=crawl_manifest.completed_at,
            raw_run_directory=crawl_manifest.raw_run_directory,
            run_summary_path=crawl_manifest.run_summary_path,
            last_successful_completed_at=crawl_manifest.completed_at,
            last_successful_manifest_path=self._artifact_path_registry.crawl_manifest_path(),
            last_successful_raw_run_directory=crawl_manifest.raw_run_directory,
            error_type=None,
            error_message=None,
        )

    def write_crawl_state_failed(
        self,
        *,
        error_type: str,
        error_message: str,
        raw_run_directory: Path | None = None,
        run_summary_path: Path | None = None,
    ) -> CrawlStateManifest:
        return self._write_terminal_state(
            status=WorkflowLifecycleStatus.FAILED,
            error_type=error_type,
            error_message=error_message,
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
        )

    def write_crawl_state_cancelled(
        self,
        *,
        error_type: str,
        error_message: str,
        raw_run_directory: Path | None = None,
        run_summary_path: Path | None = None,
    ) -> CrawlStateManifest:
        return self._write_terminal_state(
            status=WorkflowLifecycleStatus.CANCELLED,
            error_type=error_type,
            error_message=error_message,
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
        )

    def write_crawl_state_abandoned(
        self,
        *,
        reason: str,
        raw_run_directory: Path | None = None,
        run_summary_path: Path | None = None,
    ) -> CrawlStateManifest:
        """Mark an interrupted attempt as abandoned and not promotable."""

        return self.write_crawl_state_failed(
            error_type="abandoned_interrupted_run",
            error_message=reason,
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
        )

    def write_crawl_state_recovering(
        self,
        *,
        raw_run_directory: Path | None,
        run_summary_path: Path | None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> CrawlStateManifest:
        """Record an interrupted crawl that can be reconciled safely."""

        now = self._utc_now_iso()
        previous = self._require_crawl_attempt()
        return self._transition(
            status=WorkflowLifecycleStatus.RECOVERING,
            now=now,
            previous=previous,
            previous_status=previous.status,
            completed_at=now,
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
            error_type=error_type,
            error_message=error_message,
            last_error_type=error_type,
            last_error_message=error_message,
        )

    def write_crawl_state_recovered(
        self,
        *,
        crawl_manifest: CrawlManifest | None = None,
        raw_run_directory: Path | None = None,
        run_summary_path: Path | None = None,
    ) -> CrawlStateManifest:
        """Mark a previously partial/recoverable state as recovered via repair or promotion."""
        now = self._utc_now_iso()
        previous = self._require_crawl_attempt()
        return self._transition(
            status=WorkflowLifecycleStatus.COMPLETED,
            now=now,
            previous=previous,
            previous_status=previous.status,
            completed_at=now,
            raw_run_directory=(
                raw_run_directory
                or (
                    crawl_manifest.raw_run_directory
                    if crawl_manifest
                    else None
                )
            ),
            run_summary_path=(
                run_summary_path
                or (
                    crawl_manifest.run_summary_path if crawl_manifest else None
                )
            ),
            last_successful_completed_at=now,
            last_successful_manifest_path=(
                self._artifact_path_registry.crawl_manifest_path()
                if crawl_manifest
                else previous.last_successful_manifest_path
            ),
            error_type=None,
            error_message=None,
        )

    def resolve_latest_crawl_attempt(self) -> CrawlAttemptArtifacts:
        state = self._reference_resolver.read_crawl_state()
        if state is None:
            return CrawlAttemptArtifacts(None, None)
        if (
            not state.attempt_id
            or not state.raw_run_id
            or not state.crawl_session_id
            or state.raw_run_directory is None
            or state.run_summary_path is None
            or not state.raw_run_directory.is_dir()
            or not state.run_summary_path.is_file()
        ):
            return CrawlAttemptArtifacts(None, None)
        return CrawlAttemptArtifacts(
            raw_run_directory=state.raw_run_directory,
            run_summary_path=state.run_summary_path,
        )

    def write_crawl_state_incomplete(
        self,
        *,
        error_type: str,
        error_message: str,
        raw_run_directory: Path | None = None,
        run_summary_path: Path | None = None,
    ) -> CrawlStateManifest:
        return self._write_nonterminal_state(
            status=WorkflowLifecycleStatus.INCOMPLETE,
            error_type=error_type,
            error_message=error_message,
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
        )

    def write_crawl_state_resumable(
        self,
        *,
        error_type: str,
        error_message: str,
        raw_run_directory: Path | None = None,
        run_summary_path: Path | None = None,
    ) -> CrawlStateManifest:
        return self._write_nonterminal_state(
            status=WorkflowLifecycleStatus.RESUMABLE,
            error_type=error_type,
            error_message=error_message,
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
        )

    def _transition(
        self,
        *,
        status: WorkflowLifecycleStatus,
        now: str | None = None,
        previous: CrawlStateManifest | None = None,
        previous_status: WorkflowLifecycleStatus | None = None,
        source_registry_hash: str | None = None,
        crawl_settings_hash: str | None = None,
        attempt_id: str | None = None,
        started_at: str | None = None,
        updated_at: str | None = None,
        completed_at: str | None = None,
        raw_run_directory: Path | None = None,
        run_summary_path: Path | None = None,
        previous_raw_run_directory: Path | None = None,
        last_successful_completed_at: str | None = None,
        last_successful_manifest_path: Path | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        last_error_type: str | None = None,
        last_error_message: str | None = None,
        last_successful_raw_run_directory: Path | None = None,
        raw_run_id: str | None = None,
        crawl_session_id: str | None = None,
        clear_raw_run_link: bool = False,
    ) -> CrawlStateManifest:
        """Centralized state transition helper. All status changes go through here."""
        if now is None:
            now = self._utc_now_iso()
        if previous is None:
            previous = self._require_crawl_attempt()

        manifest = CrawlStateManifest(
            **self._identity_fields(),
            status=status,
            attempt_id=attempt_id
            or (previous.attempt_id if previous else None),
            started_at=started_at or previous.started_at,
            updated_at=updated_at or now,
            completed_at=completed_at,
            raw_run_directory=(
                None
                if clear_raw_run_link
                else raw_run_directory or previous.raw_run_directory
            ),
            run_summary_path=(
                None
                if clear_raw_run_link
                else run_summary_path or previous.run_summary_path
            ),
            previous_status=previous_status
            or (previous.status if previous else None),
            previous_raw_run_directory=previous_raw_run_directory
            or previous.previous_raw_run_directory,
            last_successful_completed_at=last_successful_completed_at
            or previous.last_successful_completed_at,
            last_successful_manifest_path=(
                last_successful_manifest_path
                or previous.last_successful_manifest_path
            ),
            error_type=error_type,
            error_message=error_message,
            source_registry_hash=source_registry_hash
            or previous.source_registry_hash,
            crawl_settings_hash=crawl_settings_hash
            or previous.crawl_settings_hash,
            last_error_type=last_error_type or error_type,
            last_error_message=last_error_message or error_message,
            last_transition_at=now,
            last_successful_raw_run_directory=(
                last_successful_raw_run_directory
                or previous.last_successful_raw_run_directory
            ),
            raw_run_id=(
                None
                if clear_raw_run_link
                else raw_run_id or previous.raw_run_id
            ),
            crawl_session_id=(
                None
                if clear_raw_run_link
                else crawl_session_id or previous.crawl_session_id
            ),
        )
        self._write_crawl_state_and_log(
            manifest=manifest,
            error_type=error_type,
            error_message=error_message,
        )
        return manifest

    def _write_terminal_state(
        self,
        *,
        status: WorkflowLifecycleStatus,
        error_type: str,
        error_message: str,
        raw_run_directory: Path | None,
        run_summary_path: Path | None,
    ) -> CrawlStateManifest:
        now = self._utc_now_iso()
        previous = self._require_crawl_attempt()
        return self._transition(
            status=status,
            now=now,
            previous=previous,
            previous_status=previous.status,
            completed_at=now,  # P0.5: ensure terminal state has completed_at
            error_type=error_type,
            error_message=error_message,
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
            last_error_type=error_type,
            last_error_message=error_message,
        )

    def _require_crawl_attempt(self) -> CrawlStateManifest:
        manifest = self._reference_resolver.read_crawl_state()
        if manifest is not None:
            self._require_same_generation(manifest)
            return manifest
        now = self._utc_now_iso()
        identity = self._artifact_identity
        return CrawlStateManifest(
            generation_id=identity.generation_id,
            workflow_id=identity.workflow_id,
            project_fingerprint=identity.project_fingerprint,
            config_fingerprint=identity.config_fingerprint,
            environment_name=identity.environment_name,
            environment_fingerprint=identity.environment_fingerprint,
            python_version=identity.python_version,
            dependency_lock_fingerprint=(identity.dependency_lock_fingerprint),
            status=WorkflowLifecycleStatus.RUNNING,
            attempt_id=self._new_crawl_attempt_id(),
            started_at=now,
            updated_at=now,
            completed_at=None,
            raw_run_directory=None,
            run_summary_path=None,
            previous_status=None,
            previous_raw_run_directory=None,
            last_successful_completed_at=(
                self._reference_resolver.existing_crawl_manifest_completed_at()
            ),
            last_successful_manifest_path=(
                self._reference_resolver.existing_crawl_manifest_path()
            ),
            error_type=None,
            error_message=None,
            source_registry_hash=None,
            crawl_settings_hash=None,
            raw_run_id=None,
            crawl_session_id=None,
        )

    def _write_nonterminal_state(
        self,
        *,
        status: WorkflowLifecycleStatus,
        error_type: str,
        error_message: str,
        raw_run_directory: Path | None,
        run_summary_path: Path | None,
    ) -> CrawlStateManifest:
        now = self._utc_now_iso()
        previous = self._require_crawl_attempt()
        return self._transition(
            status=status,
            now=now,
            previous=previous,
            previous_status=previous.status,
            completed_at=None,
            error_type=error_type,
            error_message=error_message,
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
            last_error_type=error_type,
            last_error_message=error_message,
        )

    def _write_crawl_state_and_log(
        self,
        *,
        manifest: CrawlStateManifest,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        manifest_path = self.crawl_state_manifest_path()
        self._write_manifest(path=manifest_path, payload=manifest.to_payload())
        log_fields = {
            "status": manifest.status,
            "attempt_id": manifest.attempt_id,
            "manifest_path": format_manifest_path(manifest_path),
            "raw_run_directory": format_manifest_path(
                manifest.raw_run_directory
            ),
            "error_type": error_type,
            "error_message": error_message,
        }
        if error_type is None:
            self._logger.info("workflow_crawl_status_written", **log_fields)
            return
        self._logger.warning("workflow_crawl_status_written", **log_fields)
