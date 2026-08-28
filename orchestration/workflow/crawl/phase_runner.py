"""Crawl phase execution for the autonomous data workflow."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Protocol, cast

from config.collection.processors import PageProcessorSettings
from crawler.runtime.loop.crawl_run_summary import CrawlTerminalOutcome
from datachecker.manifests.crawl_attempt_artifacts import CrawlAttemptArtifacts
from datachecker.manifests.crawl_manifest import CrawlManifest
from datachecker.manifests.crawl_manifest_writer import (
    CorruptCrawlManifestError,
)
from datachecker.manifests.crawl_state_manifest import CrawlStateManifest
from datachecker.manifests.crawl_state_manifest_writer import (
    CrawlStateManifestWriter,
)
from datachecker.workflow_decision import WorkflowExecutionPlan
from logger.project_logger import ProjectLogger
from orchestration.errors import StartupConfigurationError
from orchestration.workflow.phase import PhaseOutcome, PhaseStatus, RunBlocking


class CrawlExecutionResult(Protocol):
    """Structural result the workflow needs from one crawl application run."""

    @property
    def dataset_outcome(self) -> CrawlTerminalOutcome: ...


class ExecuteCrawl(Protocol):
    """Bound crawl-application lifecycle owned by the bootstrap layer."""

    def __call__(
        self,
        *,
        crawl_attempt_id: str,
        crawl_state_manifest_writer: CrawlStateManifestWriter,
        page_settings: PageProcessorSettings,
    ) -> Awaitable[CrawlExecutionResult]: ...


class MissingByMediaKind(Protocol):
    """Projects coverage gaps onto per-media-kind deficits."""

    def __call__(
        self,
        coverage_gaps: dict[str, int],
    ) -> dict[str, int]: ...


class ResolveFocusKinds(Protocol):
    """Resolves the ordered focus kinds for one deficit report."""

    def __call__(
        self,
        *,
        missing_by_kind: dict[str, int],
    ) -> tuple[str, ...]: ...


class ResolveFocusedPageSettings(Protocol):
    """Resolves the focused page-discovery policy for one focus selection."""

    def __call__(
        self,
        *,
        focus_kinds: tuple[str, ...],
    ) -> PageProcessorSettings: ...


class CrawlPromotion(Protocol):
    """Commits one complete raw crawl attempt to canonical state."""

    def __call__(
        self,
        *,
        state: CrawlStateManifest,
        attempt: CrawlAttemptArtifacts,
    ) -> CrawlManifest: ...


class CrawlPhaseRunner:
    """Execute crawl workflow phases and collect structured multimodal."""

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        run_blocking: RunBlocking,
        io_timeout_seconds: float,
        source_registry_hash: str,
        crawl_settings_hash: str,
        missing_by_media_kind: MissingByMediaKind,
        resolve_focus_kinds: ResolveFocusKinds,
        resolve_focused_page_settings: ResolveFocusedPageSettings,
        crawl_state_manifest_writer: CrawlStateManifestWriter,
        commit_crawl: CrawlPromotion,
        execute_crawl: ExecuteCrawl,
    ) -> None:
        self._logger = logger
        self._run_blocking = run_blocking
        self._io_timeout_seconds = io_timeout_seconds
        self._source_registry_hash = source_registry_hash
        self._crawl_settings_hash = crawl_settings_hash
        self._missing_by_media_kind = missing_by_media_kind
        self._resolve_focus_kinds = resolve_focus_kinds
        self._resolve_focused_page_settings = resolve_focused_page_settings
        self._crawl_state_manifest_writer = crawl_state_manifest_writer
        self._commit_crawl = commit_crawl
        self._execute_crawl = execute_crawl

    async def run(
        self,
        plan: WorkflowExecutionPlan,
    ) -> PhaseOutcome:
        """Execute one crawl decision."""

        crawl_state = await self._run_blocking(
            self._crawl_state_manifest_writer.write_crawl_state_started,
            source_registry_hash=self._source_registry_hash,
            crawl_settings_hash=self._crawl_settings_hash,
            timeout_seconds=self._io_timeout_seconds,
        )

        missing_by_kind = self._missing_by_media_kind(plan.coverage_gaps)
        focus_kinds_ordered = self._resolve_focus_kinds(
            missing_by_kind=missing_by_kind,
        )
        page_settings = self._resolve_focused_page_settings(
            focus_kinds=focus_kinds_ordered,
        )

        self._logger.info(
            "crawl_application_starting",
            crawl_attempt_id=crawl_state.attempt_id,
            workflow_reason=plan.reason.value,
            coverage_gaps=plan.coverage_gaps,
            focus_kinds=focus_kinds_ordered,
            workflow_missing_by_kind=missing_by_kind,
        )

        try:
            run_result = await self._execute_crawl(
                crawl_attempt_id=crawl_state.attempt_id,
                crawl_state_manifest_writer=(
                    self._crawl_state_manifest_writer
                ),
                page_settings=page_settings,
            )

        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            try:
                await self.record_crawl_failure(
                    exc,
                    cancelled=True,
                )
            except Exception as state_error:
                self._logger.error(
                    "crawl_cancelled_state_write_failed",
                    error_type=type(state_error).__name__,
                    error_message=str(state_error) or None,
                )

            return PhaseOutcome(status=PhaseStatus.CANCELLED)

        except StartupConfigurationError as exc:
            self._logger.exception(
                "crawl_phase_failed",
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )

            try:
                await self.record_crawl_failure(
                    exc,
                    cancelled=False,
                )
            except Exception as state_error:
                self._logger.error(
                    "crawl_failed_state_write_failed",
                    error_type=type(state_error).__name__,
                    error_message=str(state_error) or None,
                )

            raise

        except Exception as exc:
            self._logger.exception(
                "crawl_phase_failed",
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )

            try:
                await self.record_crawl_failure(
                    exc,
                    cancelled=False,
                )
            except Exception as state_error:
                self._logger.error(
                    "crawl_failed_state_write_failed",
                    error_type=type(state_error).__name__,
                    error_message=str(state_error) or None,
                )

            return PhaseOutcome(status=PhaseStatus.FAILED)

        if run_result.dataset_outcome is CrawlTerminalOutcome.SUCCESS:
            try:
                await asyncio.shield(
                    self._finalize_successful_crawl(
                        timeout=self._io_timeout_seconds,
                    )
                )

            except CorruptCrawlManifestError as exc:
                self._logger.exception(
                    "crawl_manifest_corruption_aborted",
                    recovery_rules="quarantine_and_abort",
                    manifest_path=str(exc.manifest_path),
                    quarantine_path=(
                        str(exc.quarantine_path)
                        if exc.quarantine_path is not None
                        else None
                    ),
                    corruption_type=exc.corruption_type,
                    quarantine_error=exc.quarantine_error,
                )

                try:
                    await self.record_crawl_failure(
                        exc,
                        cancelled=False,
                    )
                except Exception as state_error:
                    self._logger.error(
                        "crawl_manifest_corruption_state_write_failed",
                        error_type=type(state_error).__name__,
                        error_message=str(state_error) or None,
                    )

                return PhaseOutcome(status=PhaseStatus.FAILED)

            return PhaseOutcome(status=PhaseStatus.SUCCEEDED)

        if run_result.dataset_outcome is CrawlTerminalOutcome.CANCELLED:
            status = PhaseStatus.CANCELLED
        elif run_result.dataset_outcome is CrawlTerminalOutcome.INCOMPLETE:
            status = PhaseStatus.INCOMPLETE
        else:
            status = PhaseStatus.FAILED

        await self._persist_non_success_state(
            status=status,
            error=run_result.dataset_outcome.value,
            timeout=self._io_timeout_seconds,
        )

        return PhaseOutcome(status=status)

    async def _persist_non_success_state(
        self,
        *,
        status: PhaseStatus,
        error: str,
        timeout: float,
    ) -> None:
        """Persist the terminal state of an unsuccessful crawl."""

        latest = await self._run_blocking(
            self._crawl_state_manifest_writer.resolve_latest_crawl_attempt,
            timeout_seconds=timeout,
        )

        if status is PhaseStatus.CANCELLED:
            await self._run_blocking(
                self._crawl_state_manifest_writer.write_crawl_state_cancelled,
                error_type=status.value,
                error_message=error,
                raw_run_directory=latest.raw_run_directory,
                run_summary_path=latest.run_summary_path,
                timeout_seconds=timeout,
            )
            return

        if status is PhaseStatus.INCOMPLETE:
            await self._run_blocking(
                self._crawl_state_manifest_writer.write_crawl_state_incomplete,
                error_type=status.value,
                error_message=error,
                raw_run_directory=latest.raw_run_directory,
                run_summary_path=latest.run_summary_path,
                timeout_seconds=timeout,
            )
            return

        await self._run_blocking(
            self._crawl_state_manifest_writer.write_crawl_state_failed,
            error_type=status.value,
            error_message=error,
            raw_run_directory=latest.raw_run_directory,
            run_summary_path=latest.run_summary_path,
            timeout_seconds=timeout,
        )

    async def _finalize_successful_crawl(
        self,
        *,
        timeout: float,
    ) -> CrawlManifest:
        """Commit a complete raw crawl through the canonical committer."""

        crawl_state = await self._run_blocking(
            self._crawl_state_manifest_writer.read_current_state,
            timeout_seconds=timeout,
        )
        if crawl_state is None:
            raise RuntimeError("Crawl state is missing during finalization.")

        latest_attempt = await self._run_blocking(
            self._crawl_state_manifest_writer.resolve_latest_crawl_attempt,
            timeout_seconds=timeout,
        )
        return cast(
            CrawlManifest,
            await self._run_blocking(
                self._commit_crawl,
                state=crawl_state,
                attempt=latest_attempt,
                timeout_seconds=timeout,
            ),
        )

    async def record_crawl_failure(
        self,
        exc: BaseException,
        *,
        cancelled: bool,
    ) -> None:
        """Persist crawler failure or cancellation state."""

        latest_attempt = await self._run_blocking(
            self._crawl_state_manifest_writer.resolve_latest_crawl_attempt,
            timeout_seconds=self._io_timeout_seconds,
        )

        try:
            if cancelled:
                await self._run_blocking(
                    self._crawl_state_manifest_writer.write_crawl_state_cancelled,
                    error_type=type(exc).__name__,
                    error_message=str(exc) or None,
                    raw_run_directory=(latest_attempt.raw_run_directory),
                    run_summary_path=(latest_attempt.run_summary_path),
                    timeout_seconds=self._io_timeout_seconds,
                )
            else:
                await self._run_blocking(
                    self._crawl_state_manifest_writer.write_crawl_state_failed,
                    error_type=type(exc).__name__,
                    error_message=str(exc) or None,
                    raw_run_directory=(latest_attempt.raw_run_directory),
                    run_summary_path=(latest_attempt.run_summary_path),
                    timeout_seconds=self._io_timeout_seconds,
                )

        except Exception as state_error:
            self._logger.error(
                "crawl_failure_state_write_error",
                cancelled=cancelled,
                primary_error_type=type(exc).__name__,
                state_error_type=type(state_error).__name__,
                state_error_message=str(state_error) or None,
            )
