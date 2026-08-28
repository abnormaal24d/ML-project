"""Build, run, and shut down the crawler application."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from crawler.runtime.loop.crawl_run_summary import CrawlTerminalOutcome
from logger.log_context import bind_log_context
from orchestration.bootstrap.run_context import (
    RunContext,
    create_run_context,
)
from orchestration.errors import (
    BootstrapBuildFailure,
    ExecutionError,
    LifecycleError,
    ShutdownError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from config.collection.processors import PageProcessorSettings
    from config.settings.root import Settings
    from crawler.runtime.loop.crawl_run_summary import CrawlRunResult
    from datachecker.manifests.crawl_state_manifest_writer import (
        CrawlStateManifestWriter,
    )
    from orchestration.bootstrap.container import ApplicationContainer
    from orchestration.settings_loader import RuntimeReadiness


SUCCESS_EXIT_CODE: Final[int] = 0
FAILURE_EXIT_CODE: Final[int] = 1
INTERRUPTED_EXIT_CODE: Final[int] = 130
CANCELLED_EXIT_CODE: Final[int] = 131


@dataclass(frozen=True, slots=True)
class ApplicationRunResult:
    """Execution and shutdown result from one crawler run."""

    exit_code: int
    crawler_result: CrawlRunResult | None = None
    dataset_outcome: CrawlTerminalOutcome = CrawlTerminalOutcome.FAILED
    execution_error: BaseException | None = None
    shutdown_error: BaseException | None = None
    execution_duration_seconds: float = 0.0
    shutdown_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0


async def execute_application(
    *,
    project_root: Path,
    environment: str,
    stage: str,
    settings: Settings | None = None,
    configure_logging: bool = True,
    parent_run_context: RunContext | None = None,
    crawl_attempt_id: str | None = None,
    crawl_state_manifest_writer: CrawlStateManifestWriter | None = None,
    config_root: Path | None = None,
    runtime_readiness: RuntimeReadiness | None = None,
    page_settings_override: "PageProcessorSettings | None" = None,
) -> ApplicationRunResult:
    """Build and execute one crawler application lifecycle."""

    run_context = create_run_context(
        stage=stage,
        parent=parent_run_context,
    )

    try:
        from orchestration.bootstrap.container import (
            build_application_container,
        )

        container = build_application_container(
            project_root=project_root,
            environment=environment,
            settings=settings,
            configure_logging=configure_logging,
            run_context=run_context,
            crawl_attempt_id=crawl_attempt_id,
            crawl_state_manifest_writer=crawl_state_manifest_writer,
            config_root=config_root,
            runtime_readiness=runtime_readiness,
            page_settings_override=page_settings_override,
        )

    except BootstrapBuildFailure as exc:
        if exc.shutdown_manager is not None:
            try:
                await exc.shutdown_manager.aclose()
            except ShutdownError as shutdown_error:
                raise LifecycleError(
                    "bootstrap build and partial shutdown failed",
                    stage="bootstrap",
                    component="container",
                    primary_error=exc.build_error,
                    secondary_error=shutdown_error,
                ) from shutdown_error

        raise exc.build_error from exc

    try:
        with bind_log_context(container.logger_factory.base_context):
            run_result = await run_application(container)

    except BaseException as execution_boundary_error:
        try:
            await container.aclose()
        except BaseException as final_close_error:
            raise LifecycleError(
                "application boundary and final shutdown failed",
                stage="lifecycle",
                component="application",
                primary_error=execution_boundary_error,
                secondary_error=final_close_error,
            ) from final_close_error

        raise

    if run_result.execution_error is not None:
        if run_result.shutdown_error is not None:
            raise LifecycleError(
                "application execution and shutdown failed",
                stage="lifecycle",
                component="application",
                primary_error=run_result.execution_error,
                secondary_error=run_result.shutdown_error,
            ) from run_result.shutdown_error

        if isinstance(
            run_result.execution_error,
            (KeyboardInterrupt, asyncio.CancelledError),
        ):
            raise run_result.execution_error

        raise ExecutionError(
            cause=run_result.execution_error,
        ) from run_result.execution_error

    if run_result.shutdown_error is not None:
        raise run_result.shutdown_error

    return run_result


async def run_application(
    container: ApplicationContainer,
) -> ApplicationRunResult:
    """Run the crawler and always attempt an orderly shutdown."""

    logger = container.logger_factory.get_logger(__name__)
    started_at = time.monotonic()
    execution_started_at = time.monotonic()

    crawler_result: CrawlRunResult | None = None
    execution_error: BaseException | None = None
    shutdown_error: BaseException | None = None
    dataset_outcome = CrawlTerminalOutcome.FAILED
    execution_duration = 0.0
    shutdown_duration = 0.0

    logger.debug("application_runtime_started")
    logger.debug("application_runtime_execution_started")

    try:
        try:
            crawler_result = await container.crawler.crawl()

        except KeyboardInterrupt as exc:
            logger.info(
                "application_runtime_execution_interrupted",
                interruption_type="keyboard_interrupt",
            )
            execution_error = exc

        except asyncio.CancelledError as exc:
            logger.info(
                "application_runtime_execution_interrupted",
                interruption_type="cancelled",
            )
            execution_error = exc

        except (RuntimeError, OSError, ValueError) as exc:
            logger.exception(
                "application_runtime_execution_failed",
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )
            execution_error = exc

        else:
            logger.info(
                "application_runtime_execution_completed",
                stop_trigger=getattr(
                    crawler_result,
                    "stop_trigger",
                    None,
                ),
                terminal_outcome=getattr(
                    crawler_result,
                    "terminal_outcome",
                    None,
                ),
                crawler_completed_tasks=getattr(
                    crawler_result,
                    "completed_tasks",
                    None,
                ),
            )

        execution_duration = time.monotonic() - execution_started_at

        dataset_outcome = _map_terminal_outcome(
            crawler_result=crawler_result,
            execution_error=execution_error,
        )

        terminal_reason, terminal_details = _dataset_terminal_context(
            crawler_result=crawler_result,
            execution_error=execution_error,
            outcome=dataset_outcome,
        )

        try:
            await _apply_dataset_outcome(
                container=container,
                outcome=dataset_outcome,
                crawler_result=crawler_result,
                terminal_reason=terminal_reason,
                terminal_details=terminal_details,
            )

        except (RuntimeError, OSError, ValueError) as exc:
            logger.exception(
                "application_dataset_outcome_failed",
                outcome=dataset_outcome.value,
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )
            execution_error = execution_error or exc
            dataset_outcome = CrawlTerminalOutcome.FAILED

    finally:
        shutdown_started_at = time.monotonic()

        logger.debug("application_runtime_shutdown_started")

        try:
            await container.aclose()

        except ShutdownError as exc:
            logger.exception(
                "application_runtime_shutdown_failed",
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )
            shutdown_error = exc

        except Exception as exc:
            logger.exception(
                "application_runtime_shutdown_unhandled_exception",
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )
            shutdown_error = exc

        else:
            logger.info("application_runtime_shutdown_completed")

        shutdown_duration = time.monotonic() - shutdown_started_at

    if isinstance(execution_error, KeyboardInterrupt):
        exit_code = INTERRUPTED_EXIT_CODE
    elif isinstance(
        execution_error,
        asyncio.CancelledError,
    ):
        exit_code = CANCELLED_EXIT_CODE
    elif (
        execution_error is not None
        or shutdown_error is not None
        or dataset_outcome is not CrawlTerminalOutcome.SUCCESS
    ):
        exit_code = FAILURE_EXIT_CODE
    else:
        exit_code = SUCCESS_EXIT_CODE

    total_duration = time.monotonic() - started_at
    primary_error = execution_error or shutdown_error

    (logger.info if exit_code == SUCCESS_EXIT_CODE else logger.error)(
        (
            "application_runtime_completed"
            if exit_code == SUCCESS_EXIT_CODE
            else "application_runtime_failed"
        ),
        exit_code=exit_code,
        execution_outcome=(
            "failed" if execution_error is not None else "completed"
        ),
        shutdown_outcome=(
            "failed" if shutdown_error is not None else "completed"
        ),
        execution_error_type=(
            type(execution_error).__name__
            if execution_error is not None
            else None
        ),
        execution_error_message=(
            str(execution_error) or None
            if execution_error is not None
            else None
        ),
        shutdown_error_type=(
            type(shutdown_error).__name__
            if shutdown_error is not None
            else None
        ),
        shutdown_error_message=(
            str(shutdown_error) or None if shutdown_error is not None else None
        ),
        execution_duration_seconds=round(
            execution_duration,
            3,
        ),
        shutdown_duration_seconds=round(
            shutdown_duration,
            3,
        ),
        total_duration_seconds=round(
            total_duration,
            3,
        ),
        exc_info=(primary_error if exit_code != SUCCESS_EXIT_CODE else None),
    )

    return ApplicationRunResult(
        exit_code=exit_code,
        crawler_result=crawler_result,
        dataset_outcome=dataset_outcome,
        execution_error=execution_error,
        shutdown_error=shutdown_error,
        execution_duration_seconds=execution_duration,
        shutdown_duration_seconds=shutdown_duration,
        total_duration_seconds=total_duration,
    )


def _map_terminal_outcome(
    *,
    crawler_result: CrawlRunResult | None,
    execution_error: BaseException | None,
) -> CrawlTerminalOutcome:
    """Map crawler result and execution error to one terminal outcome."""

    if isinstance(
        execution_error,
        (KeyboardInterrupt, asyncio.CancelledError),
    ):
        return CrawlTerminalOutcome.CANCELLED

    if execution_error is not None or crawler_result is None:
        return CrawlTerminalOutcome.FAILED

    return crawler_result.terminal_outcome


def _dataset_terminal_context(
    *,
    crawler_result: CrawlRunResult | None,
    execution_error: BaseException | None,
    outcome: CrawlTerminalOutcome,
) -> tuple[str | None, dict[str, object]]:
    """Return terminal reason and persistence details."""

    if outcome is CrawlTerminalOutcome.SUCCESS:
        return None, {}

    details: dict[str, object] = {}
    stop_trigger: str | None = None

    if crawler_result is not None:
        stop_trigger = crawler_result.stop_trigger.value

        if stop_trigger is not None:
            details["crawl_stop_trigger"] = stop_trigger

        if crawler_result.unmet_requirements:
            details["unmet_requirements"] = list(
                crawler_result.unmet_requirements
            )

    if execution_error is not None:
        details["error_type"] = type(execution_error).__name__
        details["error_message"] = str(execution_error) or None

        if isinstance(
            execution_error,
            KeyboardInterrupt,
        ):
            return "keyboard_interrupt", details

        if isinstance(
            execution_error,
            asyncio.CancelledError,
        ):
            return "asyncio_cancelled", details

        return "crawler_execution_failed", details

    if stop_trigger is not None:
        return stop_trigger, details

    if outcome is CrawlTerminalOutcome.CANCELLED:
        return "crawler_cancelled", details

    if outcome is CrawlTerminalOutcome.INCOMPLETE:
        return "crawl_output_unready", details

    return "crawler_failed", details


async def _apply_dataset_outcome(
    *,
    container: ApplicationContainer,
    outcome: CrawlTerminalOutcome,
    crawler_result: CrawlRunResult | None,
    terminal_reason: str | None,
    terminal_details: dict[str, object] | None,
) -> None:
    """Persist exactly one dataset lifecycle outcome."""

    writer: Any = container.dataset_writer

    if outcome is CrawlTerminalOutcome.SUCCESS:
        await writer.commit_completed(
            crawler_result=crawler_result,
        )
        return

    if outcome is CrawlTerminalOutcome.CANCELLED:
        await writer.mark_cancelled(
            reason=terminal_reason,
            details=terminal_details,
        )
        return

    if outcome is CrawlTerminalOutcome.INCOMPLETE:
        await writer.mark_incomplete(
            reason=terminal_reason,
            details=terminal_details,
        )
        return

    await writer.mark_failed(
        reason=terminal_reason,
        details=terminal_details,
    )
