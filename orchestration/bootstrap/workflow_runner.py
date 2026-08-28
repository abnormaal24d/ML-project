"""Execute the autonomous workflow lifecycle."""

from __future__ import annotations

import asyncio
import json
import signal
from typing import TYPE_CHECKING

from config.path_resolution.project_paths import ProjectPaths
from config.path_resolution.workflow_artifact_paths import (
    ArtifactPathRegistry,
)
from datachecker.manifests.crawl_state_manifest import CrawlStateManifest
from orchestration.bootstrap.application import (
    INTERRUPTED_EXIT_CODE,
)
from orchestration.bootstrap.run_context import (
    RunContext,
    create_run_context,
    resume_run_context,
)
from orchestration.bootstrap.shutdown import (
    install_signal_handlers,
)
from orchestration.bootstrap.workflow_lock import workflow_file_lock
from orchestration.settings_loader import validate_runtime_configuration

if TYPE_CHECKING:
    from config.settings.root import Settings
    from orchestration.cli.argument_parser import RuntimeOptions
    from orchestration.settings_loader import RuntimeReadiness


def _resolve_workflow_context(
    *,
    settings: Settings,
    resume: bool,
    fresh_run: bool,
) -> RunContext:
    """Create a new workflow identity or resume recoverable state."""

    if fresh_run or not resume:
        return create_run_context(
            stage="data_workflow",
        )

    state_path = ProjectPaths(
        project_root=settings.paths.root,
    ).resolve(
        ArtifactPathRegistry(
            settings=settings.collection.datachecker,
            dataset_paths=settings.datasets.paths,
        ).crawl_state_manifest_path()
    )

    crawl_state = None
    if state_path.is_file():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            crawl_state = CrawlStateManifest.from_payload(payload)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(
                f"existing workflow state is unreadable or corrupt: {state_path}"
            ) from exc

    if crawl_state is None or not crawl_state.is_recoverable:
        return create_run_context(
            stage="data_workflow",
        )

    return resume_run_context(
        stage="data_workflow",
        workflow_id=crawl_state.workflow_id,
        generation_id=crawl_state.generation_id,
    )


def execute_workflow_command(
    *,
    options: RuntimeOptions,
    settings: Settings,
    runtime_readiness: RuntimeReadiness | None = None,
) -> int:
    """Build and execute one complete DataChecker-driven workflow."""

    readiness = runtime_readiness or validate_runtime_configuration(
        settings=settings,
        config_root=options.config_root,
    )
    workflow_context = _resolve_workflow_context(
        settings=settings,
        resume=options.resume,
        fresh_run=options.fresh_run,
    )

    with workflow_file_lock(
        project_root=settings.paths.root,
        workflow_id=workflow_context.workflow_id,
        generation_id=workflow_context.generation_id,
    ):
        from orchestration.bootstrap.application import (
            execute_application,
        )
        from orchestration.bootstrap.container import (
            build_workflow_phase_executor,
        )

        workflow_executor = build_workflow_phase_executor(
            options,
            settings=settings,
            workflow_context=workflow_context,
            execute_crawl_application=execute_application,
            runtime_readiness=readiness,
        )

        async def execute() -> int:
            loop = asyncio.get_running_loop()
            workflow_task = asyncio.current_task()

            if workflow_task is None:
                raise RuntimeError("workflow task is unavailable")

            shutdown_requested = False

            def request_shutdown(sig: signal.Signals) -> None:
                nonlocal shutdown_requested

                if shutdown_requested:
                    return

                shutdown_requested = True
                workflow_executor.logger.info(
                    "workflow_shutdown_requested",
                    signal_name=sig.name,
                )
                workflow_task.cancel()

            install_signal_handlers(
                loop=loop,
                logger=workflow_executor.logger,
                shutdown_callback_factory=lambda sig: (
                    lambda: request_shutdown(sig)
                ),
            )

            try:
                result_code = await workflow_executor.execute()
            except asyncio.CancelledError:
                workflow_executor.logger.warning(
                    "data_workflow_interrupted",
                    exit_code=INTERRUPTED_EXIT_CODE,
                )
                return int(INTERRUPTED_EXIT_CODE)

            if not isinstance(result_code, int):
                raise TypeError(
                    "Workflow executor returned a non-integer exit code: "
                    f"{type(result_code).__name__}"
                )

            return result_code

        return asyncio.run(execute())
