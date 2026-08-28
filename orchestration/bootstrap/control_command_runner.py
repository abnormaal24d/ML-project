"""Execute crawler runtime control commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from config.path_resolution.project_paths import ProjectPaths
from config.path_resolution.workflow_artifact_paths import (
    ArtifactPathRegistry,
)
from crawler.runtime.control.crawler_control_directory import (
    CrawlerControlDirectory,
)
from orchestration.bootstrap.application import (
    FAILURE_EXIT_CODE,
    SUCCESS_EXIT_CODE,
)
from orchestration.bootstrap.run_context import create_run_context

if TYPE_CHECKING:
    from config.settings.root import Settings
    from orchestration.cli.argument_parser import ControlAction


def execute_control_command(
    *,
    action: ControlAction,
    settings: Settings,
) -> int:
    """Execute one crawler pause, resume, stop, or status command."""

    from orchestration.bootstrap.logging import build_logger_factory

    logger = build_logger_factory(
        settings=settings,
        context=create_run_context(stage="control"),
    ).get_logger(__name__)

    control_directory = CrawlerControlDirectory(
        settings=settings.crawler,
        project_root=settings.paths.root,
    )

    try:
        match action:
            case "pause":
                control_directory.request_pause()

                logger.info(
                    "crawl_pause_enabled",
                    control_directory=str(control_directory.path()),
                )

            case "resume":
                control_directory.clear_pause()
                control_directory.clear_stop()

                logger.info(
                    "crawl_resume_requested",
                    control_directory=str(control_directory.path()),
                )

            case "stop":
                control_directory.request_stop()

                logger.info(
                    "crawl_stop_requested",
                    control_directory=str(control_directory.path()),
                )

            case "status":
                from datachecker.manifests.crawl_state_manifest import (
                    CrawlStateManifest,
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
                    crawl_state = CrawlStateManifest.from_payload(
                        json.loads(state_path.read_text(encoding="utf-8"))
                    )

                logger.info(
                    "crawl_status",
                    **control_directory.status(
                        crawl_state_status=(
                            crawl_state.status.value
                            if crawl_state is not None
                            else None
                        ),
                        workflow_id=(
                            crawl_state.workflow_id
                            if crawl_state is not None
                            else None
                        ),
                        generation_id=(
                            crawl_state.generation_id
                            if crawl_state is not None
                            else None
                        ),
                        attempt_id=(
                            crawl_state.attempt_id
                            if crawl_state is not None
                            else None
                        ),
                        crawl_state_path=state_path,
                    ),
                )

            case _:
                raise ValueError(f"Unsupported control action: {action}")

        return int(SUCCESS_EXIT_CODE)

    except Exception as exc:
        logger.exception(
            "crawl_control_command_failed",
            action=action,
            error_type=type(exc).__name__,
            error_message=str(exc) or None,
        )
        return int(FAILURE_EXIT_CODE)
