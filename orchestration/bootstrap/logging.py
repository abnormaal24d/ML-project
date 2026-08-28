"""Bootstrap configuration for the project logger factory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from config.path_resolution.project_paths import ProjectPaths
from logger.configuration import configure_logging
from logger.factory import ProjectLoggerFactory
from orchestration.errors import LoggingConfigurationError

from .run_context import RunContext, create_run_context

if TYPE_CHECKING:
    from config.settings.root import Settings


def build_logger_factory(
    *,
    settings: Settings,
    context: RunContext | None = None,
    configure: bool = True,
) -> ProjectLoggerFactory:
    """Configure logging for one runtime identity."""

    resolved_context = context or create_run_context(stage="runtime")

    try:
        logging_settings = settings.logging.model_copy(deep=True)

        base_log_fields = dict(logging_settings.base_log_fields)
        base_log_fields["workflow_id"] = resolved_context.workflow_id
        base_log_fields["generation_id"] = resolved_context.generation_id
        base_log_fields["root_run_id"] = resolved_context.root_run_id
        base_log_fields["run_id"] = resolved_context.run_id
        base_log_fields["crawl_session_id"] = resolved_context.crawl_session_id
        base_log_fields["stage"] = resolved_context.stage
        base_log_fields["phase"] = resolved_context.stage

        if resolved_context.parent_run_id is not None:
            base_log_fields["parent_run_id"] = resolved_context.parent_run_id

        workflow_log_path: str | None = None
        if logging_settings.file_path is not None:
            resolved_path = ProjectPaths(
                project_root=settings.paths.root,
            ).resolve(logging_settings.file_path)

            if resolved_context.workflow_id in resolved_path.parts:
                workflow_log_path = str(resolved_path)
            else:
                workflow_log_path = str(
                    ProjectPaths(project_root=resolved_path.parent).resolve(
                        Path(resolved_context.workflow_id) / resolved_path.name
                    )
                )

        runtime_logging_settings = logging_settings.model_copy(
            update={
                "base_log_fields": base_log_fields,
                "file_path": workflow_log_path,
            },
        )

        if configure:
            configure_logging(runtime_logging_settings)

        return ProjectLoggerFactory(
            root_name=runtime_logging_settings.root_name,
            base_context=base_log_fields,
        )

    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise LoggingConfigurationError(
            str(exc),
            stage="bootstrap",
            component="logger",
            cause=exc,
        ) from exc
