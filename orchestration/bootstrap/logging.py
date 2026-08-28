"""Bootstrap configuration for the project logger factory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from config.path_resolution.project_paths import ProjectPaths
from config.settings.logging import EventRateLimitRulesSettings
from logger.configuration import configure_logging
from logger.factory import ProjectLoggerFactory
from orchestration.errors import LoggingConfigurationError

from .run_context import RunContext, create_run_context

if TYPE_CHECKING:
    from config.settings.root import Settings

DEFAULT_EVENT_RATE_LIMIT_GOVERNANCE: dict[str, EventRateLimitRulesSettings] = {
    "autoscaler_pressure_ratio_calculated": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path",)
    ),
    "autoscaler_effective_max_workers_calculated": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path",)
    ),
    "autoscaler_under_pressure_evaluated": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path", "decision")
    ),
    "autoscaler_underutilized_evaluated": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path", "decision")
    ),
    "autoscaler_decision_state_updated": EventRateLimitRulesSettings(
        min_interval_sec=3.0,
        field_names=("component_path", "pressure_state_reason"),
    ),
    "autoscaler_guards_evaluated": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path",)
    ),
    "autoscaler_pause_guard_not_triggered": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path",)
    ),
    "autoscaler_stop_guard_not_triggered": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path",)
    ),
    "autoscaler_effective_cap_check": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path", "reason")
    ),
    "autoscaler_scale_up_check": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path", "reason")
    ),
    "autoscaler_scale_down_check": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path", "reason")
    ),
    "autoscaler_tick": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path", "action")
    ),
    "autoscaler_snapshot_committed": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path", "action")
    ),
    "rate_limiter_slot_reserved": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path", "host")
    ),
    "rate_limiter_sleep": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path", "host")
    ),
    "request_user_agent_resolved": EventRateLimitRulesSettings(
        min_interval_sec=3.0,
        field_names=("component_path", "host_profile", "selection_strategy"),
    ),
    "request_accept_encoding_built": EventRateLimitRulesSettings(
        min_interval_sec=3.0, field_names=("component_path", "accept_encoding")
    ),
    "request_headers_built": EventRateLimitRulesSettings(
        min_interval_sec=3.0,
        field_names=("component_path", "host", "profile_name", "host_profile"),
    ),
}


def merge_default_event_governance(
    configured: Mapping[str, EventRateLimitRulesSettings] | None,
) -> dict[str, EventRateLimitRulesSettings]:
    """Merge caller configuration over immutable project defaults."""
    merged = dict(DEFAULT_EVENT_RATE_LIMIT_GOVERNANCE)
    if configured:
        merged.update(dict(configured))
    return merged


def build_logger_factory(
    *,
    settings: Settings,
    context: RunContext | None = None,
    configure: bool = True,
) -> ProjectLoggerFactory:
    """Configure logging for one runtime identity."""

    resolved_context = context or create_run_context(
        stage="runtime",
    )

    try:
        logging_settings = settings.logging.model_copy(
            deep=True,
        )

        base_log_fields = dict(logging_settings.base_log_fields or {})
        base_log_fields["workflow_id"] = resolved_context.workflow_id
        base_log_fields["generation_id"] = resolved_context.generation_id
        base_log_fields["root_run_id"] = resolved_context.root_run_id
        base_log_fields["run_id"] = resolved_context.run_id
        base_log_fields["crawl_session_id"] = resolved_context.crawl_session_id
        base_log_fields["stage"] = resolved_context.stage
        base_log_fields["phase"] = resolved_context.stage

        if resolved_context.parent_run_id is not None:
            base_log_fields["parent_run_id"] = resolved_context.parent_run_id

        event_governance = merge_default_event_governance(
            logging_settings.event_rate_limit_governance
        )

        workflow_log_path: str | None = None

        if logging_settings.file_path is not None:
            resolved_path = ProjectPaths(
                project_root=settings.paths.root,
            ).resolve(
                logging_settings.file_path,
            )

            if resolved_context.workflow_id in resolved_path.parts:
                workflow_log_path = str(resolved_path)
            else:
                workflow_log_path = str(
                    ProjectPaths(
                        project_root=resolved_path.parent,
                    ).resolve(
                        Path(resolved_context.workflow_id) / resolved_path.name
                    )
                )

        runtime_logging_settings = logging_settings.model_copy(
            update={
                "base_log_fields": base_log_fields,
                "file_path": workflow_log_path,
                "event_rate_limit_governance": (event_governance),
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
